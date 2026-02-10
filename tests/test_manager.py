import json
import warnings

import pytest

from openbatch.manager import BatchJobManager
from openbatch.model import (
    ChatCompletionsRequest,
    EmbeddingInputInstance,
    EmbeddingsRequest,
    Message,
    PromptTemplate,
    PromptTemplateInputInstance,
    ResponsesRequest,
    ReusablePrompt,
)


@pytest.fixture
def temp_batch_file(tmp_path):
    """Provides a temporary file path for batch files."""
    return tmp_path / "test_batch.jsonl"


@pytest.fixture
def manager():
    """Provides a BatchJobManager instance."""
    return BatchJobManager()


@pytest.fixture
def manager_no_ascii():
    """Provides a BatchJobManager instance with ensure_ascii=False."""
    return BatchJobManager(ensure_ascii=False)


class TestBatchJobManagerAdd:
    def test_add_responses_request(self, manager, temp_batch_file):
        request = ResponsesRequest(model="gpt-4", input="Hello world")
        manager.add("test_id", request, temp_batch_file)

        assert temp_batch_file.exists()
        with open(temp_batch_file) as f:
            line = f.readline()
            data = json.loads(line)

        assert data["custom_id"] == "test_id"
        assert data["method"] == "POST"
        assert data["url"] == "/v1/responses"
        assert data["body"]["model"] == "gpt-4"
        assert data["body"]["input"] == "Hello world"

    def test_add_chat_completions_request(self, manager, temp_batch_file):
        request = ChatCompletionsRequest(
            model="gpt-4", messages=[{"role": "user", "content": "Hi"}]
        )
        manager.add("chat_id", request, temp_batch_file)

        assert temp_batch_file.exists()
        with open(temp_batch_file) as f:
            data = json.loads(f.readline())

        assert data["custom_id"] == "chat_id"
        assert data["url"] == "/v1/chat/completions"
        assert data["body"]["messages"][0]["role"] == "user"

    def test_add_embeddings_request(self, manager, temp_batch_file):
        request = EmbeddingsRequest(model="text-embedding-3-small", input="Text to embed")
        manager.add("emb_id", request, temp_batch_file)

        assert temp_batch_file.exists()
        with open(temp_batch_file) as f:
            data = json.loads(f.readline())

        assert data["custom_id"] == "emb_id"
        assert data["url"] == "/v1/embeddings"
        assert data["body"]["input"] == "Text to embed"

    def test_add_multiple_requests(self, manager, temp_batch_file):
        request1 = ResponsesRequest(model="gpt-4", input="First")
        request2 = ResponsesRequest(model="gpt-4", input="Second")

        manager.add("id1", request1, temp_batch_file)
        manager.add("id2", request2, temp_batch_file)

        with open(temp_batch_file) as f:
            lines = f.readlines()

        assert len(lines) == 2
        data1 = json.loads(lines[0])
        data2 = json.loads(lines[1])
        assert data1["custom_id"] == "id1"
        assert data2["custom_id"] == "id2"

    def test_add_responses_request_without_input_or_prompt_raises(self, manager, temp_batch_file):
        request = ResponsesRequest(model="gpt-4")
        with pytest.raises(ValueError, match="must define either an input or a prompt"):
            manager.add("test_id", request, temp_batch_file)

    def test_add_chat_completions_request_without_messages_raises(self, manager, temp_batch_file):
        # messages is required in the Pydantic model, so we create a request
        # and then set messages to None manually
        request = ChatCompletionsRequest(model="gpt-4", messages=[])
        request.messages = None
        with pytest.raises(ValueError, match="must define messages"):
            manager.add("test_id", request, temp_batch_file)

    def test_add_embeddings_request_without_input_raises(self, manager, temp_batch_file):
        # input is required in the Pydantic model, so we create a request
        # and then set input to None manually
        request = EmbeddingsRequest(model="text-embedding-3-small", input="dummy")
        request.input = None
        with pytest.raises(ValueError, match="must define an input"):
            manager.add("test_id", request, temp_batch_file)

    def test_add_creates_parent_directory(self, manager, tmp_path):
        nested_path = tmp_path / "subdir" / "batch.jsonl"
        request = ResponsesRequest(model="gpt-4", input="Test")
        manager.add("test_id", request, nested_path)

        assert nested_path.exists()
        assert nested_path.parent.exists()

    def test_add_with_ensure_ascii_false(self, manager_no_ascii, temp_batch_file):
        request = ResponsesRequest(model="gpt-4", input="Hello 世界")
        manager_no_ascii.add("test_id", request, temp_batch_file)

        with open(temp_batch_file, encoding="utf-8") as f:
            content = f.read()
            data = json.loads(content)

        assert "世界" in content  # Non-ASCII characters preserved
        assert data["body"]["input"] == "Hello 世界"

    def test_add_with_ensure_ascii_true(self, manager, temp_batch_file):
        request = ResponsesRequest(model="gpt-4", input="Hello 世界")
        manager.add("test_id", request, temp_batch_file)

        with open(temp_batch_file, encoding="utf-8") as f:
            raw_content = f.read()

        # ASCII escaped version should not contain the raw unicode characters
        assert "\\u" in raw_content


class TestBatchJobManagerTemplatedInstances:
    def test_add_templated_instances_responses_api(self, manager, temp_batch_file):
        template = PromptTemplate(
            messages=[Message(role="user", content="Product: {product}, Price: {price}")]
        )
        common_request = ResponsesRequest(model="gpt-4", temperature=0.7)
        instances = [
            PromptTemplateInputInstance(
                id="prod_1", prompt_value_mapping={"product": "Laptop", "price": "$1000"}
            ),
            PromptTemplateInputInstance(
                id="prod_2", prompt_value_mapping={"product": "Mouse", "price": "$20"}
            ),
        ]

        manager.add_templated_instances(template, common_request, instances, temp_batch_file)

        with open(temp_batch_file) as f:
            lines = f.readlines()

        assert len(lines) == 2
        data1 = json.loads(lines[0])
        data2 = json.loads(lines[1])

        assert data1["custom_id"] == "prod_1"
        assert "Laptop" in str(data1["body"]["input"])
        assert data2["custom_id"] == "prod_2"
        assert "Mouse" in str(data2["body"]["input"])

    def test_add_templated_instances_chat_completions_api(self, manager, temp_batch_file):
        template = PromptTemplate(
            messages=[
                Message(role="system", content="You are a {role}"),
                Message(role="user", content="{question}"),
            ]
        )
        common_request = ChatCompletionsRequest(model="gpt-4", temperature=0.5)
        instances = [
            PromptTemplateInputInstance(
                id="q1",
                prompt_value_mapping={"role": "teacher", "question": "What is math?"},
            ),
            PromptTemplateInputInstance(
                id="q2", prompt_value_mapping={"role": "chef", "question": "How to cook?"}
            ),
        ]

        manager.add_templated_instances(template, common_request, instances, temp_batch_file)

        with open(temp_batch_file) as f:
            lines = f.readlines()

        assert len(lines) == 2
        data1 = json.loads(lines[0])

        assert data1["body"]["messages"][0]["content"] == "You are a teacher"
        assert data1["body"]["messages"][1]["content"] == "What is math?"

    def test_add_templated_instances_with_reusable_prompt(self, manager, temp_batch_file):
        reusable_prompt = ReusablePrompt(id="prompt_123", version="v1", variables={})
        common_request = ResponsesRequest(model="gpt-4")
        instances = [
            PromptTemplateInputInstance(id="inst_1", prompt_value_mapping={"var": "value"})
        ]

        manager.add_templated_instances(reusable_prompt, common_request, instances, temp_batch_file)

        with open(temp_batch_file) as f:
            data = json.loads(f.readline())

        assert data["body"]["prompt"]["id"] == "prompt_123"
        assert data["body"]["prompt"]["version"] == "v1"
        assert data["body"]["prompt"]["variables"]["var"] == "value"

    def test_add_templated_instances_with_instance_options(self, manager, temp_batch_file):
        template = PromptTemplate(messages=[Message(role="user", content="{text}")])
        common_request = ResponsesRequest(model="gpt-4", temperature=0.7)
        instances = [
            PromptTemplateInputInstance(
                id="inst_1",
                prompt_value_mapping={"text": "Hello"},
                instance_request_options={"temperature": 0.9},
            ),
            PromptTemplateInputInstance(id="inst_2", prompt_value_mapping={"text": "World"}),
        ]

        manager.add_templated_instances(template, common_request, instances, temp_batch_file)

        with open(temp_batch_file) as f:
            lines = f.readlines()

        data1 = json.loads(lines[0])
        data2 = json.loads(lines[1])

        # First instance should override temperature
        assert data1["body"]["temperature"] == 0.9
        # Second instance should use common temperature
        assert data2["body"]["temperature"] == 0.7

    def test_add_templated_instances_with_embeddings_raises(self, manager, temp_batch_file):
        template = PromptTemplate(messages=[Message(role="user", content="Test")])
        common_request = EmbeddingsRequest(model="text-embedding-3-small", input="dummy")
        instances = [
            PromptTemplateInputInstance(id="inst_1", prompt_value_mapping={"text": "Test"})
        ]

        with pytest.raises(ValueError, match="Embeddings API is not supported"):
            manager.add_templated_instances(template, common_request, instances, temp_batch_file)

    def test_add_templated_instances_reusable_prompt_with_chat_raises(
        self, manager, temp_batch_file
    ):
        reusable_prompt = ReusablePrompt(id="prompt_123", version="v1", variables={})
        common_request = ChatCompletionsRequest(model="gpt-4", messages=[])
        instances = [
            PromptTemplateInputInstance(id="inst_1", prompt_value_mapping={"var": "value"})
        ]

        with pytest.raises(ValueError, match="Reusable prompts can only be used"):
            manager.add_templated_instances(
                reusable_prompt, common_request, instances, temp_batch_file
            )

    def test_add_templated_instances_appending_warning(self, manager, temp_batch_file):
        # Create the file first
        temp_batch_file.write_text("existing content\n")

        template = PromptTemplate(messages=[Message(role="user", content="Test")])
        common_request = ResponsesRequest(model="gpt-4")
        instances = [PromptTemplateInputInstance(id="inst_1", prompt_value_mapping={})]

        # Should warn when appending to existing file
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            manager.add_templated_instances(template, common_request, instances, temp_batch_file)
            assert len(w) == 1
            assert "already exists" in str(w[0].message)

    def test_add_templated_instances_suppress_warnings(self, manager, temp_batch_file):
        temp_batch_file.write_text("existing content\n")

        template = PromptTemplate(messages=[Message(role="user", content="Test")])
        common_request = ResponsesRequest(model="gpt-4")
        instances = [PromptTemplateInputInstance(id="inst_1", prompt_value_mapping={})]

        # Should not warn when suppress_warnings=True
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            manager.add_templated_instances(
                template,
                common_request,
                instances,
                temp_batch_file,
                suppress_warnings=True,
            )
            assert len(w) == 0


class TestBatchJobManagerEmbeddingRequests:
    def test_add_embedding_requests(self, manager, temp_batch_file):
        common_request = EmbeddingsRequest(model="text-embedding-3-small", dimensions=512)
        inputs = [
            EmbeddingInputInstance(id="emb_1", input="First text"),
            EmbeddingInputInstance(id="emb_2", input="Second text"),
        ]

        manager.add_embedding_requests(inputs, common_request, temp_batch_file)

        with open(temp_batch_file) as f:
            lines = f.readlines()

        assert len(lines) == 2
        data1 = json.loads(lines[0])
        data2 = json.loads(lines[1])

        assert data1["custom_id"] == "emb_1"
        assert data1["body"]["input"] == "First text"
        assert data1["body"]["dimensions"] == 512

        assert data2["custom_id"] == "emb_2"
        assert data2["body"]["input"] == "Second text"

    def test_add_embedding_requests_with_list_input(self, manager, temp_batch_file):
        common_request = EmbeddingsRequest(model="text-embedding-3-small")
        inputs = [
            EmbeddingInputInstance(id="emb_1", input=["Text 1", "Text 2"]),
        ]

        manager.add_embedding_requests(inputs, common_request, temp_batch_file)

        with open(temp_batch_file) as f:
            data = json.loads(f.readline())

        assert isinstance(data["body"]["input"], list)
        assert len(data["body"]["input"]) == 2

    def test_add_embedding_requests_with_instance_options(self, manager, temp_batch_file):
        common_request = EmbeddingsRequest(model="text-embedding-3-small", dimensions=512)
        inputs = [
            EmbeddingInputInstance(
                id="emb_1",
                input="First",
                instance_request_options={"dimensions": 256},
            ),
            EmbeddingInputInstance(id="emb_2", input="Second"),
        ]

        manager.add_embedding_requests(inputs, common_request, temp_batch_file)

        with open(temp_batch_file) as f:
            lines = f.readlines()

        data1 = json.loads(lines[0])
        data2 = json.loads(lines[1])

        # First instance should override dimensions
        assert data1["body"]["dimensions"] == 256
        # Second instance should use common dimensions
        assert data2["body"]["dimensions"] == 512

    def test_add_embedding_requests_creates_parent_directory(self, manager, tmp_path):
        nested_path = tmp_path / "subdir" / "embeddings.jsonl"
        common_request = EmbeddingsRequest(model="text-embedding-3-small")
        inputs = [EmbeddingInputInstance(id="emb_1", input="Test")]

        manager.add_embedding_requests(inputs, common_request, nested_path)

        assert nested_path.exists()
        assert nested_path.parent.exists()
