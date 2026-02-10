from pydantic import BaseModel, Field

from openbatch.model import (
    ChatCompletionsAPIStrategy,
    ChatCompletionsRequest,
    EmbeddingInputInstance,
    EmbeddingsAPIStrategy,
    EmbeddingsRequest,
    Message,
    MessagesInputInstance,
    PromptTemplate,
    PromptTemplateInputInstance,
    ReasoningConfig,
    ResponsesAPIStrategy,
    ResponsesRequest,
    ReusablePrompt,
)


class TestMessage:
    def test_message_creation(self):
        msg = Message(role="user", content="Hello")
        assert msg.role == "user"
        assert msg.content == "Hello"

    def test_message_serialize(self):
        msg = Message(role="system", content="You are helpful")
        serialized = msg.serialize()
        assert serialized == {"role": "system", "content": "You are helpful"}


class TestPromptTemplate:
    def test_prompt_template_creation(self):
        template = PromptTemplate(
            messages=[
                Message(role="system", content="You are a {role}"),
                Message(role="user", content="Help me with {task}"),
            ]
        )
        assert len(template.messages) == 2

    def test_prompt_template_format(self):
        template = PromptTemplate(
            messages=[
                Message(role="system", content="You are a {role}"),
                Message(role="user", content="Help me with {task}"),
            ]
        )
        formatted = template.format(role="assistant", task="coding")
        assert len(formatted) == 2
        assert formatted[0].content == "You are a assistant"
        assert formatted[1].content == "Help me with coding"

    def test_prompt_template_format_multiple_placeholders(self):
        template = PromptTemplate(
            messages=[
                Message(
                    role="user",
                    content="Product: {product}, Price: {price}, Category: {category}",
                )
            ]
        )
        formatted = template.format(product="Laptop", price="$1000", category="Electronics")
        assert formatted[0].content == "Product: Laptop, Price: $1000, Category: Electronics"


class TestReusablePrompt:
    def test_reusable_prompt_creation(self):
        prompt = ReusablePrompt(id="prompt_123", version="v1", variables={"name": "John"})
        assert prompt.id == "prompt_123"
        assert prompt.version == "v1"
        assert prompt.variables == {"name": "John"}


class TestReasoningConfig:
    def test_reasoning_config_default(self):
        config = ReasoningConfig()
        assert config.effort == "medium"
        assert config.summary is None

    def test_reasoning_config_custom(self):
        config = ReasoningConfig(effort="high", summary="detailed")
        assert config.effort == "high"
        assert config.summary == "detailed"


class TestInputInstances:
    def test_prompt_template_input_instance(self):
        instance = PromptTemplateInputInstance(
            id="inst_1",
            prompt_value_mapping={"name": "Alice", "age": "30"},
            instance_request_options={"temperature": 0.5},
        )
        assert instance.id == "inst_1"
        assert instance.prompt_value_mapping == {"name": "Alice", "age": "30"}
        assert instance.instance_request_options == {"temperature": 0.5}

    def test_messages_input_instance(self):
        messages = [Message(role="user", content="Hello")]
        instance = MessagesInputInstance(id="inst_2", messages=messages)
        assert instance.id == "inst_2"
        assert len(instance.messages) == 1

    def test_embedding_input_instance(self):
        instance = EmbeddingInputInstance(id="emb_1", input="Text to embed")
        assert instance.id == "emb_1"
        assert instance.input == "Text to embed"

    def test_embedding_input_instance_list(self):
        instance = EmbeddingInputInstance(id="emb_2", input=["Text 1", "Text 2"])
        assert instance.id == "emb_2"
        assert isinstance(instance.input, list)
        assert len(instance.input) == 2


class TestAPIStrategies:
    def test_responses_api_strategy(self):
        strategy = ResponsesAPIStrategy()
        assert strategy.url == "/v1/responses"
        request = strategy.create_request("test_id", {"model": "gpt-4"})
        assert request["custom_id"] == "test_id"
        assert request["method"] == "POST"
        assert request["url"] == "/v1/responses"
        assert request["body"] == {"model": "gpt-4"}

    def test_chat_completions_api_strategy(self):
        strategy = ChatCompletionsAPIStrategy()
        assert strategy.url == "/v1/chat/completions"

    def test_embeddings_api_strategy(self):
        strategy = EmbeddingsAPIStrategy()
        assert strategy.url == "/v1/embeddings"


class TestResponsesRequest:
    def test_responses_request_minimal(self):
        request = ResponsesRequest(model="gpt-4")
        assert request.model == "gpt-4"
        assert request.input is None

    def test_responses_request_with_input(self):
        request = ResponsesRequest(model="gpt-4", input="Hello world")
        assert request.input == "Hello world"

    def test_responses_request_to_dict(self):
        request = ResponsesRequest(model="gpt-4", input="Hello", temperature=0.7)
        result = request.to_dict()
        assert result["model"] == "gpt-4"
        assert result["input"] == "Hello"
        assert result["temperature"] == 0.7

    def test_responses_request_exclude_none(self):
        request = ResponsesRequest(model="gpt-4", input="Hello")
        result = request.to_dict()
        assert "temperature" not in result
        assert "max_output_tokens" not in result

    def test_responses_request_set_input_messages(self):
        request = ResponsesRequest(model="gpt-4")
        messages = [Message(role="user", content="Hello")]
        request.set_input_messages(messages)
        assert request.input == [{"role": "user", "content": "Hello"}]

    def test_responses_request_set_output_structure(self):
        class TestOutput(BaseModel):
            name: str
            age: int

        request = ResponsesRequest(model="gpt-4")
        request.set_output_structure(TestOutput)
        assert request.text is not None
        assert "format" in request.text
        assert request.text["format"]["type"] == "json_schema"
        assert request.text["format"]["name"] == "TestOutput"
        assert request.text["format"]["strict"] is True

    def test_responses_request_with_reasoning(self):
        request = ResponsesRequest(
            model="gpt-4", reasoning=ReasoningConfig(effort="high", summary="detailed")
        )
        assert request.reasoning.effort == "high"
        assert request.reasoning.summary == "detailed"


class TestChatCompletionsRequest:
    def test_chat_completions_request_minimal(self):
        messages = [{"role": "user", "content": "Hello"}]
        request = ChatCompletionsRequest(model="gpt-4", messages=messages)
        assert request.model == "gpt-4"
        assert len(request.messages) == 1

    def test_chat_completions_request_set_input_messages(self):
        request = ChatCompletionsRequest(model="gpt-4", messages=[])
        messages = [Message(role="user", content="Hi")]
        request.set_input_messages(messages)
        assert request.messages == [{"role": "user", "content": "Hi"}]

    def test_chat_completions_request_set_output_structure(self):
        class TestResponse(BaseModel):
            answer: str = Field(description="The answer")

        request = ChatCompletionsRequest(model="gpt-4", messages=[])
        request.set_output_structure(TestResponse)
        assert request.response_format is not None
        assert "format" in request.response_format
        assert request.response_format["format"]["name"] == "TestResponse"

    def test_chat_completions_request_with_temperature(self):
        request = ChatCompletionsRequest(
            model="gpt-4", messages=[{"role": "user", "content": "Hi"}], temperature=0.9
        )
        assert request.temperature == 0.9

    def test_chat_completions_request_to_dict(self):
        request = ChatCompletionsRequest(
            model="gpt-4",
            messages=[{"role": "user", "content": "Hi"}],
            temperature=0.5,
            max_completion_tokens=100,
        )
        result = request.to_dict()
        assert result["model"] == "gpt-4"
        assert result["temperature"] == 0.5
        assert result["max_completion_tokens"] == 100


class TestEmbeddingsRequest:
    def test_embeddings_request_with_string(self):
        request = EmbeddingsRequest(model="text-embedding-3-small", input="Hello")
        assert request.model == "text-embedding-3-small"
        assert request.input == "Hello"

    def test_embeddings_request_with_list(self):
        request = EmbeddingsRequest(model="text-embedding-3-small", input=["Hello", "World"])
        assert isinstance(request.input, list)
        assert len(request.input) == 2

    def test_embeddings_request_set_input(self):
        request = EmbeddingsRequest(model="text-embedding-3-small", input="test")
        request.set_input("New text")
        assert request.input == "New text"

    def test_embeddings_request_with_dimensions(self):
        request = EmbeddingsRequest(model="text-embedding-3-small", input="test", dimensions=512)
        assert request.dimensions == 512

    def test_embeddings_request_to_dict(self):
        request = EmbeddingsRequest(model="text-embedding-3-small", input="test", dimensions=256)
        result = request.to_dict()
        assert result["model"] == "text-embedding-3-small"
        assert result["input"] == "test"
        assert result["dimensions"] == 256
