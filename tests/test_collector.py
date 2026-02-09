import json
import pytest
from pathlib import Path
from pydantic import BaseModel, Field
from openbatch.collector import BatchCollector, Responses, ChatCompletions, Embeddings
from openbatch.model import ReasoningConfig


@pytest.fixture
def temp_batch_file(tmp_path):
    """Provides a temporary file path for batch files."""
    return tmp_path / "test_batch.jsonl"


class TestResponses:
    def test_responses_create(self, temp_batch_file):
        responses = Responses(temp_batch_file)
        responses.create(
            custom_id="req_1",
            model="gpt-4",
            input="What is Python?",
            instructions="You are a helpful assistant",
            max_output_tokens=100,
        )

        assert temp_batch_file.exists()
        with open(temp_batch_file, "r") as f:
            data = json.loads(f.readline())

        assert data["custom_id"] == "req_1"
        assert data["url"] == "/v1/responses"
        assert data["body"]["model"] == "gpt-4"
        assert data["body"]["input"] == "What is Python?"
        assert data["body"]["instructions"] == "You are a helpful assistant"
        assert data["body"]["max_output_tokens"] == 100

    def test_responses_parse_without_format(self, temp_batch_file):
        responses = Responses(temp_batch_file)
        responses.parse(
            custom_id="req_2",
            model="gpt-4",
            input="Analyze this text",
            instructions="Be concise",
        )

        with open(temp_batch_file, "r") as f:
            data = json.loads(f.readline())

        assert data["custom_id"] == "req_2"
        assert data["body"]["model"] == "gpt-4"
        assert "text" not in data["body"]  # No text format specified

    def test_responses_parse_with_format(self, temp_batch_file):
        class Analysis(BaseModel):
            summary: str = Field(description="Brief summary")
            sentiment: str = Field(description="Sentiment analysis")

        responses = Responses(temp_batch_file)
        responses.parse(
            custom_id="req_3",
            model="gpt-4",
            text_format=Analysis,
            input="Great product!",
            instructions="Analyze sentiment",
        )

        with open(temp_batch_file, "r") as f:
            data = json.loads(f.readline())

        assert data["custom_id"] == "req_3"
        assert "text" in data["body"]
        assert data["body"]["text"]["format"]["type"] == "json_schema"
        assert data["body"]["text"]["format"]["name"] == "Analysis"
        assert data["body"]["text"]["format"]["strict"] is True

    def test_responses_with_reasoning_config(self, temp_batch_file):
        responses = Responses(temp_batch_file)
        responses.create(
            custom_id="req_4",
            model="gpt-5-mini",
            input="Complex problem",
            reasoning=ReasoningConfig(effort="high", summary="detailed"),
        )

        with open(temp_batch_file, "r") as f:
            data = json.loads(f.readline())

        assert data["body"]["reasoning"]["effort"] == "high"
        assert data["body"]["reasoning"]["summary"] == "detailed"

    def test_responses_multiple_requests(self, temp_batch_file):
        responses = Responses(temp_batch_file)
        responses.create(custom_id="req_1", model="gpt-4", input="First")
        responses.create(custom_id="req_2", model="gpt-4", input="Second")

        with open(temp_batch_file, "r") as f:
            lines = f.readlines()

        assert len(lines) == 2
        data1 = json.loads(lines[0])
        data2 = json.loads(lines[1])
        assert data1["custom_id"] == "req_1"
        assert data2["custom_id"] == "req_2"


class TestChatCompletions:
    def test_chat_completions_create(self, temp_batch_file):
        chat = ChatCompletions(temp_batch_file)
        chat.create(
            custom_id="chat_1",
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are helpful"},
                {"role": "user", "content": "Hello"},
            ],
            temperature=0.7,
        )

        assert temp_batch_file.exists()
        with open(temp_batch_file, "r") as f:
            data = json.loads(f.readline())

        assert data["custom_id"] == "chat_1"
        assert data["url"] == "/v1/chat/completions"
        assert data["body"]["model"] == "gpt-4"
        assert len(data["body"]["messages"]) == 2
        assert data["body"]["temperature"] == 0.7

    def test_chat_completions_parse_without_format(self, temp_batch_file):
        chat = ChatCompletions(temp_batch_file)
        chat.parse(
            custom_id="chat_2",
            model="gpt-4",
            messages=[{"role": "user", "content": "Hi"}],
        )

        with open(temp_batch_file, "r") as f:
            data = json.loads(f.readline())

        assert data["custom_id"] == "chat_2"
        assert "response_format" not in data["body"]

    def test_chat_completions_parse_with_format(self, temp_batch_file):
        class Response(BaseModel):
            answer: str
            confidence: float

        chat = ChatCompletions(temp_batch_file)
        chat.parse(
            custom_id="chat_3",
            model="gpt-4",
            response_format=Response,
            messages=[{"role": "user", "content": "What is 2+2?"}],
        )

        with open(temp_batch_file, "r") as f:
            data = json.loads(f.readline())

        assert data["custom_id"] == "chat_3"
        assert "response_format" in data["body"]
        assert data["body"]["response_format"]["format"]["name"] == "Response"
        assert data["body"]["response_format"]["format"]["strict"] is True

    def test_chat_completions_with_reasoning_effort(self, temp_batch_file):
        chat = ChatCompletions(temp_batch_file)
        chat.create(
            custom_id="chat_4",
            model="o1-mini",
            messages=[{"role": "user", "content": "Complex question"}],
            reasoning_effort="high",
        )

        with open(temp_batch_file, "r") as f:
            data = json.loads(f.readline())

        assert data["body"]["reasoning_effort"] == "high"

    def test_chat_completions_multiple_requests(self, temp_batch_file):
        chat = ChatCompletions(temp_batch_file)
        chat.create(
            custom_id="chat_1", model="gpt-4", messages=[{"role": "user", "content": "Hi"}]
        )
        chat.create(
            custom_id="chat_2",
            model="gpt-4",
            messages=[{"role": "user", "content": "Bye"}],
        )

        with open(temp_batch_file, "r") as f:
            lines = f.readlines()

        assert len(lines) == 2


class TestEmbeddings:
    def test_embeddings_create_single_input(self, temp_batch_file):
        embeddings = Embeddings(temp_batch_file)
        embeddings.create(
            custom_id="emb_1",
            model="text-embedding-3-small",
            inp="Text to embed",
        )

        assert temp_batch_file.exists()
        with open(temp_batch_file, "r") as f:
            data = json.loads(f.readline())

        assert data["custom_id"] == "emb_1"
        assert data["url"] == "/v1/embeddings"
        assert data["body"]["model"] == "text-embedding-3-small"
        assert data["body"]["input"] == "Text to embed"

    def test_embeddings_create_list_input(self, temp_batch_file):
        embeddings = Embeddings(temp_batch_file)
        embeddings.create(
            custom_id="emb_2",
            model="text-embedding-3-small",
            inp=["Text 1", "Text 2", "Text 3"],
        )

        with open(temp_batch_file, "r") as f:
            data = json.loads(f.readline())

        assert isinstance(data["body"]["input"], list)
        assert len(data["body"]["input"]) == 3

    def test_embeddings_with_dimensions(self, temp_batch_file):
        embeddings = Embeddings(temp_batch_file)
        embeddings.create(
            custom_id="emb_3",
            model="text-embedding-3-small",
            inp="Test",
            dimensions=512,
        )

        with open(temp_batch_file, "r") as f:
            data = json.loads(f.readline())

        assert data["body"]["dimensions"] == 512

    def test_embeddings_multiple_requests(self, temp_batch_file):
        embeddings = Embeddings(temp_batch_file)
        embeddings.create(custom_id="emb_1", model="text-embedding-3-small", inp="First")
        embeddings.create(custom_id="emb_2", model="text-embedding-3-small", inp="Second")

        with open(temp_batch_file, "r") as f:
            lines = f.readlines()

        assert len(lines) == 2


class TestBatchCollector:
    def test_batch_collector_initialization(self, temp_batch_file):
        collector = BatchCollector(temp_batch_file)
        assert isinstance(collector.responses, Responses)
        assert isinstance(collector.chat.completions, ChatCompletions)
        assert isinstance(collector.embeddings, Embeddings)

    def test_batch_collector_responses_api(self, temp_batch_file):
        collector = BatchCollector(temp_batch_file)
        collector.responses.create(
            custom_id="req_1",
            model="gpt-4",
            input="Hello",
        )

        assert temp_batch_file.exists()
        with open(temp_batch_file, "r") as f:
            data = json.loads(f.readline())

        assert data["url"] == "/v1/responses"

    def test_batch_collector_chat_completions_api(self, temp_batch_file):
        collector = BatchCollector(temp_batch_file)
        collector.chat.completions.create(
            custom_id="chat_1",
            model="gpt-4",
            messages=[{"role": "user", "content": "Hi"}],
        )

        assert temp_batch_file.exists()
        with open(temp_batch_file, "r") as f:
            data = json.loads(f.readline())

        assert data["url"] == "/v1/chat/completions"

    def test_batch_collector_embeddings_api(self, temp_batch_file):
        collector = BatchCollector(temp_batch_file)
        collector.embeddings.create(
            custom_id="emb_1",
            model="text-embedding-3-small",
            inp="Text",
        )

        assert temp_batch_file.exists()
        with open(temp_batch_file, "r") as f:
            data = json.loads(f.readline())

        assert data["url"] == "/v1/embeddings"

    def test_batch_collector_mixed_apis_in_sequence(self, tmp_path):
        # Demonstrate that different endpoints need different files
        responses_file = tmp_path / "responses.jsonl"
        chat_file = tmp_path / "chat.jsonl"
        embeddings_file = tmp_path / "embeddings.jsonl"

        responses_collector = BatchCollector(responses_file)
        responses_collector.responses.create(
            custom_id="req_1", model="gpt-4", input="Test"
        )

        chat_collector = BatchCollector(chat_file)
        chat_collector.chat.completions.create(
            custom_id="chat_1",
            model="gpt-4",
            messages=[{"role": "user", "content": "Test"}],
        )

        embeddings_collector = BatchCollector(embeddings_file)
        embeddings_collector.embeddings.create(
            custom_id="emb_1", model="text-embedding-3-small", inp="Test"
        )

        assert responses_file.exists()
        assert chat_file.exists()
        assert embeddings_file.exists()

    def test_batch_collector_responses_parse_structured_output(self, temp_batch_file):
        class TaskAnalysis(BaseModel):
            task_type: str
            complexity: str
            estimated_time: int

        collector = BatchCollector(temp_batch_file)
        collector.responses.parse(
            custom_id="analysis_1",
            model="gpt-4",
            text_format=TaskAnalysis,
            input="Analyze this task: Build a web scraper",
            instructions="Provide structured analysis",
        )

        with open(temp_batch_file, "r") as f:
            data = json.loads(f.readline())

        assert "text" in data["body"]
        assert data["body"]["text"]["format"]["name"] == "TaskAnalysis"
        assert "task_type" in str(data["body"]["text"]["format"]["schema"])

    def test_batch_collector_chat_parse_structured_output(self, temp_batch_file):
        class CodeReview(BaseModel):
            issues: list[str]
            suggestions: list[str]
            rating: int

        collector = BatchCollector(temp_batch_file)
        collector.chat.completions.parse(
            custom_id="review_1",
            model="gpt-4",
            response_format=CodeReview,
            messages=[
                {"role": "system", "content": "You are a code reviewer"},
                {"role": "user", "content": "Review this code: def foo(): pass"},
            ],
        )

        with open(temp_batch_file, "r") as f:
            data = json.loads(f.readline())

        assert "response_format" in data["body"]
        assert data["body"]["response_format"]["format"]["name"] == "CodeReview"
