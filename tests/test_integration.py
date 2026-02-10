"""Integration tests that verify end-to-end workflows."""

import json

import pytest
from pydantic import BaseModel, Field

from openbatch import (
    BatchCollector,
    BatchJobManager,
    EmbeddingInputInstance,
    EmbeddingsRequest,
    Message,
    PromptTemplate,
    PromptTemplateInputInstance,
    ReasoningConfig,
    ResponsesRequest,
)


@pytest.fixture
def temp_dir(tmp_path):
    """Provides a temporary directory for test files."""
    return tmp_path


class TestEndToEndBatchCreation:
    """Test complete workflows from creation to file output."""

    def test_batch_collector_complete_workflow(self, temp_dir):
        """Test BatchCollector API for all three endpoint types."""
        # Separate files for each API type (as required by OpenAI)
        responses_file = temp_dir / "responses.jsonl"
        chat_file = temp_dir / "chat.jsonl"
        embeddings_file = temp_dir / "embeddings.jsonl"

        # Responses API
        responses_collector = BatchCollector(responses_file)
        responses_collector.responses.create(
            custom_id="resp_1",
            model="gpt-4",
            input="What is machine learning?",
            instructions="Be concise",
            max_output_tokens=100,
        )

        # Chat Completions API
        chat_collector = BatchCollector(chat_file)
        chat_collector.chat.completions.create(
            custom_id="chat_1",
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful assistant"},
                {"role": "user", "content": "Explain quantum computing"},
            ],
            temperature=0.7,
        )

        # Embeddings API
        embeddings_collector = BatchCollector(embeddings_file)
        embeddings_collector.embeddings.create(
            custom_id="emb_1",
            model="text-embedding-3-small",
            inp="Machine learning is a subset of artificial intelligence",
        )

        # Verify all files exist and contain valid JSON
        assert responses_file.exists()
        assert chat_file.exists()
        assert embeddings_file.exists()

        with open(responses_file) as f:
            resp_data = json.loads(f.readline())
            assert resp_data["url"] == "/v1/responses"

        with open(chat_file) as f:
            chat_data = json.loads(f.readline())
            assert chat_data["url"] == "/v1/chat/completions"

        with open(embeddings_file) as f:
            emb_data = json.loads(f.readline())
            assert emb_data["url"] == "/v1/embeddings"

    def test_batch_job_manager_templated_workflow(self, temp_dir):
        """Test BatchJobManager with prompt templates for bulk generation."""
        batch_file = temp_dir / "marketing_batch.jsonl"

        # Setup
        template = PromptTemplate(
            messages=[
                Message(
                    role="system",
                    content="You are a marketing copywriter. Generate a catchy, two-sentence description.",
                ),
                Message(
                    role="user",
                    content="Product: {product_name}, Features: {features}",
                ),
            ]
        )

        common_config = ResponsesRequest(model="gpt-4-mini", temperature=0.8, max_output_tokens=100)

        products = [
            PromptTemplateInputInstance(
                id="prod_001",
                prompt_value_mapping={
                    "product_name": "AeroGlide Drone",
                    "features": "4K camera, 30-min flight",
                },
            ),
            PromptTemplateInputInstance(
                id="prod_002",
                prompt_value_mapping={
                    "product_name": "HydroPure Bottle",
                    "features": "Self-cleaning, insulated steel",
                },
            ),
            PromptTemplateInputInstance(
                id="prod_003",
                prompt_value_mapping={
                    "product_name": "SmartDesk Pro",
                    "features": "Height adjustable, USB charging",
                },
                instance_request_options={"temperature": 0.5},  # Override for this instance
            ),
        ]

        # Generate batch
        manager = BatchJobManager()
        manager.add_templated_instances(
            prompt=template,
            common_request=common_config,
            input_instances=products,
            save_file_path=batch_file,
        )

        # Verify
        assert batch_file.exists()
        with open(batch_file) as f:
            lines = f.readlines()

        assert len(lines) == 3

        # Check first product
        data1 = json.loads(lines[0])
        assert data1["custom_id"] == "prod_001"
        assert "AeroGlide Drone" in str(data1["body"]["input"])
        assert data1["body"]["temperature"] == 0.8

        # Check third product with overridden temperature
        data3 = json.loads(lines[2])
        assert data3["custom_id"] == "prod_003"
        assert data3["body"]["temperature"] == 0.5

    def test_batch_job_manager_embeddings_workflow(self, temp_dir):
        """Test BatchJobManager for bulk embeddings generation."""
        batch_file = temp_dir / "embeddings_batch.jsonl"

        common_config = EmbeddingsRequest(model="text-embedding-3-small", dimensions=512)

        documents = [
            EmbeddingInputInstance(id="doc_1", input="The sky is blue."),
            EmbeddingInputInstance(id="doc_2", input="Grass is green."),
            EmbeddingInputInstance(id="doc_3", input="Water is wet."),
            EmbeddingInputInstance(
                id="doc_4",
                input="Fire is hot.",
                instance_request_options={"dimensions": 256},
            ),
        ]

        manager = BatchJobManager()
        manager.add_embedding_requests(
            inputs=documents, common_request=common_config, save_file_path=batch_file
        )

        assert batch_file.exists()
        with open(batch_file) as f:
            lines = f.readlines()

        assert len(lines) == 4

        data1 = json.loads(lines[0])
        assert data1["body"]["dimensions"] == 512

        data4 = json.loads(lines[3])
        assert data4["body"]["dimensions"] == 256  # Overridden


class TestStructuredOutputWorkflows:
    """Test workflows with structured JSON output."""

    def test_responses_api_with_structured_output(self, temp_dir):
        """Test Responses API with Pydantic model for structured output."""

        class SentimentAnalysis(BaseModel):
            sentiment: str = Field(description="positive, negative, or neutral")
            confidence: float = Field(description="Confidence score 0-1")
            key_phrases: list[str] = Field(description="Important phrases")

        batch_file = temp_dir / "sentiment_batch.jsonl"
        collector = BatchCollector(batch_file)

        texts_to_analyze = [
            "This product exceeded my expectations! Absolutely love it.",
            "Terrible experience. Would not recommend to anyone.",
            "It's okay, nothing special but does the job.",
        ]

        for idx, text in enumerate(texts_to_analyze):
            collector.responses.parse(
                custom_id=f"sentiment_{idx}",
                model="gpt-4",
                text_format=SentimentAnalysis,
                input=text,
                instructions="Analyze the sentiment of the given text",
            )

        assert batch_file.exists()
        with open(batch_file) as f:
            lines = f.readlines()

        assert len(lines) == 3

        # Verify structured output configuration
        data = json.loads(lines[0])
        assert "text" in data["body"]
        assert data["body"]["text"]["format"]["name"] == "SentimentAnalysis"
        assert data["body"]["text"]["format"]["strict"] is True
        schema = data["body"]["text"]["format"]["schema"]
        assert "sentiment" in schema["properties"]
        assert "confidence" in schema["properties"]
        assert "key_phrases" in schema["properties"]

    def test_chat_completions_with_structured_output(self, temp_dir):
        """Test Chat Completions API with structured output."""

        class RecipeExtraction(BaseModel):
            recipe_name: str
            ingredients: list[str]
            steps: list[str]
            prep_time_minutes: int
            difficulty: str

        batch_file = temp_dir / "recipes_batch.jsonl"
        collector = BatchCollector(batch_file)

        recipe_texts = [
            "How to make scrambled eggs: Beat 2 eggs, heat pan, cook for 2 minutes. Takes 5 minutes. Easy.",
            "Chocolate cake recipe: Mix flour, sugar, cocoa. Bake at 350F for 30 minutes. Medium difficulty. Takes 45 minutes.",
        ]

        for idx, text in enumerate(recipe_texts):
            collector.chat.completions.parse(
                custom_id=f"recipe_{idx}",
                model="gpt-4",
                response_format=RecipeExtraction,
                messages=[
                    {
                        "role": "system",
                        "content": "Extract structured recipe information",
                    },
                    {"role": "user", "content": text},
                ],
            )

        with open(batch_file) as f:
            lines = f.readlines()

        assert len(lines) == 2
        data = json.loads(lines[0])
        assert "response_format" in data["body"]
        assert data["body"]["response_format"]["format"]["name"] == "RecipeExtraction"


class TestReasoningModelsWorkflow:
    """Test workflows with reasoning models."""

    def test_responses_api_with_reasoning_config(self, temp_dir):
        """Test Responses API with reasoning configuration."""
        batch_file = temp_dir / "reasoning_batch.jsonl"
        collector = BatchCollector(batch_file)

        complex_problems = [
            {
                "id": "logic_1",
                "problem": "If all A are B, and all B are C, are all A necessarily C?",
                "effort": "high",
            },
            {
                "id": "logic_2",
                "problem": "What is the flaw in this argument: All birds can fly. Penguins are birds. Therefore penguins can fly.",
                "effort": "medium",
            },
        ]

        for item in complex_problems:
            collector.responses.create(
                custom_id=item["id"],
                model="o1-mini",
                input=item["problem"],
                instructions="Analyze the logical structure carefully",
                reasoning=ReasoningConfig(effort=item["effort"], summary="detailed"),
            )

        with open(batch_file) as f:
            lines = f.readlines()

        assert len(lines) == 2
        data1 = json.loads(lines[0])
        assert data1["body"]["reasoning"]["effort"] == "high"
        assert data1["body"]["reasoning"]["summary"] == "detailed"


class TestUnicodeAndSpecialCharacters:
    """Test handling of non-ASCII characters."""

    def test_ensure_ascii_true(self, temp_dir):
        """Test that ensure_ascii=True escapes non-ASCII characters."""
        batch_file = temp_dir / "unicode_escaped.jsonl"
        manager = BatchJobManager(ensure_ascii=True)

        request = ResponsesRequest(model="gpt-4", input="Hello 世界! Привет! مرحبا")
        manager.add("unicode_test", request, batch_file)

        with open(batch_file, encoding="utf-8") as f:
            content = f.read()

        # Should contain escaped unicode
        assert "\\u" in content
        # Raw characters should not be present
        assert "世界" not in content

    def test_ensure_ascii_false(self, temp_dir):
        """Test that ensure_ascii=False preserves non-ASCII characters."""
        batch_file = temp_dir / "unicode_raw.jsonl"
        manager = BatchJobManager(ensure_ascii=False)

        request = ResponsesRequest(model="gpt-4", input="Hello 世界! Привет! مرحبا Emoji: 🚀")
        manager.add("unicode_test", request, batch_file)

        with open(batch_file, encoding="utf-8") as f:
            content = f.read()

        # Raw characters should be present
        assert "世界" in content
        assert "Привет" in content
        assert "مرحبا" in content
        assert "🚀" in content


class TestLargeScaleBatchGeneration:
    """Test generation of large batch files."""

    def test_generate_1000_requests(self, temp_dir):
        """Test generating a batch file with 1000 requests."""
        batch_file = temp_dir / "large_batch.jsonl"

        template = PromptTemplate(messages=[Message(role="user", content="Classify: {text}")])

        common_request = ResponsesRequest(model="gpt-4-mini", max_output_tokens=10)

        # Generate 1000 instances
        instances = [
            PromptTemplateInputInstance(
                id=f"classify_{i:04d}", prompt_value_mapping={"text": f"Sample text {i}"}
            )
            for i in range(1000)
        ]

        manager = BatchJobManager()
        manager.add_templated_instances(
            prompt=template,
            common_request=common_request,
            input_instances=instances,
            save_file_path=batch_file,
        )

        # Verify
        assert batch_file.exists()
        with open(batch_file) as f:
            lines = f.readlines()

        assert len(lines) == 1000

        # Spot check first and last
        first = json.loads(lines[0])
        last = json.loads(lines[999])

        assert first["custom_id"] == "classify_0000"
        assert last["custom_id"] == "classify_0999"
        assert "Sample text 0" in str(first["body"]["input"])
        assert "Sample text 999" in str(last["body"]["input"])
