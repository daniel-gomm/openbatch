# OpenBatch: Simplify OpenAI Batch Job Creation

[](https://www.google.com/search?q=https://badge.fury.io/py/openbatch) **OpenBatch** is a lightweight Python utility designed to streamline the creation of JSONL files for the [OpenAI Batch API](https://platform.openai.com/docs/guides/batch). It provides a type-safe and intuitive interface using Pydantic models to construct requests for the `/v1/responses`, `/v1/chat/completions`, and `/v1/embeddings` endpoints.

The library offers two distinct APIs to fit your workflow:

  * **`BatchCollector`**: A high-level, fluent API that mimics the official `openai` client. It's perfect for adding individual, distinct requests to a batch file with minimal setup.
  * **`BatchJobManager`**: A lower-level API designed for programmatically generating large batches of requests from templates and lists of inputs. It's ideal for scalable tasks like classification, data extraction, or bulk embeddings.

-----

## Installation

```bash
pip install openbatch
```

-----

## Quickstart: The `BatchCollector` API

The `BatchCollector` provides the simplest way to get started. You instantiate it with a file path, and then use its methods to add requests one by one. This example showcases calls to the Responses and Embeddings APIs.

```python
from pydantic import BaseModel, Field
from typing import List
from openbatch import BatchCollector, ReasoningConfig

# Define a Pydantic model for structured JSON output
class LogicalAnalysis(BaseModel):
    premise: str
    conclusion: str
    is_valid: bool = Field(description="Whether the conclusion logically follows from the premise.")

# 1. Initialize the collector with the desired output file path
BATCH_FILE = "my_api_batch.jsonl"
collector = BatchCollector(batch_file_path=BATCH_FILE)

# 2. Add a standard request to the Responses API
collector.responses.create(
    custom_id="request-1-response",
    model="gpt-4o",
    instructions="You are a historian. Provide a concise summary.",
    input="What were the main causes of the French Revolution?",
    max_output_tokens=200
)

# 3. Add a structured request using a reasoning model.
# Note: Reasoning models may not support 'temperature', and it is omitted here.
collector.responses.parse(
    custom_id="request-2-reasoning",
    model="gpt-5-mini",  # Hypothetical reasoning model
    text_format=LogicalAnalysis,
    instructions="Analyze the logical argument provided by the user.",
    input="Premise: All birds can fly. A penguin is a bird. Conclusion: Therefore, a penguin can fly.",
    reasoning=ReasoningConfig(effort="high") # Configure the reasoning effort
)

# 4. Add an Embedding request to showcase API breadth
collector.embeddings.create(
    custom_id="request-3-embedding",
    model="text-embedding-3-small",
    inp="OpenBatch simplifies creating batch jobs."
)

print(f"Batch file '{BATCH_FILE}' created successfully.")
```

-----

## Advanced Usage: The `BatchJobManager` API

For more complex or repetitive tasks, the `BatchJobManager` is the more appropriate tool. It excels at generating thousands of requests from a single template, for any supported API.

### Example 1: Batch Job from a Prompt Template (Responses API)

Imagine you want to generate marketing copy for 10,000 new products. Instead of creating each request manually, you can use a template with the Responses API.

```python
from openbatch import (
    BatchJobManager,
    PromptTemplate,
    Message,
    ResponsesRequest,
    PromptTemplateInputInstance
)

# 1. Define a prompt template with placeholders
copywriting_template = PromptTemplate(
    messages=[
        Message(role="system", content="You are a marketing copywriter. Generate a catchy, two-sentence description."),
        Message(role="user", content="Product: {product_name}, Features: {features}")
    ]
)

# 2. Define the common configuration for all requests
common_request_config = ResponsesRequest(
    model="gpt-4o-mini",
    temperature=0.8,
    max_output_tokens=100
)

# 3. Create a list of input instances
product_instances = [
    PromptTemplateInputInstance(
        id="prod_001",
        prompt_value_mapping={"product_name": "AeroGlide Drone", "features": "4K camera, 30-min flight"}
    ),
    PromptTemplateInputInstance(
        id="prod_002",
        prompt_value_mapping={"product_name": "HydroPure Bottle", "features": "Self-cleaning, insulated steel"}
    ),
    # ... add up to 9,998 more products
]

# 4. Use the manager to generate the batch file
manager = BatchJobManager()
manager.add_templated_instances(
    prompt=copywriting_template,
    common_request=common_request_config,
    input_instances=product_instances,
    save_file_path="copywriting_batch.jsonl"
)
```

### Example 2: Batch Embedding Requests

Similarly, you can easily create a batch job for generating embeddings for a large number of documents.

```python
from openbatch import BatchJobManager, EmbeddingsRequest, EmbeddingInputInstance

# 1. Define the common configuration for all embedding requests
common_embedding_config = EmbeddingsRequest(
    model="text-embedding-3-small",
    dimensions=512
)

# 2. Create a list of input instances
documents_to_embed = [
    EmbeddingInputInstance(id="doc_1", input="The sky is blue."),
    EmbeddingInputInstance(id="doc_2", input="Grass is green."),
    # ... add thousands more documents
]

# 3. Use the manager to generate the batch file
manager = BatchJobManager()
manager.add_embedding_requests(
    inputs=documents_to_embed,
    common_request=common_embedding_config,
    save_file_path="embeddings_batch.jsonl"
)
```

-----

## Configuring the Request

The `common_request` objects (`ResponsesRequest`, `EmbeddingsRequest`, etc.) are Pydantic models that expose all available API parameters. You can configure any parameter by passing it to the constructor.

```python
from openbatch import ResponsesRequest, ReasoningConfig

# Example of a more detailed configuration for the Responses API
detailed_config = ResponsesRequest(
    model="gpt-4o",
    service_tier="flex",
    reasoning=ReasoningConfig(effort="minimal"),
    max_output_tokens=500
)
```

You can also override any common setting on a per-instance basis by using the `instance_request_options` field.

-----

## What's Next?

`OpenBatch` helps you create the batch file. The next steps involve using that file with the OpenAI API:

1.  **Upload File**: Upload your generated `.jsonl` file to OpenAI.
2.  **Create Batch Job**: Create a new batch job pointing to your uploaded file.
3.  **Retrieve Results**: Monitor the job's status and, once completed, download the output file with the results.

For detailed instructions on these steps, please refer to the **[Official OpenAI Batch API Documentation](https://platform.openai.com/docs/api-reference/batch)**.