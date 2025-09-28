# Getting Started

This guide will walk you through installing `OpenBatch` and creating your first batch job file. We'll focus on using the modern `/v1/responses` API, exploring both the simple `BatchCollector` and the powerful `BatchJobManager`.

-----

## 1\. Installation

First, install the library using pip:

```bash
pip install openbatch
```

-----

## 2\. Simple Usage: The `BatchCollector`

The `BatchCollector` is the quickest way to create a batch file. It's designed for when you need to add a few different requests together. These examples use the `/v1/responses` endpoint.

```python
from pydantic import BaseModel, Field
from typing import List
from openbatch import BatchCollector, ReasoningConfig

# Define Pydantic models for structured JSON output
class Ingredient(BaseModel):
    name: str
    quantity: str

class Recipe(BaseModel):
    title: str = Field(description="The name of the recipe")
    ingredients: List[Ingredient]
    instructions: List[str]

# Initialize the collector with the output file path
BATCH_FILE = "./responses_batch.jsonl"
collector = BatchCollector(batch_file_path=BATCH_FILE)

# Add a standard text-generation request
print("Adding a standard responses request...")
collector.responses.create(
    custom_id="request-1",
    model="gpt-4o-mini",
    instructions="You are a poetic assistant, skilled in explaining complex programming concepts with creative flair.",
    input="Explain the concept of recursion in programming.",
    temperature=0.8,
    max_output_tokens=150
)

# Add a request that returns structured JSON
print("Adding a structured JSON responses request...")
collector.responses.parse(
    custom_id="request-2",
    model="gpt-5-mini",
    text_format=Recipe, # Pass the Pydantic model to enforce the output structure
    instructions="Generate a simple recipe based on the user's input. The recipe should be clear and concise.",
    input="I have chicken, rice, and broccoli.",
    reasoning=ReasoningConfig(effort="low", summary="concise") # Add reasoning config
)

print(f"Batch file '{BATCH_FILE}' created successfully.")
```

-----

## 3\. Advanced Usage: The `BatchJobManager`

For larger, repetitive tasks, the `BatchJobManager` excels. It can generate thousands of similar requests from a single template, making it perfect for scaling up your workflows.

Here, we'll use a template to generate product descriptions for a list of items.

```python
from openbatch import (
    BatchJobManager,
    PromptTemplate,
    Message,
    ResponsesRequest,
    ReasoningConfig,
    PromptTemplateInputInstance
)

# 1. Define a prompt template with placeholders `{product_name}` and `{features}`
description_template = PromptTemplate(
    messages=[
        Message(
            role="system",
            content="You are a marketing copywriter. Write a compelling, short product description (2-3 sentences) based on the provided features."
        ),
        Message(
            role="user",
            content="Product Name: {product_name}\nFeatures: {features}"
        )
    ]
)

# 2. Define the common configuration for all requests
# Note the additional options like service_tier and reasoning
common_config = ResponsesRequest(
    model="gpt-4o-mini",
    temperature=0.7,
    max_output_tokens=100,
    service_tier="flex", # Specify the processing tier
    reasoning=ReasoningConfig(summary="auto")
)

# 3. Create a list of input instances to fill the template
product_data = [
    {
        "id": "item_101",
        "name": "AeroGlide Drone",
        "feats": "4K camera, 30-min flight time, obstacle avoidance, lightweight design"
    },
    {
        "id": "item_102",
        "name": "HydroPure Water Bottle",
        "feats": "Self-cleaning UV-C light, insulated stainless steel, 24-hour cold retention"
    },
]

input_instances = [
    PromptTemplateInputInstance(
        id=product["id"],
        prompt_value_mapping={
            "product_name": product["name"],
            "features": product["feats"]
        }
    ) for product in product_data
]

# 4. Use the manager to generate the batch file
BATCH_FILE = "product_descriptions.jsonl"
manager = BatchJobManager()
manager.add_templated_instances(
    prompt=description_template,
    common_request=common_config,
    input_instances=input_instances,
    save_file_path=BATCH_FILE
)

print(f"Batch file '{BATCH_FILE}' with {len(input_instances)} requests created.")
```

-----

## 4\. Next Steps

`OpenBatch`'s job is done once the `.jsonl` file is created. Your next steps involve using the OpenAI API:

1.  **Upload the File**: Use the OpenAI API to upload your generated file.
2.  **Create the Batch Job**: Start a new batch job, pointing to the file ID from the upload step.
3.  **Retrieve the Results**: Monitor the job and download the results when complete.

For details, see the [official OpenAI Batch API documentation](https://platform.openai.com/docs/api-reference/batch).