# BatchCollector API

The `BatchCollector` provides a high-level, fluent API for creating batch files. Its structure mimics the official `openai` client, making it the ideal tool for adding individual, distinct requests to a single batch file.

The first step is to instantiate the collector with the path to your desired output file.

```python
from openbatch import BatchCollector

# All requests will be appended to this file
collector = BatchCollector(batch_file_path="my_batch_job.jsonl") 
```

Once initialized, you can use the helper attributes (`chat.completions`, `responses`, `embeddings`) to add requests.

-----

## Chat Completions API (`collector.chat.completions`)

This interface is used to add requests to the `/v1/chat/completions` endpoint.

### `create(...)`

Adds a standard chat completion request to the batch file. Use this for general-purpose text generation, question answering, or any standard chat interaction.

**Example**

```python
collector.chat.completions.create(
    custom_id="chat-req-1",
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of the Netherlands?"}
    ],
    temperature=0.7
)
```

**API Reference**
::: openbatch.collector.ChatCompletions.create

### `parse(...)`

Adds a chat completion request that **enforces a structured JSON output** based on a Pydantic model. This is perfect for data extraction, classification, or when you need a predictable, machine-readable response.

**Example**

```python
from pydantic import BaseModel, Field

# 1. Define your desired output structure
class CityInfo(BaseModel):
    city: str
    country: str
    population: int = Field(description="Estimated population")

# 2. Add the request using .parse()
collector.chat.completions.parse(
    custom_id="chat-json-req-2",
    model="gpt-4o",
    response_format=CityInfo, # Pass the Pydantic model here
    messages=[
        {"role": "user", "content": "Extract information about Paris, France."}
    ]
)
```

**API Reference**
::: openbatch.collector.ChatCompletions.parse

-----

## Responses API (`collector.responses`)

This interface is used to add requests to the `/v1/responses` endpoint, which is optimized for specific features like reusable prompts and advanced tool use.

### `create(...)`

Adds a standard responses API request to the batch file.

**Example**

```python
collector.responses.create(
    custom_id="resp-req-1",
    model="gpt-4o",
    instructions="You are a travel agent.",
    input="Suggest three activities to do in Amsterdam."
)
```

**API Reference**
::: openbatch.collector.Responses.create

### `parse(...)`

Adds a responses API request that enforces a structured JSON output based on a Pydantic model. This works similarly to the `chat.completions.parse` method.

**Example**

```python
from pydantic import BaseModel
from typing import List

# 1. Define your desired output structure
class ItineraryItem(BaseModel):
    activity: str
    description: str
    estimated_duration_hours: float

class TravelPlan(BaseModel):
    city: str
    suggestions: List[ItineraryItem]

# 2. Add the request using .parse()
collector.responses.parse(
    custom_id="resp-json-req-2",
    model="gpt-4o",
    text_format=TravelPlan, # Pass the Pydantic model here
    instructions="Create a travel plan based on the user's request.",
    input="I have one day in Tokyo. What should I do?"
)
```

**API Reference**
::: openbatch.collector.Responses.parse

-----

## Embeddings API (`collector.embeddings`)

This interface is used to add requests to the `/v1/embeddings` endpoint.

### `create(...)`

Adds an embedding request to the batch file. You can provide a single string or a list of strings as input.

**Example**

```python
collector.embeddings.create(
    custom_id="embed-req-1",
    model="text-embedding-3-small",
    inp="OpenAI batching is a powerful feature.",
    dimensions=256
)
```

**API Reference**
::: openbatch.collector.Embeddings.create