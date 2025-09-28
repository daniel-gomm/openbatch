# Requests

These Pydantic models represent the configuration for a request to a specific OpenAI API endpoint. They contain all the parameters you can set, such as `model`, `temperature`, `max_tokens`, etc.

You use these models when defining a `common_request` for the `BatchJobManager` or when creating a request via the `BatchCollector`.

---

## `ChatCompletionsRequest`

Configuration for a `/v1/chat/completions` API request.

::: openbatch.model.ChatCompletionsRequest

---

## `ResponsesRequest`

Configuration for a `/v1/responses` API request.

::: openbatch.model.ResponsesRequest

---

## `EmbeddingsRequest`

Configuration for a `/v1/embeddings` API request.

::: openbatch.model.EmbeddingsRequest