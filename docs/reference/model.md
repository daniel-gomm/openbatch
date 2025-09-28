# Data Models

This section covers the core data models used to structure your prompts, inputs, and instances.

---

## Prompts

### `PromptTemplate`

A template containing a sequence of messages with placeholders for string formatting.

::: openbatch.model.PromptTemplate

### `ReusablePrompt`

A reference to a reusable prompt on the OpenAI server (see [Official Documentation](https://platform.openai.com/docs/guides/prompt-engineering/prompt-engineering-best-practices#reusable-prompts)).

::: openbatch.model.ReusablePrompt

---

## Input Instances

Input instances package the unique data for each request in a batch.

### `PromptTemplateInputInstance`

Used with `BatchJobManager` to provide values for a `PromptTemplate` or `ReusablePrompt`.

::: openbatch.model.PromptTemplateInputInstance

### `EmbeddingInputInstance`

Used with `BatchJobManager` to provide the text(s) for an embedding request.

::: openbatch.model.EmbeddingInputInstance

### `MessagesInputInstance`

A generic input instance defined by a complete list of messages.

::: openbatch.model.MessagesInputInstance

---

## Core Components

### `Message`

Represents a single message in a conversation.

::: openbatch.model.Message