# Welcome to OpenBatch

**OpenBatch** is a lightweight Python utility designed to streamline the creation of JSONL files for the [OpenAI Batch API](https://platform.openai.com/docs/guides/batch). It provides a type-safe and intuitive interface using Pydantic models to construct requests for various endpoints, including `/v1/chat/completions`, `/v1/embeddings`, and the new `/v1/responses` endpoint.

## Key Features

- **Type-Safe & Modern**: Built with Pydantic for robust, self-documenting, and editor-friendly request models.
- **Dual APIs for Flexibility**:
    - **`BatchCollector`**: A high-level, fluent API that mimics the official `openai` client. It's perfect for adding individual, distinct requests to a batch file with minimal setup.
    - **`BatchJobManager`**: A lower-level API designed for programmatically generating large batches of requests from templates and lists of inputs. It's ideal for scalable tasks like classification, data extraction, or bulk embeddings.
- **Support for All Batch Endpoints**: Easily create requests for Chat Completions, Embeddings, and the Responses API.
- **Structured Outputs**: Natively supports defining a Pydantic model as the desired JSON output structure, simplifying data extraction tasks.

## How It Works

The library simplifies the process of creating the `jsonl` file required by the OpenAI Batch API.

1.  **Define Your Requests**: Use the library's Pydantic models to define either individual requests (`BatchCollector`) or a common template and a list of inputs (`BatchJobManager`).
2.  **Generate the File**: The library writes each request as a properly formatted JSON object on a new line in the specified output file.
3.  **Upload to OpenAI**: Once the file is generated, you can upload it to OpenAI and start your batch job.

Ready to get started? Head over to the **[Getting Started](./getting_started.md)** guide!