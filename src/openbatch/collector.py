from os import PathLike
from pathlib import Path
from types import SimpleNamespace

from pydantic import BaseModel

from openbatch.manager import BatchJobManager
from openbatch.model import ChatCompletionsRequest, EmbeddingsRequest, ResponsesRequest


class Responses:
    """
    A utility class for easily constructing and adding individual
    Responses API requests to a batch job file.

    It acts as a high-level interface for the '/v1/responses' endpoint.
    """

    def __init__(self, batch_file_path: str | PathLike):
        """
        Initializes the Responses collector.

        Args:
            batch_file_path (Union[str, PathLike]): The path to the JSONL file
                where the batch requests will be written.
        """
        self.batch_file_path = Path(batch_file_path)
        self._manager = BatchJobManager()

    def parse(
        self, custom_id: str, model: str, text_format: type[BaseModel] | None = None, **kwargs
    ) -> None:
        """
        Creates a ResponsesRequest, optionally enforcing a JSON output structure,
        and adds it to the batch file. Use it like the `OpenAI().responses.parse()` method.

        Args:
            custom_id (str): A unique ID for the request in the batch file.
            model (str): The model ID to use for the request.
            text_format (Optional[Type[BaseModel]]): An optional Pydantic model
                to define the desired JSON output structure for the response.
            **kwargs: Additional parameters for the ResponsesRequest (e.g., instructions, input, temperature).
        """
        request = ResponsesRequest.model_validate({"model": model, **kwargs})
        if text_format is not None:
            request.set_output_structure(text_format)
        self._add_request(custom_id, request)

    def create(self, custom_id: str, model: str, **kwargs) -> None:
        """
        Creates a standard ResponsesRequest and adds it to the batch file. Use it like the `OpenAI().responses.create()` method.

        Args:
            custom_id (str): A unique ID for the request in the batch file.
            model (str): The model ID to use for the request.
            **kwargs: Additional parameters for the ResponsesRequest.
        """
        request = ResponsesRequest.model_validate({"model": model, **kwargs})
        self._add_request(custom_id, request)

    def _add_request(self, custom_id: str, request: ResponsesRequest) -> None:
        self._manager.add(custom_id, request, self.batch_file_path)


class ChatCompletions:
    """
    A utility class for easily constructing and adding individual
    Chat Completions API requests to a batch job file.

    It acts as a high-level interface for the '/v1/chat/completions' endpoint.
    """

    def __init__(self, batch_file_path: str | PathLike):
        """
        Initializes the ChatCompletions collector.

        Args:
            batch_file_path (Union[str, PathLike]): The path to the JSONL file
                where the batch requests will be written.
        """
        self.batch_file_path = Path(batch_file_path)
        self._manager = BatchJobManager()

    def parse(
        self, custom_id: str, model: str, response_format: type[BaseModel] | None = None, **kwargs
    ) -> None:
        """
        Creates a ChatCompletionsRequest, optionally enforcing a JSON output structure,
        and adds it to the batch file. Use it like the `OpenAI().chat.completions.parse()` method.

        Args:
            custom_id (str): A unique ID for the request in the batch file.
            model (str): The model ID to use for the request.
            response_format (Optional[Type[BaseModel]]): An optional Pydantic model
                to define the desired JSON output structure.
            **kwargs: Additional parameters for the ChatCompletionsRequest (e.g., messages, temperature).
        """
        request = ChatCompletionsRequest.model_validate({"model": model, **kwargs})
        if response_format is not None:
            request.set_output_structure(response_format)
        self._add_request(custom_id, request)

    def create(self, custom_id: str, model: str, **kwargs) -> None:
        """
        Creates a standard ChatCompletionsRequest and adds it to the batch file. Use it like the `OpenAI().chat.completions.create()` method.

        Args:
            custom_id (str): A unique ID for the request in the batch file.
            model (str): The model ID to use for the request.
            **kwargs: Additional parameters for the ChatCompletionsRequest.
        """
        request = ChatCompletionsRequest.model_validate({"model": model, **kwargs})
        self._add_request(custom_id, request)

    def _add_request(self, custom_id: str, request: ChatCompletionsRequest) -> None:
        self._manager.add(custom_id, request, self.batch_file_path)


class Embeddings:
    """
    A utility class for easily constructing and adding individual
    Embeddings API requests to a batch job file.

    It acts as a high-level interface for the '/v1/embeddings' endpoint.
    """

    def __init__(self, batch_file_path: str | PathLike):
        """
        Initializes the Embeddings collector.

        Args:
            batch_file_path (Union[str, PathLike]): The path to the JSONL file
                where the batch requests will be written.
        """
        self.batch_file_path = Path(batch_file_path)
        self._manager = BatchJobManager()

    def create(self, custom_id: str, model: str, inp: str | list[str], **kwargs) -> None:
        """
        Creates an EmbeddingsRequest and adds it to the batch file. Use it like the `OpenAI().embeddings.create()` method.

        Args:
            custom_id (str): A unique ID for the request in the batch file.
            model (str): The model ID to use for the request.
            inp (Union[str, list[str]]): The input text(s) to be embedded.
            **kwargs: Additional parameters for the EmbeddingsRequest.
        """
        request = EmbeddingsRequest.model_validate({"model": model, "input": inp, **kwargs})
        self._manager.add(custom_id, request, self.batch_file_path)


class BatchCollector:
    """
    A high-level utility class for creating OpenAI batch job files.

    This class provides a convenient and familiar interface, mimicking the structure
    of the official OpenAI Python client (e.g., `openai.chat.completions.create`).
    It's designed for adding individual API requests to a JSONL file one by one,
    making it ideal for simple or ad-hoc batch creation tasks.

    After initialization, requests can be added via the following attributes:
    - `collector.responses`: For `/v1/responses` endpoint requests.
    - `collector.chat.completions`: For `/v1/chat/completions` endpoint requests.
    - `collector.embeddings`: For `/v1/embeddings` endpoint requests.

    Args:
        batch_file_path (Union[str, PathLike]): The path to the JSONL file
            where the batch requests will be written. The file will be created
            if it doesn't exist and appended to if it does.
    """

    def __init__(self, batch_file_path: str | PathLike):
        self.responses = Responses(batch_file_path)
        self.chat = SimpleNamespace()
        self.chat.completions = ChatCompletions(batch_file_path)
        self.embeddings = Embeddings(batch_file_path)
