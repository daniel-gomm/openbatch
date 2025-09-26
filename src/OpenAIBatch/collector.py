from os import PathLike
from typing import Union, Optional

from pydantic import BaseModel

from OpenAIBatch.manager import BatchJobManager
from OpenAIBatch.model import ResponsesAPIStrategy, ChatCompletionsAPIStrategy, ResponsesRequest, CompletionsRequest, \
    EmbeddingsRequest, EmbeddingsAPIStrategy


class Responses:

    def __init__(self, batch_file_path: Union[str, PathLike]):
        self.batch_file_path = batch_file_path
        self.batch_manager = BatchJobManager()

    def parse(self, custom_id: str, model: str, text_format: Optional[type[BaseModel]] = None, **kwargs) -> None:
        request = ResponsesRequest.model_validate({"model": model, **kwargs})
        if text_format is not None:
            request.set_output_structure(text_format)
        self._add_request(custom_id, request)

    def create(self, custom_id: str, model: str, **kwargs) -> None:
        request = ResponsesRequest.model_validate({"model": model, **kwargs})
        self._add_request(custom_id, request)

    def _add_request(self, custom_id: str, request: ResponsesRequest) -> None:
        self.batch_manager.add(custom_id, request, self.batch_file_path, ResponsesAPIStrategy())

class Completions:
    def __init__(self, batch_file_path: Union[str, PathLike]):
        self.batch_file_path = batch_file_path
        self.batch_manager = BatchJobManager()

    def parse(self, custom_id: str, model: str, response_format: Optional[type[BaseModel]] = None, text_format: Optional[type[BaseModel]] = None, **kwargs) -> None:
        request = CompletionsRequest.model_validate({"model": model, **kwargs})
        if response_format is not None:
            request.set_output_structure(text_format)
        self._add_request(custom_id, request)

    def create(self, custom_id: str, model: str, **kwargs) -> None:
        request = CompletionsRequest.model_validate({"model": model, **kwargs})
        self._add_request(custom_id, request)

    def _add_request(self, custom_id: str, request: CompletionsRequest) -> None:
        self.batch_manager.add(custom_id, request, self.batch_file_path, ChatCompletionsAPIStrategy())

class Embeddings:
    def __init__(self, batch_file_path: Union[str, PathLike]):
        self.batch_file_path = batch_file_path
        self.batch_manager = BatchJobManager()

    def create(self, custom_id: str, model: str, input: Union[str, list[str]], **kwargs) -> None:
        request = EmbeddingsRequest.model_validate({"model": model, "input": input, **kwargs})
        self.batch_manager.add(custom_id, request, self.batch_file_path, EmbeddingsAPIStrategy())


class RequestCollector:
    def __init__(self, batch_file_path: Union[str, PathLike]):
        self.responses = Responses(batch_file_path)
        self.chat = ()
        self.chat.completions = Completions(batch_file_path)
        self.embeddings = Embeddings(batch_file_path)
