import json
from copy import deepcopy
from pathlib import Path
from typing import TypeVar, Iterable, Union

from openbatch.model import PromptTemplate, ReusablePrompt, PromptTemplateInputInstance, ResponsesRequest, \
    ResponsesAPIStrategy, EmbeddingsAPIStrategy, ChatCompletionsAPIStrategy, BaseRequest, \
    EmbeddingsRequest, EmbeddingInputInstance, ChatCompletionsRequest

B = TypeVar("B", bound=BaseRequest)
R = TypeVar("R", bound=Union[ResponsesRequest, ChatCompletionsRequest])


class BatchJobManager:
    """
    Manages the creation of batch job request files.

    Provides methods to generate request line-by-line JSON files based on
    prompt templates, common request configurations, and input instances.
    """

    def add_templated_instances(self, prompt: Union[PromptTemplate, ReusablePrompt],
                                common_request: R, input_instances: Iterable[PromptTemplateInputInstance],
                                save_file_path: str | Path) -> None:
        """
       Adds multiple templated input instances to a batch request file.

       This method iterates over input instances, applies variable mapping to the
       prompt, merges instance-specific options, and adds the resulting request
       to the batch file.

       Args:
           prompt (Union[PromptTemplate, ReusablePrompt]): The prompt definition, either
               an in-memory template or a reference to a reusable prompt.
           common_request (Union[ResponsesRequest, ChatCompletionsRequest]): The base request configuration (e.g., model, temperature).
               Must be a TextGenerationRequest subclass (ResponsesRequest or ChatCompletionsRequest).
           input_instances (Iterable[PromptTemplateInputInstance]): An iterable of
               instances containing prompt variable mappings and instance options.
           save_file_path (str | Path): The path to the batch job request file (JSONL format).

       Raises:
           ValueError: If `common_request` is an `EmbeddingsRequest` or any other
                       unsupported request type.
       """
        if isinstance(common_request, EmbeddingsRequest):
            raise ValueError("Embeddings API is not supported with templated instances.")
        elif not isinstance(common_request, ResponsesRequest) and not isinstance(common_request, ChatCompletionsRequest):
            raise ValueError(f"Unsupported request type: {type(common_request)}")

        save_file_path = Path(save_file_path)
        save_file_path.parent.mkdir(parents=True, exist_ok=True)

        for instance in input_instances:
            request = deepcopy(common_request)
            if instance.instance_request_options is not None:
                request = request.model_copy(update=instance.instance_request_options)
            request = self._handle_prompt(prompt, request, instance)

            self.add(instance.id, request, save_file_path)

    def add_embedding_requests(self, inputs: Iterable[EmbeddingInputInstance], common_request: EmbeddingsRequest,
                               save_file_path: Union[str, Path]) -> None:
        """
        Adds multiple embedding request instances to a batch request file.

        This method iterates over embedding input instances, sets the input text(s),
        merges instance-specific options, and adds the request to the batch file.

        Args:
            inputs (Iterable[EmbeddingInputInstance]): An iterable of instances
                containing the text(s) to embed and instance options.
            common_request (EmbeddingsRequest): The base embedding request configuration.
            save_file_path (Union[str, Path]): The path to the batch job request file (JSONL format).
        """
        save_file_path = Path(save_file_path)
        save_file_path.parent.mkdir(parents=True, exist_ok=True)

        for instance in inputs:
            request = deepcopy(common_request)
            if instance.instance_request_options is not None:
                request = request.model_copy(update=instance.instance_request_options)
            request.set_input(instance.input)

            self.add(instance.id, request, save_file_path)

    @staticmethod
    def add(custom_id: str, request: B, save_file_path: Union[str, Path]) -> None:
        """
        Creates a single batch request object and appends it to the specified file.

        This is the core method for generating the JSONL file content. It determines
        the appropriate API strategy based on the request type and serializes
        the full request structure.

        Args:
            custom_id (str): A unique identifier for this specific request in the batch.
            request (Union[ResponsesRequest, ChatCompletionsRequest, EmbeddingsRequest]): The API-specific request configuration object.
            save_file_path (Union[str, Path]): The path to the batch job request file (JSONL format).

        Raises:
            ValueError: If the request type is unsupported or if a required field
                        (like `input` or `messages`) is missing from the request.
        """
        if isinstance(request, ResponsesRequest):
            strategy = ResponsesAPIStrategy()
            if request.input is None and request.prompt is None:
                raise ValueError("Responses request must define either an input or a prompt.")
        elif isinstance(request, ChatCompletionsRequest):
            strategy = ChatCompletionsAPIStrategy()
            if request.messages is None:
                raise ValueError("Chat Completions request must define messages.")
        elif isinstance(request, EmbeddingsRequest):
            strategy = EmbeddingsAPIStrategy()
            if request.input is None:
                raise ValueError("Embeddings request must define an input.")
        else:
            raise ValueError(f"Unsupported request type: {type(request)}")

        save_file_path = Path(save_file_path)
        save_file_path.parent.mkdir(parents=True, exist_ok=True)

        batch_request = strategy.create_request(
            custom_id=custom_id,
            body=request.to_dict()
        )

        with open(save_file_path, 'a+') as outfile:
            outfile.write(json.dumps(batch_request) + "\n")



    @staticmethod
    def _handle_prompt(prompt: PromptTemplate | ReusablePrompt, request: R, instance: PromptTemplateInputInstance) -> R:
        if isinstance(prompt, ReusablePrompt):
            if not isinstance(request, ResponsesRequest):
                raise ValueError("Reusable prompts can only be used with ResponsesOptions.")
            request.prompt = ReusablePrompt(
                id=prompt.id,
                version=prompt.version,
                variables=instance.prompt_value_mapping
            )
        elif isinstance(prompt, PromptTemplate):
            messages = prompt.format(**instance.prompt_value_mapping)
            request.set_input_messages(messages)
        return request
