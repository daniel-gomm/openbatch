import json
from copy import deepcopy
from pathlib import Path
from typing import TypeVar, Iterable, Union, Literal

from OpenAIBatch.model import PromptTemplate, ReusablePrompt, PromptTemplateInputInstance, RequestStrategy, \
    ResponsesRequest, TestGenerationRequest, ResponsesAPIStrategy, EmbeddingsAPIStrategy, ChatCompletionsAPIStrategy, \
    BaseRequest, EmbeddingsRequest, EmbeddingInputInstance

B = TypeVar("B", bound=BaseRequest)
R = TypeVar("R", bound=TestGenerationRequest)




class BatchJobManager:

    def add_templated_instances(self, prompt: PromptTemplate | ReusablePrompt,
                                common_request: R, input_instances: Iterable[PromptTemplateInputInstance],
                                save_file_path: str | Path,
                                request_strategy: Union[RequestStrategy, Literal["completion", "responses"]]) -> None:
        if isinstance(request_strategy, EmbeddingsAPIStrategy):
            raise ValueError("EmbeddingsAPIStrategy cannot be used with templated instances.")
        elif isinstance(request_strategy, str) and request_strategy == "embeddings":
            raise ValueError("Embeddings request strategy cannot be used with templated instances.")

        save_file_path = Path(save_file_path)
        save_file_path.parent.mkdir(parents=True, exist_ok=True)

        for instance in input_instances:
            body_options = deepcopy(common_request)
            body_options = self._handle_prompt(prompt, body_options, instance)
            body_options = body_options.model_validate({**common_request.to_dict(), **instance.instance_request_options})

            self.add(instance.id, body_options, save_file_path, request_strategy)

    def add_embedding_requests(self, inputs: Iterable[EmbeddingInputInstance], common_request: EmbeddingsRequest,
                               save_file_path: Union[str, Path]) -> None:
        save_file_path = Path(save_file_path)
        save_file_path.parent.mkdir(parents=True, exist_ok=True)

        for instance in inputs:
            body_options = deepcopy(common_request)
            body_options = body_options.model_validate({**common_request.to_dict(), **instance.instance_request_options})
            body_options.set_input(instance.input)

            self.add(instance.id, body_options, save_file_path, EmbeddingsAPIStrategy())

    @staticmethod
    def add(custom_id: str, request: B, save_file_path: Union[str, Path],
            request_strategy: Union[RequestStrategy, Literal["completion", "responses", "embeddings"]]) -> None:
        if isinstance(request_strategy, str):
            if request_strategy == "responses":
                strategy = ResponsesAPIStrategy()
            elif request_strategy == "embeddings":
                strategy = EmbeddingsAPIStrategy()
            elif request_strategy == "completion":
                strategy = ChatCompletionsAPIStrategy()
            else:
                raise ValueError(f"Unknown request strategy string: {request_strategy}")
        else:
            strategy = request_strategy

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
            request.update(
                {
                    "prompt": {
                        "id": prompt.id,
                        "version": prompt.version,
                        "variables": instance.prompt_value_mapping
                    }
                }
            )
        elif isinstance(prompt, PromptTemplate):
            messages = prompt.format(**instance.prompt_value_mapping)
            request.set_input_messages(messages)
        return request
