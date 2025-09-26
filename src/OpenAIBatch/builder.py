import json
import os
from copy import deepcopy
from pathlib import Path
from typing import Dict, Any, TypeVar, Self, Literal, Iterable

from pydantic import BaseModel

from OpenAIBatch.model import PromptTemplate, ReusablePrompt, InputInstance
from OpenAIBatch._utils import ensure_strict_json_schema

T = TypeVar("T", bound=BaseModel)

class BatchJobBuilder:

    def __init__(self, prompt: PromptTemplate | ReusablePrompt, model: str):
        self.prompt = prompt
        self.body_options: Dict[str, Any] = {
            "model": model
        }

    def enforce_structured_output(self, response_model: type[T]) -> Self:
        json_schema = response_model.model_json_schema()
        schema = ensure_strict_json_schema(json_schema, path=(), root=json_schema)
        self.body_options["text"] = {
            "format": {
                "type": "json_schema",
                "name": response_model.__name__,
                "schema": schema,
                "strict": True
            }
        }
        return self

    def set_reasoning_effort(self, effort: Literal["low", "medium", "high"]) -> Self:
        self.body_options["reasoning"] = {"effort": effort}
        return self

    def set_max_output_tokens(self, max_tokens: int) -> Self:
        self.body_options["max_output_tokens"] = max_tokens
        return self

    def set_top_p(self, top_p: float) -> Self:
        self.body_options["top_p"] = top_p
        return self

    def set_prompt_cache_key(self, cache_key: str) -> Self:
        self.body_options["prompt_cache_key"] = cache_key
        return self

    def build_batch_job(self, input_instances: Iterable[InputInstance], save_file_path: str | Path, url: str = "/v1/responses") -> None:
        base_options: Dict[str, Any] = {
            "method": "POST",
            "url": url
        }
        save_file_path = Path(save_file_path)
        if not os.path.exists(save_file_path.parent):
            save_file_path.parent.mkdir(parents=True, exist_ok=True)

        if isinstance(self.prompt, ReusablePrompt):
            with open(save_file_path, 'a+') as outfile:
                for instance in input_instances:
                    body_options = deepcopy(self.body_options)
                    body_options["prompt"] = {
                        "id": self.prompt.id,
                        "version": self.prompt.version,
                        "variables": instance.prompt_value_mapping
                    }
                    options = deepcopy(base_options)
                    options["custom_id"] = instance.id
                    options["body"] = body_options
                    outfile.write(json.dumps(options) + "\n")
        elif isinstance(self.prompt, PromptTemplate):
            with open(save_file_path, 'a+') as outfile:
                for instance in input_instances:
                    body_options = deepcopy(self.body_options)
                    messages = self.prompt.format(**instance.prompt_value_mapping)
                    body_options["input"] = [m.serialize() for m in messages]
                    options = deepcopy(base_options)
                    options["custom_id"] = instance.id
                    options["body"] = body_options
                    outfile.write(json.dumps(options) + "\n")
        else:
            raise NotImplementedError
