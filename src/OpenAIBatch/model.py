from abc import abstractmethod, ABC
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Literal, TypeVar, Union

from pydantic import BaseModel, Field

from OpenAIBatch._utils import type_to_json_schema


T = TypeVar("T", bound=BaseModel)

@dataclass
class Message:
    role: str
    content: str

    def serialize(self):
        return {"role": self.role, "content": self.content}

@dataclass
class PromptTemplate:
    messages: List[Message]

    def format(self, **kwargs) -> List[Message]:
        formatted_messages = []
        for message in self.messages:
            formatted_content = message.content.format(**kwargs)
            formatted_messages.append(Message(role=message.role, content=formatted_content))
        return formatted_messages

class ReusablePrompt(BaseModel):
    id: str
    version: str
    variables: Dict[str, Any]

class ReasoningConfig(BaseModel):
    effort: Literal["minimal", "low", "medium", "high"] = Field(default="medium", description="Constrains effort on reasoning for reasoning models.")
    summary: Optional[Literal["auto", "concise", "detailed"]] = Field(None, description="A summary of the reasoning performed by the model.")

class InputInstance(BaseModel):
    id: str = Field(description="Unique identifier of the input instance.")
    instance_request_options: Optional[Dict[str, Any]] = Field(None, description="Options specific to the input instance that to set in the request.")

class MessagesInputInstance(InputInstance):
    messages: List[Message] = Field(description="List of messages to be sent to the model.")

class PromptTemplateInputInstance(InputInstance):
    prompt_value_mapping: Dict[str, str] = Field(description="Mapping of prompt variable names to their values.")

class EmbeddingInputInstance(InputInstance):
    input: Union[str, List[str]] = Field(description="Text(s) to be embedded.")

class RequestStrategy(ABC):
    @property
    @abstractmethod
    def url(self) -> str:
        pass

    def create_request(self, custom_id: str, body: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "custom_id": custom_id,
            "method": "POST",
            "url": self.url,
            "body": body
        }

class ResponsesAPIStrategy(RequestStrategy):
    """Strategy for the /v1/responses endpoint."""
    @property
    def url(self) -> str:
        return "/v1/responses"

class ChatCompletionsAPIStrategy(RequestStrategy):
    """Strategy for the /v1/chat/completions endpoint."""
    @property
    def url(self) -> str:
        return "/v1/chat/completions"

class EmbeddingsAPIStrategy(RequestStrategy):
    """Strategy for the /v1/embeddings endpoint."""
    @property
    def url(self) -> str:
        return "/v1/embeddings"

class BaseRequest(BaseModel, ABC):
    """Base class for API-specific job configurations."""
    model: str = Field("gpt-4.1", description="Model ID used to generate the response, like gpt-4o or o3.")

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump(exclude_none=True)


class TestGenerationRequest(BaseRequest, ABC):
    tools: Optional[List[object]] = Field(None, description="An array of tools the model may call while generating a response.")
    top_p: Optional[float] = Field(None, ge=0, le=1, description="An alternative to sampling with temperature, called nucleus sampling, where the model considers the results of the tokens with top_p probability mass.")
    parallel_tool_calls: Optional[bool] = Field(None, description="Whether to allow the model to run tool calls in parallel.")
    prompt_cache_key: Optional[str] = Field(None, description="Used by OpenAI to cache responses for similar requests to optimize your cache hit rates.")
    safety_identifier: Optional[str] = Field(None, description="A stable identifier used to help detect users of your application that may be violating OpenAI's usage policies.")
    service_tier: Optional[Literal["auto", "default", "flex", "priority"]] = Field(None, description="Specifies the processing type used for serving the request.")
    store: Optional[bool] = Field(None, description="Whether to store the generated model response for later retrieval via API.")
    temperature: Optional[float] = Field(None, ge=0, le=2, description="What sampling temperature to use, between 0 and 2.")
    tool_choice: Optional[str | object] = Field(None, description="How the model should select which tool (or tools) to use when generating a response.")
    top_logprobs: Optional[int] = Field(None, ge=0, le=20, description="An integer between 0 and 20 specifying the number of most likely tokens to return at each token position, each with an associated log probability.")


    @abstractmethod
    def set_output_structure(self, output_type: type[T]) -> None:
        pass

    @abstractmethod
    def set_input_messages(self, messages: List[Message]) -> None:
        pass


class ResponsesRequest(TestGenerationRequest):
    conversation: Optional[str] = Field(None, description="The conversation that this response belongs to.")
    include: Optional[List[Literal["code_interpreter_call.outputs", "computer_call_output.output.image_url", "file_search_call.results", "message.input_image.image_url", "message.output_text.logprobs", "reasoning.encrypted_content"]]] = Field(None, description="Specify additional output data to include in the model response.")
    input: Optional[str | List[Dict[str, str]]] = Field(None, description="Text, image, or file inputs to the model, used to generate a response.")
    instructions: Optional[str] = Field(None, description="A system (or developer) message inserted into the model's context.")
    max_output_tokens: Optional[int] = Field(None, gt=0, description="An upper bound for the number of tokens that can be generated for a response, including visible output tokens and reasoning tokens.")
    max_tool_calls: Optional[int] = Field(None, gt=0, description="The maximum number of total calls to built-in tools that can be processed in a response.")
    previous_response_id: Optional[str] = Field(None, description="The unique ID of the previous response to the model. Use this to create multi-turn conversations.")
    prompt: Optional[ReusablePrompt] = Field(None, description="Reference to a prompt template and its variables.")
    reasoning: Optional[ReasoningConfig] = Field(None, description="Configuration options for reasoning models.")
    text: Optional[object] = Field(None, description="Configuration options for a text response from the model.")
    truncation: Optional[Literal["auto", "disabled"]] = Field(None, description="The truncation strategy to use for the model response.")

    def set_input_messages(self, messages: List[Message]) -> None:
        self.input = [m.serialize() for m in messages]

    def set_output_structure(self, output_type: type[T]) -> None:
        schema = type_to_json_schema(output_type)
        self.text = {
            "format": {
                "type": "json_schema",
                "name": output_type.__name__,
                "schema": schema,
                "strict": True
            }
        }

class CompletionsRequest(TestGenerationRequest):
    messages: List[Dict[str, str]] = None
    frequency_penalty: Optional[float] = Field(None, ge=-2, le=2, description="Number between -2.0 and 2.0. Positive values penalize new tokens based on their existing frequency in the text so far, decreasing the model's likelihood to repeat the same line verbatim.")
    logit_bias: Optional[Dict] = Field(None, description="Modify the likelihood of specified tokens appearing in the completion.")
    logprobs: Optional[bool] = Field(None, description="Whether to return log probabilities of the output tokens or not.")
    max_completion_tokens: Optional[int] = Field(None, gt=0, description="An upper bound for the number of tokens that can be generated for a completion, including visible output tokens and reasoning tokens.")
    modalities: Optional[List[str]] = Field(None, description="Output types that you would like the model to generate.")
    n: Optional[int] = Field(None, description="How many chat completion choices to generate for each input message.")
    prediction: Optional[object] = Field(None, description="Configuration for a Predicted Output, which can greatly improve response times when large parts of the model response are known ahead of time.")
    presence_penalty: Optional[float] = Field(None, ge=-2, le=2, description="Number between -2.0 and 2.0. Positive values penalize new tokens based on whether they appear in the text so far, increasing the model's likelihood to talk about new topics.")
    reasoning_effort: Optional[Literal["minimal", "low", "medium", "high"]] = Field(None, description="Constrains effort on reasoning for reasoning models.")
    response_format: Optional[Dict] = Field(None, description="An object specifying the format that the model must output.")
    verbosity: Optional[Literal["low", "medium", "high"]] = Field(None, description="Constrains the verbosity of the model's response.")
    web_search_options: Optional[object] = Field(None, description="This tool searches the web for relevant results to use in a response.")

    def set_input_messages(self, messages: List[Message]) -> None:
        self.messages = [m.serialize() for m in messages]

    def set_output_structure(self, output_type: type[T]) -> None:
        schema = type_to_json_schema(output_type)
        self.response_format = {
            "format": {
                "type": "json_schema",
                "name": output_type.__name__,
                "schema": schema,
                "strict": True
            }
        }

class EmbeddingsRequest(BaseRequest):
    input: Union[str | List[str]]
    dimensions: Optional[int] = Field(None, ge=1, description="The number of dimensions the resulting output embeddings should have. Only supported in text-embedding-3 and later models.")
    encoding_format: Optional[Literal["base64", "float"]] = Field(None, description="The format to return the embeddings in. Can be either float or base64.")
    user: Optional[str] = Field(None, description="A unique identifier representing your end-user, which can help OpenAI to monitor and detect abuse. ")

    def set_input(self, inp: Union[str | List[str]]) -> None:
        self.input = inp
