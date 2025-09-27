from abc import abstractmethod, ABC
from typing import List, Dict, Any, Optional, Literal, TypeVar, Union

from pydantic import BaseModel, Field

from openbatch._utils import type_to_json_schema


T = TypeVar("T", bound=BaseModel)

class Message(BaseModel):
    """
    Represents a single message in a conversation or prompt.

    Attributes:
        role (str): The role of the message sender (e.g., "user", "assistant", "system").
        content (str): The text content of the message.
    """
    role: str
    content: str

    def serialize(self):
        """
        Converts the Message instance into a dictionary suitable for API requests.

        Returns:
            Dict[str, str]: A dictionary with 'role' and 'content' keys.
        """
        return {"role": self.role, "content": self.content}

class PromptTemplate(BaseModel):
    """
    A template containing a sequence of messages, where the content can contain
    placeholders for string formatting.

    Attributes:
        messages (List[Message]): A list of Message objects that form the template.
    """
    messages: List[Message]

    def format(self, **kwargs) -> List[Message]:
        """
        Formats the content of each message in the template using the provided keyword arguments.

        Args:
            **kwargs: Keyword arguments used to substitute placeholders in message content.

        Returns:
            List[Message]: A new list of Message objects with formatted content.
        """
        formatted_messages = []
        for message in self.messages:
            formatted_content = message.content.format(**kwargs)
            formatted_messages.append(Message(role=message.role, content=formatted_content))
        return formatted_messages

class ReusablePrompt(BaseModel):
    """
    References a reusable prompt template and its associated variables.

    Attributes:
        id (str): The unique identifier of the reusable prompt template.
        version (str): The specific version of the prompt template to use.
        variables (Dict[str, Any]): A dictionary of variable names and their values
                                     to be used when formatting the prompt.
    """
    id: str
    version: str
    variables: Dict[str, Any]

class ReasoningConfig(BaseModel):
    """
    Configuration options for reasoning models.

    Attributes:
       effort (Literal["minimal", "low", "medium", "high"]): Constrains effort on reasoning
           for reasoning models. Defaults to "medium".
       summary (Optional[Literal["auto", "concise", "detailed"]]): A summary of the
           reasoning performed by the model. Optional.
    """
    effort: Literal["minimal", "low", "medium", "high"] = Field(default="medium", description="Constrains effort on reasoning for reasoning models.")
    summary: Optional[Literal["auto", "concise", "detailed"]] = Field(None, description="A summary of the reasoning performed by the model.")

class InputInstance(BaseModel):
    """
    Base class for defining a single input instance for a batch job.

    Attributes:
        id (str): Unique identifier of the input instance.
        instance_request_options (Optional[Dict[str, Any]]): Options specific to the
            input instance that can be set in the API request. Optional.
    """
    id: str = Field(description="Unique identifier of the input instance.")
    instance_request_options: Optional[Dict[str, Any]] = Field(None, description="Options specific to the input instance that to set in the request.")

class MessagesInputInstance(InputInstance):
    """
    An input instance defined by a list of messages.

    Attributes:
        id (str): Unique identifier of the input instance.
        messages (List[Message]): List of messages to be sent to the model for this instance.
        instance_request_options (Optional[Dict[str, Any]]): Options specific to the
            input instance that can be set in the API request. Optional.
    """
    messages: List[Message] = Field(description="List of messages to be sent to the model.")

class PromptTemplateInputInstance(InputInstance):
    """
    An input instance defined by mapping values to variables in a prompt template.

    Attributes:
        id (str): Unique identifier of the input instance.
        prompt_value_mapping (Dict[str, str]): Mapping of prompt variable names
            to their values for this instance.
        instance_request_options (Optional[Dict[str, Any]]): Options specific to the
            input instance that can be set in the API request. Optional.
    """
    prompt_value_mapping: Dict[str, str] = Field(description="Mapping of prompt variable names to their values.")

class EmbeddingInputInstance(InputInstance):
    """
    An input instance specifically for embedding requests.

    Attributes:
        id (str): Unique identifier of the input instance.
        input (Union[str, List[str]]): The text or list of texts to be embedded for this instance.
        instance_request_options (Optional[Dict[str, Any]]): Options specific to the
            input instance that can be set in the API request. Optional.
    """
    input: Union[str, List[str]] = Field(description="Text(s) to be embedded.")

class RequestStrategy(ABC):
    """
    Abstract base class defining the strategy for creating a request
    for a specific API endpoint.
    """
    @property
    @abstractmethod
    def url(self) -> str:
        """
        Abstract property for the specific API endpoint URL path.

        Returns:
            str: The URL path for the API.
        """
        pass

    def create_request(self, custom_id: str, body: Dict[str, Any]) -> Dict[str, Any]:
        """
        Creates a structured request dictionary for a batch job.

        Args:
            custom_id (str): A unique identifier for the request.
            body (Dict[str, Any]): The API-specific request body content.

        Returns:
            Dict[str, Any]: A dictionary representing the complete request structure.
        """
        return {
            "custom_id": custom_id,
            "method": "POST",
            "url": self.url,
            "body": body
        }

class ResponsesAPIStrategy(RequestStrategy):
    """Strategy for creating requests to the /v1/responses endpoint."""
    @property
    def url(self) -> str:
        return "/v1/responses"

class ChatCompletionsAPIStrategy(RequestStrategy):
    """Strategy for creating requests to the /v1/chat/completions endpoint."""
    @property
    def url(self) -> str:
        return "/v1/chat/completions"

class EmbeddingsAPIStrategy(RequestStrategy):
    """Strategy for creating requests to the /v1/embeddings endpoint."""
    @property
    def url(self) -> str:
        return "/v1/embeddings"

class BaseRequest(BaseModel, ABC):
    """
    Abstract base class for API-specific request configurations (job configurations).

    Attributes:
        model (str): Model ID used to generate the response, like "gpt-4.1". Defaults to "gpt-4.1".
    """
    model: str = Field("gpt-4.1", description="Model ID used to generate the response, like gpt-4o or o3.")

    def to_dict(self) -> Dict[str, Any]:
        """
        Converts the request configuration object to a dictionary, excluding fields that are None.

        Returns:
            Dict[str, Any]: The configuration as a dictionary.
        """
        return self.model_dump(exclude_none=True)

class TextGenerationRequest(BaseRequest, ABC):
    """
    Abstract base class for text generation requests, including common parameters
    for various text generation endpoints.

    Attributes:
        model (str): Model ID used to generate the response, like "gpt-4.1". Defaults to "gpt-4.1".
        tools (Optional[List[object]]): An array of tools the model may call.
        top_p (Optional[float]): An alternative to sampling with temperature (nucleus sampling).
        parallel_tool_calls (Optional[bool]): Whether to allow parallel tool calls.
        prompt_cache_key (Optional[str]): Used by OpenAI to cache responses.
        safety_identifier (Optional[str]): A stable identifier for policy monitoring.
        service_tier (Optional[Literal["auto", "default", "flex", "priority"]]): Specifies the processing type.
        store (Optional[bool]): Whether to store the generated model response.
        temperature (Optional[float]): Sampling temperature to use (0 to 2).
        tool_choice (Optional[str | object]): How the model should select which tool to use.
        top_logprobs (Optional[int]): Number of most likely tokens to return at each position (0 to 20).
    """
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


class ResponsesRequest(TextGenerationRequest):
    """
        Configuration for a /v1/responses API request.

        Attributes:
            model (str): Model ID used to generate the response, like "gpt-4.1". Defaults to "gpt-4.1".
            conversation (Optional[str]): The conversation this response belongs to.
            include (Optional[List[Literal[...]]]): Specify additional output data to include.
            input (Optional[str | List[Dict[str, str]]]): Text, image, or file inputs to the model.
            instructions (Optional[str]): A system or developer message.
            max_output_tokens (Optional[int]): Upper bound for generated tokens.
            max_tool_calls (Optional[int]): Maximum number of tool calls allowed.
            previous_response_id (Optional[str]): ID of the previous response for multi-turn.
            prompt (Optional[ReusablePrompt]): Reference to a prompt template and its variables.
            reasoning (Optional[ReasoningConfig]): Configuration for reasoning models.
            text (Optional[object]): Configuration options for a text response from the model (e.g., JSON schema).
            truncation (Optional[Literal["auto", "disabled"]]): The truncation strategy to use.
            tools (Optional[List[object]]): An array of tools the model may call.
            top_p (Optional[float]): An alternative to sampling with temperature (nucleus sampling).
            parallel_tool_calls (Optional[bool]): Whether to allow parallel tool calls.
            prompt_cache_key (Optional[str]): Used by OpenAI to cache responses.
            safety_identifier (Optional[str]): A stable identifier for policy monitoring.
            service_tier (Optional[Literal["auto", "default", "flex", "priority"]]): Specifies the processing type.
            store (Optional[bool]): Whether to store the generated model response.
            temperature (Optional[float]): Sampling temperature to use (0 to 2).
            tool_choice (Optional[str | object]): How the model should select which tool to use.
            top_logprobs (Optional[int]): Number of most likely tokens to return at each position (0 to 20).
        """
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

class ChatCompletionsRequest(TextGenerationRequest):
    """
    Configuration for a /v1/chat/completions API request.

    Attributes:
        model (str): Model ID used to generate the response, like "gpt-4.1". Defaults to "gpt-4.1".
        messages (List[Dict[str, str]]): A list of messages in the conversation.
        frequency_penalty (Optional[float]): Penalizes new tokens based on frequency (-2.0 to 2.0).
        logit_bias (Optional[Dict]): Modifies the likelihood of specified tokens.
        logprobs (Optional[bool]): Whether to return log probabilities.
        max_completion_tokens (Optional[int]): Upper bound for generated completion tokens.
        modalities (Optional[List[str]]): Output types the model should generate.
        n (Optional[int]): How many chat completion choices to generate.
        prediction (Optional[object]): Configuration for a Predicted Output.
        presence_penalty (Optional[float]): Penalizes new tokens based on presence (-2.0 to 2.0).
        reasoning_effort (Optional[Literal["minimal", "low", "medium", "high"]]): Constrains reasoning effort.
        response_format (Optional[Dict]): Specifies the format that the model must output (e.g., JSON schema).
        verbosity (Optional[Literal["low", "medium", "high"]]): Constrains the response verbosity.
        web_search_options (Optional[object]): Configuration for the web search tool.
        tools (Optional[List[object]]): An array of tools the model may call.
        top_p (Optional[float]): An alternative to sampling with temperature (nucleus sampling).
        parallel_tool_calls (Optional[bool]): Whether to allow parallel tool calls.
        prompt_cache_key (Optional[str]): Used by OpenAI to cache responses.
        safety_identifier (Optional[str]): A stable identifier for policy monitoring.
        service_tier (Optional[Literal["auto", "default", "flex", "priority"]]): Specifies the processing type.
        store (Optional[bool]): Whether to store the generated model response.
        temperature (Optional[float]): Sampling temperature to use (0 to 2).
        tool_choice (Optional[str | object]): How the model should select which tool to use.
        top_logprobs (Optional[int]): Number of most likely tokens to return at each position (0 to 20).
    """
    messages: List[Dict[str, str]] = Field(None, description="A list of messages comprising the conversation so far.")
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
    """
    Configuration for a /v1/embeddings API request.

    Attributes:
        model (str): Model ID used to generate the response, like "text-embedding-3-small".
        input (Union[str | List[str]]): Input text or array of tokens to embed.
        dimensions (Optional[int]): The desired number of dimensions for the resulting embeddings.
        encoding_format (Optional[Literal["base64", "float"]]): The format to return the embeddings in.
        user (Optional[str]): A unique identifier representing the end-user.
    """
    input: Union[str | List[str]] = Field(None, description="Input text to embed, encoded as a string or array of tokens.")
    dimensions: Optional[int] = Field(None, ge=1, description="The number of dimensions the resulting output embeddings should have. Only supported in text-embedding-3 and later models.")
    encoding_format: Optional[Literal["base64", "float"]] = Field(None, description="The format to return the embeddings in. Can be either float or base64.")
    user: Optional[str] = Field(None, description="A unique identifier representing your end-user, which can help OpenAI to monitor and detect abuse. ")

    def set_input(self, inp: Union[str | List[str]]) -> None:
        self.input = inp
