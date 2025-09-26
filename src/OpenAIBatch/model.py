from dataclasses import dataclass
from typing import List, Dict


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

@dataclass
class ReusablePrompt:
    id: str
    version: str

@dataclass
class InputInstance:
    id: str
    prompt_value_mapping: Dict[str, str]
