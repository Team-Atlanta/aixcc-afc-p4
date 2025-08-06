from typing import Literal, TypeAlias, TypedDict


class SystemMessage(TypedDict):
    role: Literal["system"]
    content: str


class UserMessage(TypedDict):
    role: Literal["user"]
    content: str


class AssistantMessage(TypedDict):
    role: Literal["assistant"]
    content: str


BaseMessage = SystemMessage | UserMessage | AssistantMessage

Prompt: TypeAlias = list[BaseMessage]
Completion: TypeAlias = AssistantMessage
