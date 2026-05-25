from typing import TypedDict, Any


class ToolCallFunction(TypedDict, total=False):
    name: str
    description: str
    arguments: dict[str, Any]


class ToolCall(TypedDict, total=False):
    function: ToolCallFunction


class LogprobTop(TypedDict, total=False):
    token: str
    logprob: float
    bytes: list[int]


class Logprob(TypedDict, total=False):
    token: str
    logprob: float
    bytes: list[int]
    top_logprobs: list[LogprobTop]


class Message(TypedDict, total=False):
    role: str
    content: str
    thinking: str
    tool_calls: list[ToolCall]
    images: list[str]


class NativeResponse(TypedDict, total=False):
    """Normalized shape matching the Ollama chat API response."""
    model: str
    created_at: str
    message: Message
    done: bool
    done_reason: str
    total_duration: int
    load_duration: int
    prompt_eval_count: int
    prompt_eval_duration: int
    eval_count: int
    eval_duration: int
    logprobs: list[Logprob]
    extra: dict[str, Any]
