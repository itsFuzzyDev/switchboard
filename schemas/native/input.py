from schemas.base import InputSchema, transform
from schemas.provider import Provider

OLLAMA_TO_OPENAI = {
    "model": "model",
    "messages": "messages",
    "stream": "stream",
    "options.temperature": "temperature",
    "options.max_tokens": "max_tokens",
    "options.top_p": "top_p",
    "options.stop": "stop",
    "options.frequency_penalty": "frequency_penalty",
    "options.presence_penalty": "presence_penalty",
    "options.seed": "seed",
    "options.reasoning_effort": "reasoning_effort",
    "format": "response_format",
    "tools": "tools",
    "tool_choice": "tool_choice",
}


class OllamaInputSchema(InputSchema):
    def to_provider(self, data: dict, provider: str) -> dict:
        if provider == Provider.ollama:
            return data
        return transform(data, OLLAMA_TO_OPENAI)
