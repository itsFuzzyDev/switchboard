from schemas.base import Schema, transform
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


class OllamaInputSchema(Schema):
    def to_provider(self, data: dict, provider: str) -> dict:
        if provider == Provider.ollama:
            return data
        if provider == Provider.claude:
            out = transform(data, OLLAMA_TO_OPENAI)
            effort = data.get("options", {}).get("reasoning_effort")
            if effort:
                out["thinking"] = {"type": "enabled", "budget_tokens": 16000 if effort in ("high", "max") else 4000}
            return out
        return transform(data, OLLAMA_TO_OPENAI)

    def from_provider(self, raw: dict, provider: str) -> dict:
        raise NotImplementedError("Use OutputSchema")