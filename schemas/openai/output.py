from schemas.base import Schema, transform

_STREAM = {
    "model": "model",
    "choices[0].delta.content": "message.content",
    "choices[0].delta.role": "message.role",
    "choices[0].delta.tool_calls": "message.tool_calls",
    "choices[0].finish_reason": "stop_reason",
    "id": "id",
}

_FULL = {
    "model": "model",
    "choices[0].message.content": "message.content",
    "choices[0].message.role": "message.role",
    "choices[0].message.tool_calls": "message.tool_calls",
    "choices[0].finish_reason": "stop_reason",
    "usage.prompt_tokens": "prompt_eval_count",
    "usage.completion_tokens": "eval_count",
    "id": "id",
}


class OpenAIOutputSchema(Schema):
    def to_provider(self, data: dict, provider: str) -> dict:
        raise NotImplementedError("Use InputSchema")

    def from_provider(self, raw: dict, provider: str) -> dict:
        if raw.get("choices") and raw["choices"][0].get("delta"):
            return transform(raw, _STREAM)
        return transform(raw, _FULL)
