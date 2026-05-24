from schemas.base import Schema, transform

class OpenAIOutputSchema(Schema):
    def to_provider(self, data: dict, provider: str) -> dict:
        raise NotImplementedError("Use InputSchema")

    def from_provider(self, raw: dict, provider: str) -> dict:
        # OpenAI → Ollama
        if raw.get("choices") and raw["choices"][0].get("delta"):
            # stream chunk
            return transform(raw, {
                "model": "model",
                "choices[0].delta.content": "message.content",
                "choices[0].delta.role": "message.role",
                "choices[0].delta.tool_calls": "message.tool_calls",
                "choices[0].finish_reason": "stop_reason",
                "id": "id",
            })
        return transform(raw, {
            "model": "model",
            "choices[0].message.content": "message.content",
            "choices[0].message.role": "message.role",
            "choices[0].message.tool_calls": "message.tool_calls",
            "choices[0].finish_reason": "stop_reason",
            "usage.prompt_tokens": "prompt_eval_count",
            "usage.completion_tokens": "eval_count",
            "id": "id",
        })