from schemas.base import Schema, transform
from schemas.provider import Provider

OPENAI_TO_OLLAMA = {
    "model": "model",
    "choices[0].message.content": "message.content",
    "choices[0].message.role": "message.role",
    "choices[0].message.tool_calls": "message.tool_calls",
    "choices[0].finish_reason": "stop_reason",
    "usage.prompt_tokens": "prompt_eval_count",
    "usage.completion_tokens": "eval_count",
    "id": "id",
}

OPENAI_STREAM_TO_OLLAMA = {
    "model": "model",
    "choices[0].delta.content": "message.content",
    "choices[0].delta.role": "message.role",
    "choices[0].delta.tool_calls": "message.tool_calls",
    "choices[0].finish_reason": "stop_reason",
    "id": "id",
}

CLAUDE_TO_OLLAMA = {
    "model": "model",
    "stop_reason": "stop_reason",
    "usage.input_tokens": "prompt_eval_count",
    "usage.output_tokens": "eval_count",
    "id": "id",
}

GEMINI_TO_OLLAMA = {
    "model": "model",
    "usageMetadata.promptTokenCount": "prompt_eval_count",
    "usageMetadata.candidatesTokenCount": "eval_count",
}


class OllamaOutputSchema(Schema):
    def to_provider(self, data: dict, provider: str) -> dict:
        raise NotImplementedError("Use InputSchema")

    def from_provider(self, raw: dict, provider: str) -> dict:
        if provider == Provider.ollama:
            return transform(raw, {"message.content": "message.content", "message.role": "message.role", "done": "done", "prompt_eval_count": "prompt_eval_count", "eval_count": "eval_count", "model": "model"})
        if provider == Provider.claude:
            out = transform(raw, CLAUDE_TO_OLLAMA)
            blocks = raw.get("content", [])
            texts = [b.get("text", "") for b in blocks if b.get("type") == "text"]
            thinking = "".join(b.get("thinking", "") for b in blocks if b.get("type") == "thinking")
            out["message"] = {"role": "assistant", "content": "".join(texts)}
            out["reasoning_content"] = thinking or None
            return out
        if provider == Provider.gemini:
            out = transform(raw, GEMINI_TO_OLLAMA)
            parts = raw.get("candidates", [{}])[0].get("content", {}).get("parts", [])
            out["message"] = {"role": "model", "content": "".join(p.get("text", "") for p in parts)}
            return out
        if raw.get("choices") and raw["choices"][0].get("delta"):
            return transform(raw, OPENAI_STREAM_TO_OLLAMA)
        return transform(raw, OPENAI_TO_OLLAMA)