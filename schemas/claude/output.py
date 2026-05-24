from schemas.base import Schema, transform

class ClaudeOutputSchema(Schema):
    def to_provider(self, data: dict, provider: str) -> dict:
        raise NotImplementedError("Use InputSchema")

    def from_provider(self, raw: dict, provider: str) -> dict:
        # Claude → Ollama
        out = transform(raw, {
            "model": "model",
            "stop_reason": "stop_reason",
            "usage.input_tokens": "prompt_eval_count",
            "usage.output_tokens": "eval_count",
            "id": "id",
        })
        blocks = raw.get("content", [])
        texts = [b.get("text", "") for b in blocks if b.get("type") == "text"]
        thinking = "".join(b.get("thinking", "") for b in blocks if b.get("type") == "thinking")
        out["message"] = {"role": "assistant", "content": "".join(texts)}
        out["reasoning_content"] = thinking or None
        return out