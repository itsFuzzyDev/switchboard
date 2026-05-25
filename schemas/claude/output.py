from schemas.base import OutputSchema, transform


class ClaudeOutputSchema(OutputSchema):
    def from_provider(self, raw: dict, provider: str) -> dict:
        out = transform(raw, {
            "model": "model",
            "stop_reason": "done_reason",
            "usage.input_tokens": "prompt_eval_count",
            "usage.output_tokens": "eval_count",
            "id": "extra.id",
        })
        blocks = raw.get("content", [])
        texts = [b.get("text", "") for b in blocks if b.get("type") == "text"]
        thinking = "".join(b.get("thinking", "") for b in blocks if b.get("type") == "thinking")
        out["message"] = {"role": "assistant", "content": "".join(texts)}
        if thinking:
            out["message"]["thinking"] = thinking
        return out
