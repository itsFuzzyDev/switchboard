from schemas.base import Schema, transform

class GeminiOutputSchema(Schema):
    def to_provider(self, data: dict, provider: str) -> dict:
        raise NotImplementedError("Use InputSchema")

    def from_provider(self, raw: dict, provider: str) -> dict:
        # Gemini → Ollama
        out = transform(raw, {
            "model": "model",
            "usageMetadata.promptTokenCount": "prompt_eval_count",
            "usageMetadata.candidatesTokenCount": "eval_count",
        })
        parts = raw.get("candidates", [{}])[0].get("content", {}).get("parts", [])
        out["message"] = {"role": "model", "content": "".join(p.get("text", "") for p in parts)}
        return out