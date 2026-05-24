from schemas.base import OutputSchema, transform

OLLAMA_TO_OLLAMA = {
    "model": "model",
    "message.content": "message.content",
    "message.role": "message.role",
    "done": "done",
    "prompt_eval_count": "prompt_eval_count",
    "eval_count": "eval_count",
}


class OllamaOutputSchema(OutputSchema):
    def from_provider(self, raw: dict, provider: str) -> dict:
        return transform(raw, OLLAMA_TO_OLLAMA)
