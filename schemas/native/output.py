from schemas.base import OutputSchema, transform

OLLAMA_TO_NATIVE = {
    "model": "model",
    "message.content": "message.content",
    "message.role": "message.role",
    "message.thinking": "message.thinking",
    "message.tool_calls": "message.tool_calls",
    "done": "done",
    "done_reason": "done_reason",
    "total_duration": "total_duration",
    "load_duration": "load_duration",
    "prompt_eval_count": "prompt_eval_count",
    "prompt_eval_duration": "prompt_eval_duration",
    "eval_count": "eval_count",
    "eval_duration": "eval_duration",
    "logprobs": "logprobs",
}


class OllamaOutputSchema(OutputSchema):
    def from_provider(self, raw: dict, provider: str) -> dict:
        return transform(raw, OLLAMA_TO_NATIVE)
