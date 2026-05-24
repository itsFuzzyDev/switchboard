from schemas.base import InputSchema


class OpenAIInputSchema(InputSchema):
    def to_provider(self, data: dict, provider: str) -> dict:
        out = {"model": data.get("model"), "messages": data.get("messages", []), "stream": data.get("stream", False)}
        opts = data.get("options", {})
        if opts.get("temperature") is not None: out["temperature"] = opts["temperature"]
        if opts.get("max_tokens") is not None: out["max_tokens"] = opts["max_tokens"]
        if opts.get("top_p") is not None: out["top_p"] = opts["top_p"]
        if opts.get("stop") is not None: out["stop"] = opts["stop"]
        if opts.get("frequency_penalty") is not None: out["frequency_penalty"] = opts["frequency_penalty"]
        if opts.get("presence_penalty") is not None: out["presence_penalty"] = opts["presence_penalty"]
        if opts.get("seed") is not None: out["seed"] = opts["seed"]
        if opts.get("reasoning_effort") is not None: out["reasoning_effort"] = opts["reasoning_effort"]
        if data.get("format"): out["response_format"] = data["format"]
        if data.get("tools"): out["tools"] = data["tools"]
        if data.get("tool_choice"): out["tool_choice"] = data["tool_choice"]
        return out
