from schemas.base import InputSchema


class ClaudeInputSchema(InputSchema):
    def to_provider(self, data: dict, provider: str) -> dict:
        msgs = []
        system = None
        for m in data.get("messages", []):
            if m.get("role") == "system":
                system = m.get("content")
            else:
                msgs.append({"role": m.get("role"), "content": m.get("content")})
        out = {"model": data.get("model"), "messages": msgs, "stream": data.get("stream", False), "max_tokens": data.get("options", {}).get("max_tokens", 4096)}
        if system:
            out["system"] = system
        effort = data.get("options", {}).get("reasoning_effort")
        if effort:
            out["thinking"] = {"type": "enabled", "budget_tokens": 16000 if effort in ("high", "max") else 4000}
        return out
