from schemas.base import InputSchema


class GeminiInputSchema(InputSchema):
    def to_provider(self, data: dict, provider: str) -> dict:
        contents = []
        system_instruction = None
        for m in data.get("messages", []):
            if m.get("role") == "system":
                system_instruction = {"parts": [{"text": m.get("content")}]}
            else:
                contents.append({"role": "user" if m.get("role") == "user" else "model", "parts": [{"text": m.get("content")}]})
        out = {"contents": contents}
        if system_instruction:
            out["systemInstruction"] = system_instruction
        opts = data.get("options", {})
        gen_config = {}
        if opts.get("temperature") is not None: gen_config["temperature"] = opts["temperature"]
        if opts.get("max_tokens") is not None: gen_config["maxOutputTokens"] = opts["max_tokens"]
        if opts.get("top_p") is not None: gen_config["topP"] = opts["top_p"]
        if opts.get("stop") is not None: gen_config["stopSequences"] = opts["stop"] if isinstance(opts["stop"], list) else [opts["stop"]]
        if gen_config:
            out["generationConfig"] = gen_config
        return out
