from schemas.base import Schema

class GeminiInputSchema(Schema):
    def to_provider(self, data: dict, provider: str) -> dict:
        # Ollama → Gemini
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
        return out

    def from_provider(self, raw: dict, provider: str) -> dict:
        raise NotImplementedError("Use OutputSchema")