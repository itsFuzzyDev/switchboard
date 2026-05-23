from __future__ import annotations
from enum import Enum
from typing import Any, Literal
from pydantic import BaseModel, Field

class Provider(str, Enum):
    openai = "openai"
    ollama = "ollama"
    gemini = "gemini"
    groq = "groq"
    claude = "claude"
    deepseek = "deepseek"
    mistral = "mistral"
    openrouter = "openrouter"
    xai = "xai"

class OllamaMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str
    images: list[str] | None = None

class OllamaOptions(BaseModel):
    temperature: float | None = Field(0.7, ge=0, le=2)
    top_p: float | None = None
    top_k: int | None = None
    seed: int | None = None
    max_tokens: int | None = None
    stop: list[str] | None = None
    frequency_penalty: float | None = None
    presence_penalty: float | None = None
    reasoning_effort: Literal["low", "medium", "high", "max"] | None = None

class OllamaRequest(BaseModel):
    model: str
    messages: list[OllamaMessage]
    stream: bool = False
    options: OllamaOptions | None = None
    format: Literal["json"] | dict | None = None
    keep_alive: str | int | None = None
    tools: list[dict] | None = None
    tool_choice: str | dict | None = None

    def to_provider(self, provider: Provider) -> dict[str, Any]:
        """Convert this Ollama-style request into the target provider's native request body."""
        fn = _CONVERTERS.get(provider)
        if not fn:
            raise ValueError(f"unsupported provider: {provider}")
        return fn(self)

# ---------------------------------------------------------------------------
# helpers — shared building blocks used by multiple converters
# ---------------------------------------------------------------------------

def _build_openai_options(r: OllamaRequest) -> dict[str, Any]:
    """Map OllamaOptions fields to OpenAI-compatible parameter names."""
    d: dict[str, Any] = {}
    o = r.options
    if not o:
        return d
    mapping = {
        "temperature": o.temperature,
        "max_tokens": o.max_tokens,
        "top_p": o.top_p,
        "stop": o.stop,
        "frequency_penalty": o.frequency_penalty,
        "presence_penalty": o.presence_penalty,
        "seed": o.seed,
    }
    for k, v in mapping.items():
        if v is not None:
            d[k] = v
    return d

def _build_response_format(r: OllamaRequest) -> dict[str, Any] | None:
    """Turn the Ollama `format` field into an OpenAI-style response_format object."""
    if r.format == "json":
        return {"type": "json_object"}
    if isinstance(r.format, dict):
        return {
            "type": "json_schema",
            "json_schema": {"name": "schema", "schema": r.format, "strict": True},
        }
    return None

def _wrap_tools_openai(r: OllamaRequest) -> list[dict] | None:
    """Wrap each raw tool dict into OpenAI's {type: "function", function: ...} shape."""
    if not r.tools:
        return None
    return [{"type": "function", "function": t} for t in r.tools]

def _build_openai_request(r: OllamaRequest, extra: dict | None = None) -> dict:
    """Base builder for any provider that speaks the OpenAI chat-completions schema."""
    d: dict = {
        "model": r.model,
        "messages": [m.model_dump(exclude_none=True) for m in r.messages],
        "stream": r.stream,
    }
    d.update(_build_openai_options(r))

    fmt = _build_response_format(r)
    if fmt:
        d["response_format"] = fmt

    tools = _wrap_tools_openai(r)
    if tools:
        d["tools"] = tools

    if r.tool_choice:
        d["tool_choice"] = r.tool_choice
    if extra:
        d.update(extra)
    return d

# ---------------------------------------------------------------------------
# provider-specific converters — each returns the native request dict
# ---------------------------------------------------------------------------

def _ollama(r: OllamaRequest) -> dict:
    """Ollama native: keep the exact same shape but nest options under the "options" key."""
    d = r.model_dump(exclude_none=True)
    if r.options:
        d["options"] = r.options.model_dump(exclude_none=True)
    return d

def _openai(r: OllamaRequest) -> dict:
    return _build_openai_request(r)

def _groq(r: OllamaRequest) -> dict:
    """Groq is OpenAI-compatible plus a reasoning_effort flag on some models."""
    d = _build_openai_request(r)
    if r.options and r.options.reasoning_effort:
        d["reasoning_effort"] = r.options.reasoning_effort
    return d

def _deepseek(r: OllamaRequest) -> dict:
    """DeepSeek is OpenAI-compatible but remaps reasoning levels and requires a thinking toggle."""
    d = _build_openai_request(r)
    o = r.options
    if o and o.reasoning_effort:
        # DeepSeek only accepts "high" or "max"; map low/medium -> high
        effort = o.reasoning_effort
        d["reasoning_effort"] = {"low": "high", "medium": "high", "high": "high", "max": "max"}.get(effort, effort)
        d["thinking"] = {"type": "enabled"}
    return d

def _mistral(r: OllamaRequest) -> dict:
    """Mistral is OpenAI-compatible plus a reasoning_effort flag on reasoning models."""
    d = _build_openai_request(r)
    if r.options and r.options.reasoning_effort:
        d["reasoning_effort"] = r.options.reasoning_effort
    return d

def _openrouter(r: OllamaRequest) -> dict:
    """OpenRouter is OpenAI-compatible; preserve the exact model slug (can include provider prefixes)."""
    d = _build_openai_request(r)
    d["model"] = r.model
    return d

def _xai(r: OllamaRequest) -> dict:
    """xAI (Grok) is OpenAI-compatible."""
    return _build_openai_request(r)

def _claude(r: OllamaRequest) -> dict:
    """Anthropic Claude: system prompt moves to a top-level key; messages alternate user/assistant only."""
    msgs, system_text = [], None
    for m in r.messages:
        if m.role == "system":
            system_text = m.content
        else:
            msgs.append({"role": m.role, "content": m.content})

    d: dict = {"model": r.model, "messages": msgs, "stream": r.stream, "max_tokens": 4096}
    if system_text:
        d["system"] = system_text

    o = r.options
    if o:
        if o.max_tokens is not None:
            d["max_tokens"] = o.max_tokens
        if o.temperature is not None:
            d["temperature"] = o.temperature
        if o.top_p is not None:
            d["top_p"] = o.top_p
        if o.stop is not None:
            d["stop_sequences"] = o.stop
        if o.reasoning_effort:
            # Claude "thinking" mode needs a token budget
            budget = 16000 if o.reasoning_effort in ("high", "max") else 4000
            d["thinking"] = {"type": "enabled", "budget_tokens": budget}

    if r.tools:
        d["tools"] = [
            {"name": t.get("name"), "description": t.get("description"), "input_schema": t.get("parameters", t)}
            for t in r.tools
        ]
    if r.tool_choice:
        d["tool_choice"] = r.tool_choice
    return d

def _gemini(r: OllamaRequest) -> dict:
    """Google Gemini: messages become "contents" with role user/model; system becomes systemInstruction."""
    contents, system_instruction = [], None
    for m in r.messages:
        if m.role == "system":
            system_instruction = m.content
        else:
            # Gemini uses "user" and "model" roles
            gemini_role = "user" if m.role == "user" else "model"
            contents.append({"role": gemini_role, "parts": [{"text": m.content}]})

    d: dict = {"contents": contents}
    if system_instruction:
        d["systemInstruction"] = {"parts": [{"text": system_instruction}]}

    # Gemini-specific generationConfig mapping
    cfg: dict = {}
    o = r.options
    if o:
        if o.temperature is not None:
            cfg["temperature"] = o.temperature
        if o.top_p is not None:
            cfg["topP"] = o.top_p
        if o.max_tokens is not None:
            cfg["maxOutputTokens"] = o.max_tokens
        if o.stop is not None:
            cfg["stopSequences"] = o.stop
    if cfg:
        d["generationConfig"] = cfg

    if r.tools:
        d["tools"] = [{"functionDeclarations": [t]} for t in r.tools]
    return d

# ---------------------------------------------------------------------------
# dispatch table — maps each provider to its converter function
# ---------------------------------------------------------------------------
_CONVERTERS: dict[Provider, Any] = {
    Provider.ollama: _ollama,
    Provider.openai: _openai,
    Provider.groq: _groq,
    Provider.deepseek: _deepseek,
    Provider.mistral: _mistral,
    Provider.openrouter: _openrouter,
    Provider.xai: _xai,
    Provider.claude: _claude,
    Provider.gemini: _gemini,
}
