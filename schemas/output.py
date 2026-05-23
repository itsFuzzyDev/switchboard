from __future__ import annotations
from datetime import datetime, timezone
from typing import Any
from pydantic import BaseModel, Field

from schemas.input import Provider

class OllamaMessage(BaseModel):
    role: str = "assistant"
    content: str

class OllamaResponse(BaseModel):
    model: str
    created_at: datetime | None = Field(default_factory=lambda: datetime.now(timezone.utc))
    message: OllamaMessage
    done: bool = True
    total_duration: int | None = None
    load_duration: int | None = None
    prompt_eval_count: int | None = None
    prompt_eval_duration: int | None = None
    eval_count: int | None = None
    eval_duration: int | None = None
    reasoning_content: str | None = None

    @classmethod
    def from_provider(cls, raw: dict[str, Any], provider: Provider, model: str = "") -> OllamaResponse:
        """Parse a provider's raw JSON response into a normalized OllamaResponse."""
        fn = _PARSERS.get(provider)
        if not fn:
            raise ValueError(f"unsupported provider: {provider}")
        return fn(raw, model)

class OllamaStreamChunk(BaseModel):
    model: str | None = None
    created_at: datetime | None = Field(default_factory=lambda: datetime.now(timezone.utc))
    message: OllamaMessage | None = None
    done: bool = False
    reasoning_content: str | None = None

    @classmethod
    def from_provider_chunk(cls, raw: dict[str, Any], provider: Provider, model: str = "") -> OllamaStreamChunk:
        """Parse a single streaming chunk from a provider into a normalized OllamaStreamChunk."""
        fn = _STREAM_PARSERS.get(provider)
        if not fn:
            raise ValueError(f"unsupported provider: {provider}")
        return fn(raw, model)

# ---------------------------------------------------------------------------
# non-streaming parsers
# ---------------------------------------------------------------------------

def _extract_openai_message(raw: dict) -> dict:
    """Pull the first choice's message dict out of an OpenAI-style response."""
    choices = raw.get("choices", [])
    return choices[0].get("message", {}) if choices else {}

def _extract_openai_usage(raw: dict) -> dict:
    """Pull the usage object out of an OpenAI-style response."""
    return raw.get("usage", {})

def _parse_openai_response(raw: dict, model: str) -> OllamaResponse:
    """Parser for any provider that returns an OpenAI chat-completions shaped response."""
    msg = _extract_openai_message(raw)
    usage = _extract_openai_usage(raw)
    return OllamaResponse(
        model=raw.get("model") or model,
        message=OllamaMessage(role="assistant", content=msg.get("content") or ""),
        prompt_eval_count=usage.get("prompt_tokens"),
        eval_count=usage.get("completion_tokens"),
        reasoning_content=msg.get("reasoning_content") or raw.get("reasoning") or None,
    )

def _parse_ollama_response(raw: dict, model: str) -> OllamaResponse:
    """Ollama responses are already in our target schema — just validate them."""
    return OllamaResponse.model_validate(raw)

def _parse_claude_response(raw: dict, model: str) -> OllamaResponse:
    """Claude returns content blocks: text blocks go to content, thinking blocks go to reasoning_content."""
    blocks = raw.get("content", [])
    texts = [b.get("text", "") for b in blocks if b.get("type") == "text"]
    thinking = [b.get("thinking", "") for b in blocks if b.get("type") == "thinking"]
    usage = raw.get("usage", {})
    return OllamaResponse(
        model=raw.get("model") or model,
        message=OllamaMessage(role="assistant", content="".join(texts)),
        prompt_eval_count=usage.get("input_tokens"),
        eval_count=usage.get("output_tokens"),
        reasoning_content="".join(thinking) or None,
    )

def _parse_gemini_response(raw: dict, model: str) -> OllamaResponse:
    """Gemini returns candidates with parts; we concatenate all text parts."""
    candidates = raw.get("candidates", [])
    if not candidates:
        content = ""
    else:
        parts = candidates[0].get("content", {}).get("parts", [])
        content = "".join(p.get("text", "") for p in parts)
    usage = raw.get("usageMetadata", {})
    return OllamaResponse(
        model=model,
        message=OllamaMessage(role="assistant", content=content),
        prompt_eval_count=usage.get("promptTokenCount"),
        eval_count=usage.get("candidatesTokenCount"),
    )

# dispatch table for full responses
_PARSERS: dict[Provider, Any] = {
    Provider.ollama: _parse_ollama_response,
    Provider.openai: _parse_openai_response,
    Provider.groq: _parse_openai_response,
    Provider.deepseek: _parse_openai_response,
    Provider.mistral: _parse_openai_response,
    Provider.openrouter: _parse_openai_response,
    Provider.xai: _parse_openai_response,
    Provider.claude: _parse_claude_response,
    Provider.gemini: _parse_gemini_response,
}

# ---------------------------------------------------------------------------
# streaming parsers — each turns one SSE/NDJSON chunk into an OllamaStreamChunk
# ---------------------------------------------------------------------------

def _parse_openai_stream_chunk(raw: dict, model: str) -> OllamaStreamChunk:
    """OpenAI-style streaming: delta object carries the next token; finish_reason signals completion."""
    choices = raw.get("choices", [])
    delta = choices[0].get("delta", {}) if choices else {}
    finish_reason = choices[0].get("finish_reason") if choices else None
    return OllamaStreamChunk(
        model=raw.get("model") or model,
        message=OllamaMessage(role="assistant", content=delta.get("content") or ""),
        done=finish_reason is not None,
        reasoning_content=delta.get("reasoning_content") or None,
    )

def _parse_ollama_stream_chunk(raw: dict, model: str) -> OllamaStreamChunk:
    """Ollama streaming chunks already contain a message object and a done flag."""
    msg = raw.get("message", {})
    return OllamaStreamChunk(
        model=raw.get("model") or model,
        message=OllamaMessage(role=msg.get("role", "assistant"), content=msg.get("content", "")),
        done=raw.get("done", False),
    )

def _parse_claude_stream_chunk(raw: dict, model: str) -> OllamaStreamChunk:
    """Claude streams content_block_delta events. Text deltas become content; thinking deltas become reasoning."""
    event_type = raw.get("type", "")
    delta = raw.get("delta", {})

    if event_type == "content_block_delta":
        if delta.get("type") == "text_delta":
            return OllamaStreamChunk(
                model=model, message=OllamaMessage(content=delta.get("text", "")), done=False
            )
        if delta.get("type") == "thinking_delta":
            return OllamaStreamChunk(
                model=model, message=OllamaMessage(content=""),
                reasoning_content=delta.get("thinking", ""), done=False
            )

    # message_delta with a stop_reason means the stream is finished
    if event_type == "message_delta" and raw.get("delta", {}).get("stop_reason"):
        return OllamaStreamChunk(model=model, done=True)

    # Fallback for any unhandled event shapes
    return OllamaStreamChunk(model=model, message=OllamaMessage(content=""), done=False)

def _parse_gemini_stream_chunk(raw: dict, model: str) -> OllamaStreamChunk:
    """Gemini stream chunks look like the non-streaming response but arrive incrementally."""
    candidates = raw.get("candidates", [])
    if not candidates:
        return OllamaStreamChunk(model=model, message=OllamaMessage(content=""), done=False)
    parts = candidates[0].get("content", {}).get("parts", [])
    text = "".join(p.get("text", "") for p in parts)
    return OllamaStreamChunk(model=model, message=OllamaMessage(content=text), done=False)

# dispatch table for stream chunks
_STREAM_PARSERS: dict[Provider, Any] = {
    Provider.ollama: _parse_ollama_stream_chunk,
    Provider.openai: _parse_openai_stream_chunk,
    Provider.groq: _parse_openai_stream_chunk,
    Provider.deepseek: _parse_openai_stream_chunk,
    Provider.mistral: _parse_openai_stream_chunk,
    Provider.openrouter: _parse_openai_stream_chunk,
    Provider.xai: _parse_openai_stream_chunk,
    Provider.claude: _parse_claude_stream_chunk,
    Provider.gemini: _parse_gemini_stream_chunk,
}
