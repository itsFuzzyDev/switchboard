from __future__ import annotations
import json, importlib
from typing import Any, Iterator

from schemas import Provider

# Base URLs for each provider
_ENDPOINTS = {
    Provider.openai: "https://api.openai.com/v1",
    Provider.groq: "https://api.groq.com/openai/v1",
    Provider.deepseek: "https://api.deepseek.com/v1",
    Provider.mistral: "https://api.mistral.ai/v1",
    Provider.openrouter: "https://openrouter.ai/api/v1",
    Provider.xai: "https://api.x.ai/v1",
    Provider.ollama: "http://localhost:11434",
    Provider.claude: "https://api.anthropic.com/v1",
    Provider.gemini: "https://generativelanguage.googleapis.com/v1/models",
}


def _chat_url(provider: Provider, base: str, model: str) -> str:
    if provider == Provider.gemini:
        return f"{base}/{model}:generateContent"
    if provider == Provider.claude:
        return f"{base}/messages"
    return f"{base}/chat/completions"


def _auth_headers(provider: Provider, api_key: str | None) -> dict[str, str]:
    h = {"content-type": "application/json"}
    if provider in (Provider.gemini, Provider.ollama):
        return h
    if api_key:
        h["authorization"] = f"Bearer {api_key}"
    return h


# ---------------------------------------------------------------------------
# thin HTTP wrapper
# ---------------------------------------------------------------------------

class _Http:
    def __init__(self) -> None:
        self._client: Any = None
        try:
            import httpx
            self._client = httpx.Client(timeout=120)
            self._mod = "httpx"
        except Exception:
            try:
                import requests
                self._client = requests.Session()
                self._mod = "requests"
            except Exception as exc:
                raise ImportError("Switchboard needs httpx or requests installed") from exc

    def _call(self, method: str, url: str, *, json: dict | None = None, headers: dict) -> dict:
        r = self._client.request(method, url, json=json, headers=headers)
        r.raise_for_status()
        return r.json()

    def post(self, url: str, *, json: dict, headers: dict) -> dict:
        return self._call("POST", url, json=json, headers=headers)

    def get(self, url: str, *, headers: dict) -> dict:
        return self._call("GET", url, headers=headers)

    def stream_post(self, url: str, *, json: dict, headers: dict) -> Iterator[str]:
        if self._mod == "httpx":
            with self._client.stream("POST", url, json=json, headers=headers) as r:
                r.raise_for_status()
                for line in r.iter_lines():
                    yield line
            return
        r = self._client.post(url, json=json, headers=headers, stream=True)
        r.raise_for_status()
        for line in r.iter_lines(decode_unicode=True):
            yield line


# ---------------------------------------------------------------------------
# schema loader
# ---------------------------------------------------------------------------

# OpenAI-compatible providers (use openai schema)
_OPENAI_COMPATIBLE = {Provider.groq, Provider.deepseek, Provider.mistral, Provider.openrouter, Provider.xai}


def _schema_name(provider: Provider) -> str:
    if provider in _OPENAI_COMPATIBLE:
        return "openai"
    if provider == Provider.ollama:
        return "native"
    return provider.value


def _load_schema(provider: Provider):
    """Load input/output schema classes for a provider."""
    name = _schema_name(provider)
    input_mod = importlib.import_module(f"schemas.{name}.input")
    output_mod = importlib.import_module(f"schemas.{name}.output")
    in_cls = next((getattr(input_mod, c) for c in dir(input_mod) if c.endswith("InputSchema")), None)
    out_cls = next((getattr(output_mod, c) for c in dir(output_mod) if c.endswith("OutputSchema")), None)
    if not in_cls or not out_cls:
        raise ImportError(f"Schema for provider '{provider.value}' not found")
    return in_cls(), out_cls()


# ---------------------------------------------------------------------------
# public client
# ---------------------------------------------------------------------------

class Switchboard:
    def __init__(
        self,
        provider: str | Provider,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
        http_client: Any | None = None,
    ) -> None:
        self.provider = Provider(provider)
        self.api_key = api_key
        self.model = model
        self.base_url = base_url or _ENDPOINTS.get(self.provider, "")
        self._http = http_client or _Http()
        
        # Load schemas for this provider
        self.input_schema, self.output_schema = _load_schema(self.provider)

    def _url(self, model: str, endpoint: str = "chat") -> str:
        return _chat_url(self.provider, self.base_url, model or self.model or "")

    def _url_with_key(self, model: str) -> str:
        url = self._url(model)
        if self.provider == Provider.gemini and self.api_key:
            url = f"{url}?key={self.api_key}"
        return url

    def _headers(self) -> dict[str, str]:
        h = _auth_headers(self.provider, self.api_key)
        if self.provider == Provider.claude:
            h["anthropic-version"] = "2023-06-01"
        return h

    def chat(self, messages: list[dict], *, options: dict | None = None, thinking: bool | None = None, reasoning_effort: str | None = None, stream: bool = False, tools: list[dict] | None = None, **kwargs) -> dict | Iterator[dict]:
        """Send a chat request."""
        kwargs.pop("skip_thinking_check", None)
        opts = dict(options or {})
        if reasoning_effort:
            opts["reasoning_effort"] = reasoning_effort
        elif thinking:
            opts["reasoning_effort"] = "high"

        req = {"model": self.model or "default", "messages": messages, "stream": stream}
        if opts:
            req["options"] = opts
        if tools:
            req["tools"] = tools
        if kwargs:
            req.update(kwargs)

        # Convert to provider format
        provider_req = self.input_schema.to_provider(req, self.provider.value)

        if stream:
            return self._stream(provider_req)
        return self._send(provider_req)

    def _send(self, provider_req: dict) -> dict:
        body = provider_req
        url = self._url_with_key(provider_req.get("model", self.model or ""))
        raw = self._http.post(url, json=body, headers=self._headers())
        return self.output_schema.from_provider(raw, self.provider.value)

    def _stream(self, provider_req: dict) -> Iterator[dict]:
        body = provider_req
        is_sse = self.provider != Provider.ollama
        url = self._url_with_key(provider_req.get("model", self.model or ""))
        
        for line in self._http.stream_post(url, json=body, headers=self._headers()):
            if not line:
                continue
            data = line
            if is_sse:
                if not line.startswith("data: "):
                    continue
                data = line[6:]
                if data == "[DONE]":
                    break
            try:
                raw = json.loads(data)
            except json.JSONDecodeError:
                continue
            yield self.output_schema.from_provider(raw, self.provider.value)

    def generate(self, prompt: str, **kwargs) -> dict:
        """Simple generate endpoint (non-chat)."""
        messages = [{"role": "user", "content": prompt}]
        return self.chat(messages, **kwargs)

    def list(self) -> list[dict]:
        """List available models."""
        url = self.base_url
        if self.provider == Provider.ollama:
            url = f"{url}/api/tags"
        elif self.provider == Provider.gemini:
            url = f"{url}?key={self.api_key}" if self.api_key else url
        else:
            url = f"{url}/models"
        
        result = self._http.get(url, headers=self._headers())
        
        if self.provider == Provider.ollama:
            return [{"id": m["name"]} for m in result.get("models", [])]
        elif self.provider == Provider.gemini:
            return result.get("models", [])
        return result.get("data", [])