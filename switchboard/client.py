from __future__ import annotations
import json, importlib
from typing import Any, Iterator, AsyncIterator

from schemas import Provider, InputSchema, OutputSchema


class SwitchboardError(Exception):
    """Base exception for all Switchboard errors."""


class ApiError(SwitchboardError):
    """Raised when the provider API returns an error response."""
    def __init__(self, message: str, status_code: int | None = None, response_body: dict | str | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.response_body = response_body


class SchemaError(SwitchboardError):
    """Raised when schema transformation fails."""


class ConfigurationError(SwitchboardError):
    """Raised when the client is misconfigured."""


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

_OPENAI_COMPATIBLE = {Provider.groq, Provider.deepseek, Provider.mistral, Provider.openrouter, Provider.xai}


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
        if provider == Provider.claude:
            h["x-api-key"] = api_key
        else:
            h["authorization"] = f"Bearer {api_key}"
    return h


def _schema_name(provider: Provider) -> str:
    if provider in _OPENAI_COMPATIBLE:
        return "openai"
    if provider == Provider.ollama:
        return "native"
    return provider.value


def _load_schema(provider: Provider) -> tuple[InputSchema, OutputSchema]:
    name = _schema_name(provider)
    try:
        input_mod = importlib.import_module(f"schemas.{name}.input")
        output_mod = importlib.import_module(f"schemas.{name}.output")
    except ImportError as exc:
        raise SchemaError(f"Schema module for provider '{provider.value}' not found") from exc

    def _find(mod: Any, suffix: str) -> type | None:
        for attr in dir(mod):
            if attr.endswith(suffix) and not attr.startswith("_"):
                obj = getattr(mod, attr)
                if isinstance(obj, type) and obj.__module__ == mod.__name__:
                    return obj
        return None

    in_cls = _find(input_mod, "InputSchema")
    out_cls = _find(output_mod, "OutputSchema")

    if not in_cls or not out_cls:
        raise SchemaError(f"Schema classes for provider '{provider.value}' not found in schemas.{name}")
    return in_cls(), out_cls()


class _Http:
    def __init__(self) -> None:
        self._client: Any = None
        self._mod: str = ""
        try:
            import httpx
            self._client = httpx.Client(timeout=120)
            self._mod = "httpx"
        except ImportError as _httpx_err:
            try:
                import requests
                self._client = requests.Session()
                self._mod = "requests"
            except ImportError as _requests_err:
                raise ConfigurationError("Switchboard needs httpx or requests installed") from _requests_err

    def _call(self, method: str, url: str, *, json: dict | None = None, headers: dict) -> dict:
        try:
            if self._mod == "requests":
                r = self._client.request(method, url, json=json, headers=headers, timeout=120)
            else:
                r = self._client.request(method, url, json=json, headers=headers)
            r.raise_for_status()
            return r.json()
        except Exception as exc:
            status = getattr(exc, "response", None) and getattr(exc.response, "status_code", None)
            body = None
            if hasattr(exc, "response") and exc.response is not None:
                try:
                    body = exc.response.json()
                except Exception:
                    body = exc.response.text
            raise ApiError(str(exc), status_code=status, response_body=body) from exc

    def post(self, url: str, *, json: dict, headers: dict) -> dict:
        return self._call("POST", url, json=json, headers=headers)

    def get(self, url: str, *, headers: dict) -> dict:
        return self._call("GET", url, headers=headers)

    def stream_post(self, url: str, *, json: dict, headers: dict) -> Iterator[str]:
        try:
            if self._mod == "httpx":
                with self._client.stream("POST", url, json=json, headers=headers) as r:
                    r.raise_for_status()
                    for line in r.iter_lines():
                        yield line
                return
            r = self._client.post(url, json=json, headers=headers, stream=True, timeout=120)
            r.raise_for_status()
            for line in r.iter_lines(decode_unicode=True):
                yield line
        except Exception as exc:
            status = getattr(exc, "response", None) and getattr(exc.response, "status_code", None)
            body = None
            if hasattr(exc, "response") and exc.response is not None:
                try:
                    body = exc.response.json()
                except Exception:
                    body = exc.response.text
            raise ApiError(str(exc), status_code=status, response_body=body) from exc


class _AsyncHttp:
    def __init__(self) -> None:
        try:
            import httpx
            self._client = httpx.AsyncClient(timeout=120)
        except ImportError as _httpx_err:
            raise ConfigurationError("AsyncSwitchboard requires httpx") from _httpx_err

    async def _call(self, method: str, url: str, *, json: dict | None = None, headers: dict) -> dict:
        try:
            r = await self._client.request(method, url, json=json, headers=headers)
            r.raise_for_status()
            return r.json()
        except Exception as exc:
            status = getattr(exc, "response", None) and getattr(exc.response, "status_code", None)
            body = None
            if hasattr(exc, "response") and exc.response is not None:
                try:
                    body = exc.response.json()
                except Exception:
                    body = exc.response.text
            raise ApiError(str(exc), status_code=status, response_body=body) from exc

    async def post(self, url: str, *, json: dict, headers: dict) -> dict:
        return await self._call("POST", url, json=json, headers=headers)

    async def get(self, url: str, *, headers: dict) -> dict:
        return await self._call("GET", url, headers=headers)

    async def stream_post(self, url: str, *, json: dict, headers: dict) -> AsyncIterator[str]:
        try:
            async with self._client.stream("POST", url, json=json, headers=headers) as r:
                r.raise_for_status()
                async for line in r.aiter_lines():
                    yield line
        except Exception as exc:
            status = getattr(exc, "response", None) and getattr(exc.response, "status_code", None)
            body = None
            if hasattr(exc, "response") and exc.response is not None:
                try:
                    body = exc.response.json()
                except Exception:
                    body = exc.response.text
            raise ApiError(str(exc), status_code=status, response_body=body) from exc


class _BaseSwitchboard:
    def __init__(
        self,
        provider: str | Provider,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
    ) -> None:
        try:
            self.provider = Provider(provider)
        except ValueError as exc:
            raise ConfigurationError(f"Unknown provider: {provider}") from exc
        self.api_key = api_key
        self.model = model
        self.base_url = base_url or _ENDPOINTS.get(self.provider, "")
        if not self.base_url:
            raise ConfigurationError(f"No endpoint configured for provider '{self.provider.value}'")
        self.input_schema, self.output_schema = _load_schema(self.provider)

    def _url(self, model: str) -> str:
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

    def _build_request(
        self,
        messages: list[dict],
        *,
        options: dict | None = None,
        thinking: bool | None = None,
        reasoning_effort: str | None = None,
        stream: bool = False,
        tools: list[dict] | None = None,
        **kwargs,
    ) -> dict:
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
        return self.input_schema.to_provider(req, self.provider.value)

    def _parse_stream_line(self, line: str, is_sse: bool) -> dict | None:
        if not line:
            return None
        if is_sse:
            if not line.startswith("data: "):
                return None
            line = line[6:]
            if line == "[DONE]":
                return None
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            return None


class Switchboard(_BaseSwitchboard):
    def __init__(
        self,
        provider: str | Provider,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
        http_client: Any | None = None,
    ) -> None:
        super().__init__(provider, api_key, model, base_url)
        self._http = http_client or _Http()

    def chat(
        self,
        messages: list[dict],
        *,
        options: dict | None = None,
        thinking: bool | None = None,
        reasoning_effort: str | None = None,
        stream: bool = False,
        tools: list[dict] | None = None,
        **kwargs,
    ) -> dict | Iterator[dict]:
        provider_req = self._build_request(
            messages,
            options=options,
            thinking=thinking,
            reasoning_effort=reasoning_effort,
            stream=stream,
            tools=tools,
            **kwargs,
        )
        return self._stream(provider_req) if stream else self._send(provider_req)

    def _send(self, provider_req: dict) -> dict:
        url = self._url_with_key(provider_req.get("model", self.model or ""))
        try:
            raw = self._http.post(url, json=provider_req, headers=self._headers())
        except Exception as exc:
            if isinstance(exc, ApiError):
                raise
            raise ApiError(str(exc)) from exc
        try:
            return self.output_schema.from_provider(raw, self.provider.value)
        except Exception as exc:
            raise SchemaError(f"Failed to transform provider response: {exc}") from exc

    def _stream(self, provider_req: dict) -> Iterator[dict]:
        is_sse = self.provider != Provider.ollama
        url = self._url_with_key(provider_req.get("model", self.model or ""))
        try:
            for line in self._http.stream_post(url, json=provider_req, headers=self._headers()):
                raw = self._parse_stream_line(line, is_sse)
                if raw is None:
                    continue
                try:
                    yield self.output_schema.from_provider(raw, self.provider.value)
                except Exception as exc:
                    raise SchemaError(f"Failed to transform stream chunk: {exc}") from exc
        except Exception as exc:
            if isinstance(exc, (ApiError, SchemaError)):
                raise
            raise ApiError(str(exc)) from exc

    def generate(self, prompt: str, **kwargs) -> dict:
        return self.chat([{"role": "user", "content": prompt}], **kwargs)

    def list(self) -> list[dict]:
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
        if self.provider == Provider.gemini:
            return [{"id": m.get("name", "")} for m in result.get("models", [])]
        return result.get("data", [])


class AsyncSwitchboard(_BaseSwitchboard):
    def __init__(
        self,
        provider: str | Provider,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
        http_client: Any | None = None,
    ) -> None:
        super().__init__(provider, api_key, model, base_url)
        self._http = http_client or _AsyncHttp()

    async def chat(
        self,
        messages: list[dict],
        *,
        options: dict | None = None,
        thinking: bool | None = None,
        reasoning_effort: str | None = None,
        stream: bool = False,
        tools: list[dict] | None = None,
        **kwargs,
    ) -> dict | AsyncIterator[dict]:
        provider_req = self._build_request(
            messages,
            options=options,
            thinking=thinking,
            reasoning_effort=reasoning_effort,
            stream=stream,
            tools=tools,
            **kwargs,
        )
        if stream:
            return self._stream(provider_req)
        return await self._send(provider_req)

    async def _send(self, provider_req: dict) -> dict:
        url = self._url_with_key(provider_req.get("model", self.model or ""))
        try:
            raw = await self._http.post(url, json=provider_req, headers=self._headers())
        except Exception as exc:
            if isinstance(exc, ApiError):
                raise
            raise ApiError(str(exc)) from exc
        try:
            return self.output_schema.from_provider(raw, self.provider.value)
        except Exception as exc:
            raise SchemaError(f"Failed to transform provider response: {exc}") from exc

    async def _stream(self, provider_req: dict) -> AsyncIterator[dict]:
        is_sse = self.provider != Provider.ollama
        url = self._url_with_key(provider_req.get("model", self.model or ""))
        try:
            async for line in self._http.stream_post(url, json=provider_req, headers=self._headers()):
                raw = self._parse_stream_line(line, is_sse)
                if raw is None:
                    continue
                try:
                    yield self.output_schema.from_provider(raw, self.provider.value)
                except Exception as exc:
                    raise SchemaError(f"Failed to transform stream chunk: {exc}") from exc
        except Exception as exc:
            if isinstance(exc, (ApiError, SchemaError)):
                raise
            raise ApiError(str(exc)) from exc

    async def generate(self, prompt: str, **kwargs) -> dict:
        return await self.chat([{"role": "user", "content": prompt}], **kwargs)

    async def list(self) -> list[dict]:
        url = self.base_url
        if self.provider == Provider.ollama:
            url = f"{url}/api/tags"
        elif self.provider == Provider.gemini:
            url = f"{url}?key={self.api_key}" if self.api_key else url
        else:
            url = f"{url}/models"

        result = await self._http.get(url, headers=self._headers())

        if self.provider == Provider.ollama:
            return [{"id": m["name"]} for m in result.get("models", [])]
        if self.provider == Provider.gemini:
            return [{"id": m.get("name", "")} for m in result.get("models", [])]
        return result.get("data", [])
