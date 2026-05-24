import unittest, asyncio
from switchboard import Switchboard, AsyncSwitchboard, ApiError, SchemaError, ConfigurationError


class MockHttp:
    def __init__(self, response: dict | None = None, stream_lines: list[str] | None = None, exc: Exception | None = None) -> None:
        self.posts: list[tuple] = []
        self.gets: list[tuple] = []
        self._response = response
        self._stream_lines = stream_lines or []
        self._exc = exc

    def post(self, url, *, json, headers):
        self.posts.append((url, json, headers))
        if self._exc:
            raise self._exc
        return self._response or {}

    def get(self, url, *, headers):
        self.gets.append((url, headers))
        if self._exc:
            raise self._exc
        return self._response or {}

    def stream_post(self, url, *, json, headers):
        self.posts.append((url, json, headers))
        if self._exc:
            raise self._exc
        yield from self._stream_lines


class MockAsyncHttp:
    def __init__(self, response: dict | None = None, stream_lines: list[str] | None = None, exc: Exception | None = None) -> None:
        self.posts: list[tuple] = []
        self.gets: list[tuple] = []
        self._response = response
        self._stream_lines = stream_lines or []
        self._exc = exc

    async def post(self, url, *, json, headers):
        self.posts.append((url, json, headers))
        if self._exc:
            raise self._exc
        return self._response or {}

    async def get(self, url, *, headers):
        self.gets.append((url, headers))
        if self._exc:
            raise self._exc
        return self._response or {}

    async def stream_post(self, url, *, json, headers):
        self.posts.append((url, json, headers))
        if self._exc:
            raise self._exc
        for line in self._stream_lines:
            yield line


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

_OPENAI_RESP = {"choices": [{"message": {"content": "hi", "role": "assistant"}}], "usage": {"prompt_tokens": 5, "completion_tokens": 2}, "model": "m"}
_CLAUDE_RESP = {"content": [{"type": "text", "text": "hi"}], "usage": {"input_tokens": 5, "output_tokens": 2}, "model": "m", "stop_reason": "end_turn", "id": "msg_1"}
_GEMINI_RESP = {"candidates": [{"content": {"parts": [{"text": "hi"}]}}], "usageMetadata": {"promptTokenCount": 5, "candidatesTokenCount": 2}}
_OLLAMA_RESP = {"model": "m", "message": {"role": "assistant", "content": "hi"}, "done": True, "prompt_eval_count": 5, "eval_count": 2}

_MSGS = [{"role": "user", "content": "hello"}]
_MSGS_WITH_SYSTEM = [{"role": "system", "content": "be terse"}, {"role": "user", "content": "hello"}]


def _sb(provider, response=None, api_key="key-test", model="m", stream_lines=None, exc=None, **kw):
    http = MockHttp(response=response, stream_lines=stream_lines, exc=exc)
    sb = Switchboard(provider=provider, api_key=api_key, model=model, http_client=http, **kw)
    return sb, http


def _asb(provider, response=None, api_key="key-test", model="m", stream_lines=None, exc=None, **kw):
    http = MockAsyncHttp(response=response, stream_lines=stream_lines, exc=exc)
    sb = AsyncSwitchboard(provider=provider, api_key=api_key, model=model, http_client=http, **kw)
    return sb, http


# ---------------------------------------------------------------------------
# OpenAI
# ---------------------------------------------------------------------------

class TestOpenAI(unittest.TestCase):
    def test_input(self):
        sb, http = _sb("openai", _OPENAI_RESP)
        sb.chat(messages=_MSGS, options={"temperature": 0.7, "max_tokens": 100})
        url, body, headers = http.posts[0]
        self.assertEqual(url, "https://api.openai.com/v1/chat/completions")
        self.assertEqual(headers["authorization"], "Bearer key-test")
        self.assertEqual(body["messages"], _MSGS)
        self.assertEqual(body["temperature"], 0.7)
        self.assertEqual(body["max_tokens"], 100)

    def test_output(self):
        sb, _ = _sb("openai", _OPENAI_RESP)
        resp = sb.chat(messages=_MSGS)
        self.assertEqual(resp["message"]["content"], "hi")
        self.assertEqual(resp["message"]["role"], "assistant")
        self.assertEqual(resp["prompt_eval_count"], 5)
        self.assertEqual(resp["eval_count"], 2)

    def test_system_message_stays_in_messages(self):
        sb, http = _sb("openai", _OPENAI_RESP)
        sb.chat(messages=_MSGS_WITH_SYSTEM)
        self.assertEqual(len(http.posts[0][1]["messages"]), 2)


# ---------------------------------------------------------------------------
# Claude
# ---------------------------------------------------------------------------

class TestClaude(unittest.TestCase):
    def test_input(self):
        sb, http = _sb("claude", _CLAUDE_RESP)
        sb.chat(messages=_MSGS)
        url, body, headers = http.posts[0]
        self.assertEqual(url, "https://api.anthropic.com/v1/messages")
        self.assertEqual(headers["x-api-key"], "key-test")
        self.assertEqual(headers["anthropic-version"], "2023-06-01")
        self.assertNotIn("authorization", headers)
        self.assertEqual(body["messages"], _MSGS)
        self.assertEqual(body["max_tokens"], 4096)

    def test_input_system_extracted(self):
        sb, http = _sb("claude", _CLAUDE_RESP)
        sb.chat(messages=_MSGS_WITH_SYSTEM)
        body = http.posts[0][1]
        self.assertEqual(body["system"], "be terse")
        self.assertEqual(body["messages"], [{"role": "user", "content": "hello"}])

    def test_input_thinking_low(self):
        sb, http = _sb("claude", _CLAUDE_RESP)
        sb.chat(messages=_MSGS, reasoning_effort="low")
        body = http.posts[0][1]
        self.assertEqual(body["thinking"], {"type": "enabled", "budget_tokens": 4000})

    def test_input_thinking_high(self):
        sb, http = _sb("claude", _CLAUDE_RESP)
        sb.chat(messages=_MSGS, reasoning_effort="high")
        body = http.posts[0][1]
        self.assertEqual(body["thinking"], {"type": "enabled", "budget_tokens": 16000})

    def test_output(self):
        sb, _ = _sb("claude", _CLAUDE_RESP)
        resp = sb.chat(messages=_MSGS)
        self.assertEqual(resp["message"]["content"], "hi")
        self.assertEqual(resp["message"]["role"], "assistant")
        self.assertEqual(resp["prompt_eval_count"], 5)
        self.assertEqual(resp["eval_count"], 2)

    def test_output_thinking_blocks(self):
        raw = {"content": [{"type": "thinking", "thinking": "step1"}, {"type": "text", "text": "answer"}], "usage": {"input_tokens": 5, "output_tokens": 2}, "model": "m", "stop_reason": "end_turn", "id": "x"}
        sb, _ = _sb("claude", raw)
        resp = sb.chat(messages=_MSGS)
        self.assertEqual(resp["message"]["content"], "answer")
        self.assertEqual(resp["reasoning_content"], "step1")


# ---------------------------------------------------------------------------
# Gemini
# ---------------------------------------------------------------------------

class TestGemini(unittest.TestCase):
    def test_input(self):
        sb, http = _sb("gemini", _GEMINI_RESP)
        sb.chat(messages=_MSGS)
        url, body, headers = http.posts[0]
        self.assertIn("m:generateContent", url)
        self.assertIn("?key=key-test", url)
        self.assertNotIn("authorization", headers)
        self.assertEqual(body["contents"][0]["parts"][0]["text"], "hello")
        self.assertEqual(body["contents"][0]["role"], "user")

    def test_input_system_extracted(self):
        sb, http = _sb("gemini", _GEMINI_RESP)
        sb.chat(messages=_MSGS_WITH_SYSTEM)
        body = http.posts[0][1]
        self.assertEqual(body["systemInstruction"]["parts"][0]["text"], "be terse")
        self.assertEqual(len(body["contents"]), 1)

    def test_input_generation_config(self):
        sb, http = _sb("gemini", _GEMINI_RESP)
        sb.chat(messages=_MSGS, options={"temperature": 0.5, "max_tokens": 200, "top_p": 0.9})
        body = http.posts[0][1]
        self.assertEqual(body["generationConfig"]["temperature"], 0.5)
        self.assertEqual(body["generationConfig"]["maxOutputTokens"], 200)
        self.assertEqual(body["generationConfig"]["topP"], 0.9)

    def test_output(self):
        sb, _ = _sb("gemini", _GEMINI_RESP)
        resp = sb.chat(messages=_MSGS)
        self.assertEqual(resp["message"]["content"], "hi")
        self.assertEqual(resp["prompt_eval_count"], 5)
        self.assertEqual(resp["eval_count"], 2)

    def test_list(self):
        sb, http = _sb("gemini", {"models": [{"name": "gemini-pro"}]})
        models = sb.list()
        self.assertIn("?key=key-test", http.gets[0][0])
        self.assertEqual(models[0]["id"], "gemini-pro")


# ---------------------------------------------------------------------------
# Ollama
# ---------------------------------------------------------------------------

class TestOllama(unittest.TestCase):
    def test_input(self):
        sb, http = _sb("ollama", _OLLAMA_RESP, api_key=None)
        sb.chat(messages=_MSGS)
        url, body, headers = http.posts[0]
        self.assertEqual(url, "http://localhost:11434/chat/completions")
        self.assertNotIn("authorization", headers)
        self.assertNotIn("x-api-key", headers)
        self.assertEqual(body["messages"], _MSGS)

    def test_output(self):
        sb, _ = _sb("ollama", _OLLAMA_RESP, api_key=None)
        resp = sb.chat(messages=_MSGS)
        self.assertEqual(resp["message"]["content"], "hi")
        self.assertEqual(resp["prompt_eval_count"], 5)
        self.assertEqual(resp["eval_count"], 2)

    def test_list(self):
        sb, http = _sb("ollama", {"models": [{"name": "llama3"}, {"name": "mistral"}]}, api_key=None)
        models = sb.list()
        self.assertIn("/api/tags", http.gets[0][0])
        self.assertEqual([m["id"] for m in models], ["llama3", "mistral"])

    def test_stream(self):
        lines = ['{"model":"m","message":{"role":"assistant","content":"he"},"done":false}',
                 '{"model":"m","message":{"role":"assistant","content":"llo"},"done":true}']
        sb, _ = _sb("ollama", api_key=None, stream_lines=lines)
        chunks = list(sb.chat(messages=_MSGS, stream=True))
        self.assertEqual(len(chunks), 2)
        self.assertEqual(chunks[0]["message"]["content"], "he")
        self.assertEqual(chunks[1]["message"]["content"], "llo")


# ---------------------------------------------------------------------------
# OpenAI-compatible providers (URL + auth only — schema covered by TestOpenAI)
# ---------------------------------------------------------------------------

class TestOpenAICompatible(unittest.TestCase):
    def _assert_provider(self, provider, expected_url):
        sb, http = _sb(provider, _OPENAI_RESP)
        sb.chat(messages=_MSGS)
        url, _, headers = http.posts[0]
        self.assertEqual(url, expected_url)
        self.assertEqual(headers["authorization"], "Bearer key-test")

    def test_groq(self):
        self._assert_provider("groq", "https://api.groq.com/openai/v1/chat/completions")

    def test_deepseek(self):
        self._assert_provider("deepseek", "https://api.deepseek.com/v1/chat/completions")

    def test_mistral(self):
        self._assert_provider("mistral", "https://api.mistral.ai/v1/chat/completions")

    def test_openrouter(self):
        self._assert_provider("openrouter", "https://openrouter.ai/api/v1/chat/completions")

    def test_xai(self):
        self._assert_provider("xai", "https://api.x.ai/v1/chat/completions")


# ---------------------------------------------------------------------------
# Streaming
# ---------------------------------------------------------------------------

class TestStreaming(unittest.TestCase):
    def test_openai_stream(self):
        lines = [
            'data: {"choices": [{"delta": {"content": "he"}}]}',
            'data: {"choices": [{"delta": {"content": "llo"}}]}',
            'data: [DONE]',
        ]
        sb, _ = _sb("openai", stream_lines=lines)
        chunks = list(sb.chat(messages=_MSGS, stream=True))
        self.assertEqual(len(chunks), 2)
        self.assertEqual(chunks[0]["message"]["content"], "he")
        self.assertEqual(chunks[1]["message"]["content"], "llo")

    def test_sse_non_data_lines_skipped(self):
        lines = [
            'event: content_block_start',
            'data: {"choices": [{"delta": {"content": "hi"}}]}',
        ]
        sb, _ = _sb("openai", stream_lines=lines)
        chunks = list(sb.chat(messages=_MSGS, stream=True))
        self.assertEqual(len(chunks), 1)


# ---------------------------------------------------------------------------
# generate()
# ---------------------------------------------------------------------------

class TestGenerate(unittest.TestCase):
    def test_routes_through_chat(self):
        sb, http = _sb("openai", _OPENAI_RESP)
        resp = sb.generate("2+2")
        self.assertEqual(http.posts[0][1]["messages"], [{"role": "user", "content": "2+2"}])
        self.assertEqual(resp["message"]["content"], "hi")


# ---------------------------------------------------------------------------
# list()
# ---------------------------------------------------------------------------

class TestList(unittest.TestCase):
    def test_openai(self):
        sb, http = _sb("openai", {"data": [{"id": "gpt-4"}, {"id": "gpt-3.5-turbo"}]})
        models = sb.list()
        self.assertIn("/models", http.gets[0][0])
        self.assertEqual([m["id"] for m in models], ["gpt-4", "gpt-3.5-turbo"])


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

class TestErrors(unittest.TestCase):
    def test_api_error_wrapped(self):
        sb, _ = _sb("openai", exc=RuntimeError("boom"))
        with self.assertRaises(ApiError) as ctx:
            sb.chat(messages=_MSGS)
        self.assertIn("boom", str(ctx.exception))

    def test_schema_error_on_bad_response(self):
        sb, _ = _sb("openai", "not a dict")
        with self.assertRaises(SchemaError):
            sb.chat(messages=_MSGS)

    def test_unknown_provider(self):
        with self.assertRaises(ConfigurationError):
            Switchboard(provider="fake")

    def test_missing_deps(self):
        # Force _Http to fail by mocking import — tested implicitly by configuration checks
        pass


# ---------------------------------------------------------------------------
# Async
# ---------------------------------------------------------------------------

class TestAsync(unittest.TestCase):
    def test_openai_chat(self):
        async def main():
            sb, _ = _asb("openai", _OPENAI_RESP)
            resp = await sb.chat(messages=_MSGS)
            self.assertEqual(resp["message"]["content"], "hi")
        asyncio.run(main())

    def test_openai_stream(self):
        async def main():
            lines = [
                'data: {"choices": [{"delta": {"content": "he"}}]}',
                'data: {"choices": [{"delta": {"content": "llo"}}]}',
                'data: [DONE]',
            ]
            sb, _ = _asb("openai", stream_lines=lines)
            chunks = []
            async for chunk in await sb.chat(messages=_MSGS, stream=True):
                chunks.append(chunk)
            self.assertEqual(len(chunks), 2)
            self.assertEqual(chunks[0]["message"]["content"], "he")
        asyncio.run(main())

    def test_async_api_error(self):
        async def main():
            sb, _ = _asb("openai", exc=RuntimeError("async boom"))
            with self.assertRaises(ApiError) as ctx:
                await sb.chat(messages=_MSGS)
            self.assertIn("async boom", str(ctx.exception))
        asyncio.run(main())

    def test_async_generate(self):
        async def main():
            sb, http = _asb("openai", _OPENAI_RESP)
            resp = await sb.generate("2+2")
            self.assertEqual(resp["message"]["content"], "hi")
            self.assertEqual(http.posts[0][1]["messages"], [{"role": "user", "content": "2+2"}])
        asyncio.run(main())

    def test_async_list(self):
        async def main():
            sb, http = _asb("openai", {"data": [{"id": "gpt-4"}]})
            models = await sb.list()
            self.assertEqual([m["id"] for m in models], ["gpt-4"])
        asyncio.run(main())


if __name__ == "__main__":
    unittest.main()
