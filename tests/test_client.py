import unittest
from switchboard import Switchboard


class MockHttp:
    """Fake HTTP client for unit tests."""
    def __init__(self, responses: list[dict] | None = None) -> None:
        self.posts: list[tuple] = []
        self.gets: list[tuple] = []
        self._responses = responses or []
        self._stream_lines: list[str] = []

    def post(self, url, *, json, headers):
        self.posts.append((url, json, headers))
        return self._responses.pop(0) if self._responses else {"choices": [{"message": {"content": "ok"}}], "usage": {}}

    def get(self, url, *, headers):
        self.gets.append((url, headers))
        return self._responses.pop(0) if self._responses else {"data": [{"id": "gpt-4"}]}

    def stream_post(self, url, *, json, headers):
        self.posts.append((url, json, headers))
        for line in self._stream_lines:
            yield line


class TestSwitchboardChat(unittest.TestCase):
    def test_chat_openai(self):
        http = MockHttp(responses=[{"choices": [{"message": {"content": "hi"}}], "usage": {"prompt_tokens": 2, "completion_tokens": 1}, "model": "gpt-4"}])
        sb = Switchboard(provider="openai", api_key="sk-test", model="gpt-4", http_client=http)
        resp = sb.chat(messages=[{"role": "user", "content": "hello"}])
        self.assertEqual(resp["message"]["content"], "hi")
        self.assertEqual(http.posts[0][0], "https://api.openai.com/v1/chat/completions")
        self.assertEqual(http.posts[0][2]["authorization"], "Bearer sk-test")

    def test_chat_claude(self):
        http = MockHttp(responses=[{"content": [{"type": "text", "text": "yo"}], "usage": {"input_tokens": 3, "output_tokens": 1}, "model": "claude-3"}])
        sb = Switchboard(provider="claude", api_key="sk-test", model="claude-3", http_client=http)
        resp = sb.chat(messages=[{"role": "user", "content": "hi"}])
        self.assertEqual(resp["message"]["content"], "yo")
        self.assertEqual(http.posts[0][0], "https://api.anthropic.com/v1/messages")

    def test_chat_gemini_with_key_query_param(self):
        http = MockHttp(responses=[{"candidates": [{"content": {"parts": [{"text": "ok"}]}}], "usageMetadata": {}}])
        sb = Switchboard(provider="gemini", api_key="AIza-test", model="gemini-pro", http_client=http)
        resp = sb.chat(messages=[{"role": "user", "content": "hi"}])
        self.assertTrue("?key=AIza-test" in http.posts[0][0])

    def test_stream_openai(self):
        http = MockHttp()
        http._stream_lines = [
            'data: {"choices": [{"delta": {"content": "he"}}]}',
            'data: {"choices": [{"delta": {"content": "llo"}}]}',
            'data: [DONE]',
        ]
        sb = Switchboard(provider="openai", api_key="sk-test", model="gpt-4", http_client=http)
        chunks = list(sb.chat(messages=[{"role": "user", "content": "hi"}], stream=True))
        self.assertEqual(len(chunks), 2)
        self.assertEqual(chunks[0]["message"]["content"], "he")
        self.assertEqual(chunks[1]["message"]["content"], "llo")

    def test_generate_routes_through_chat(self):
        http = MockHttp(responses=[{"choices": [{"message": {"content": "42"}}], "usage": {}}])
        sb = Switchboard(provider="openai", api_key="sk-test", http_client=http)
        resp = sb.generate(model="gpt-4", prompt="2+2")
        self.assertEqual(resp["message"]["content"], "42")
        # verify it sent a chat completions request with a single user message
        self.assertEqual(http.posts[0][1]["messages"], [{"role": "user", "content": "2+2"}])


class TestSwitchboardList(unittest.TestCase):
    def test_list_openai(self):
        http = MockHttp()
        http._responses = [{"data": [{"id": "gpt-4"}, {"id": "gpt-3.5-turbo"}]}]
        sb = Switchboard(provider="openai", api_key="sk-test", http_client=http)
        models = sb.list()
        self.assertEqual([m["id"] for m in models], ["gpt-4", "gpt-3.5-turbo"])

    def test_list_ollama(self):
        http = MockHttp()
        http._responses = [{"models": [{"name": "llama3"}, {"name": "mistral"}]}]
        sb = Switchboard(provider="ollama", http_client=http)
        models = sb.list()
        self.assertEqual([m["id"] for m in models], ["llama3", "mistral"])


class TestThinkingCapability(unittest.TestCase):
    def test_thinking_passes_through_for_any_model(self):
        """We no longer block thinking by model — we just send it and let the provider decide."""
        http = MockHttp(responses=[{"choices": [{"message": {"content": "ok"}}], "usage": {}}])
        sb = Switchboard(provider="openai", api_key="sk-test", model="gpt-4", http_client=http)
        resp = sb.chat(messages=[{"role": "user", "content": "hi"}], thinking=True)
        self.assertEqual(resp["message"]["content"], "ok")
        # reasoning_effort should still be passed through in options
        self.assertEqual(http.posts[0][1].get("reasoning_effort"), "high")

    def test_thinking_allowed_for_reasoning_model(self):
        http = MockHttp(responses=[{"choices": [{"message": {"content": "ok"}}], "usage": {}}])
        sb = Switchboard(provider="deepseek", api_key="sk-test", model="deepseek-v4-pro", http_client=http)
        resp = sb.chat(messages=[{"role": "user", "content": "hi"}], thinking=True)
        self.assertEqual(resp["message"]["content"], "ok")
        # verify reasoning_effort was passed through in options
        self.assertEqual(http.posts[0][1].get("reasoning_effort"), "high")

    def test_skip_thinking_check_bypasses_validation(self):
        http = MockHttp(responses=[{"choices": [{"message": {"content": "ok"}}], "usage": {}}])
        sb = Switchboard(provider="openai", api_key="sk-test", model="gpt-4", http_client=http)
        resp = sb.chat(messages=[{"role": "user", "content": "hi"}], thinking=True, skip_thinking_check=True)
        self.assertEqual(resp["message"]["content"], "ok")

    def test_reasoning_effort_passed_to_request(self):
        http = MockHttp(responses=[{"content": [{"type": "text", "text": "ok"}], "usage": {}, "model": "claude-sonnet-4"}])
        sb = Switchboard(provider="claude", api_key="sk-test", model="claude-sonnet-4", http_client=http)
        resp = sb.chat(messages=[{"role": "user", "content": "hi"}], reasoning_effort="low")
        self.assertEqual(resp["message"]["content"], "ok")
        # Claude gets thinking budget set based on effort level
        self.assertIn("thinking", http.posts[0][1])


if __name__ == "__main__":
    unittest.main()
