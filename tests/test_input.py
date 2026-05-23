import unittest
from schemas import OllamaRequest, Provider


class TestInput(unittest.TestCase):
    def test_openai_conversion(self):
        req = OllamaRequest(
            model="gpt-4",
            messages=[{"role": "user", "content": "hi"}],
            options={"temperature": 0.5, "max_tokens": 100},
        )
        out = req.to_provider(Provider.openai)
        self.assertEqual(out["model"], "gpt-4")
        self.assertEqual(out["messages"], [{"role": "user", "content": "hi"}])
        self.assertEqual(out["temperature"], 0.5)
        self.assertEqual(out["max_tokens"], 100)
        self.assertFalse(out["stream"])

    def test_claude_system_extraction(self):
        req = OllamaRequest(
            model="claude-3",
            messages=[
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "hi"},
            ],
        )
        out = req.to_provider(Provider.claude)
        self.assertEqual(out["system"], "sys")
        self.assertEqual(out["messages"], [{"role": "user", "content": "hi"}])
        self.assertEqual(out["max_tokens"], 4096)

    def test_gemini_contents(self):
        req = OllamaRequest(
            model="gemini-pro",
            messages=[
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "hi"},
            ],
        )
        out = req.to_provider(Provider.gemini)
        self.assertEqual(out["contents"], [{"role": "user", "parts": [{"text": "hi"}]}])
        self.assertEqual(out["systemInstruction"], {"parts": [{"text": "sys"}]})

    def test_tools_wrapped(self):
        req = OllamaRequest(
            model="m",
            messages=[{"role": "user", "content": "hi"}],
            tools=[{"name": "foo", "description": "d", "parameters": {"type": "object"}}],
        )
        openai_out = req.to_provider(Provider.openai)
        self.assertEqual(
            openai_out["tools"],
            [{"type": "function", "function": {"name": "foo", "description": "d", "parameters": {"type": "object"}}}],
        )
        claude_out = req.to_provider(Provider.claude)
        self.assertEqual(
            claude_out["tools"],
            [{"name": "foo", "description": "d", "input_schema": {"type": "object"}}],
        )

    def test_json_format(self):
        req = OllamaRequest(
            model="m",
            messages=[{"role": "user", "content": "hi"}],
            format="json",
        )
        out = req.to_provider(Provider.openai)
        self.assertEqual(out["response_format"], {"type": "json_object"})

    def test_all_providers_runnable(self):
        req = OllamaRequest(
            model="m",
            messages=[{"role": "user", "content": "hi"}],
        )
        for p in Provider:
            self.assertIsInstance(req.to_provider(p), dict)

    def test_deepseek_reasoning(self):
        req = OllamaRequest(
            model="ds",
            messages=[{"role": "user", "content": "hi"}],
            options={"reasoning_effort": "max"},
        )
        out = req.to_provider(Provider.deepseek)
        self.assertEqual(out["reasoning_effort"], "max")
        self.assertEqual(out["thinking"], {"type": "enabled"})


if __name__ == "__main__":
    unittest.main()
