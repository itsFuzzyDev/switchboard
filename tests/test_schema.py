import unittest
from schemas import Provider, transform, get_path, set_path
from schemas.openai.input import OpenAIInputSchema
from schemas.openai.output import OpenAIOutputSchema
from schemas.claude.input import ClaudeInputSchema
from schemas.claude.output import ClaudeOutputSchema
from schemas.gemini.input import GeminiInputSchema
from schemas.gemini.output import GeminiOutputSchema
from schemas.native.input import OllamaInputSchema
from schemas.native.output import OllamaOutputSchema


class TestPathUtils(unittest.TestCase):
    def test_get_path_simple(self):
        self.assertEqual(get_path({"a": 1}, "a"), 1)
        self.assertEqual(get_path({"a": {"b": 2}}, "a.b"), 2)

    def test_get_path_index(self):
        self.assertEqual(get_path({"items": [0, 1, 2]}, "items[1]"), 1)

    def test_set_path(self):
        d = {}
        set_path(d, "a.b.c", 1)
        self.assertEqual(d, {"a": {"b": {"c": 1}}})


class TestTransform(unittest.TestCase):
    def test_simple_mapping(self):
        mapping = {"a": "x", "b": "y"}
        result = transform({"a": 1, "b": 2, "c": 3}, mapping)
        self.assertEqual(result, {"x": 1, "y": 2, "extra": {"c": 3}})


class TestOllamaPassthrough(unittest.TestCase):
    def test_input_passthrough(self):
        schema = OllamaInputSchema()
        data = {"model": "llama3", "messages": [{"role": "user", "content": "hi"}]}
        result = schema.to_provider(data, Provider.ollama)
        self.assertEqual(result, data)

    def test_output_passthrough(self):
        schema = OllamaOutputSchema()
        raw = {"model": "llama3", "message": {"role": "assistant", "content": "hi"}, "done": True}
        result = schema.from_provider(raw, Provider.ollama)
        self.assertEqual(result["message"]["content"], "hi")


class TestOpenAIConversion(unittest.TestCase):
    def test_input(self):
        schema = OpenAIInputSchema()
        data = {"model": "gpt-4", "messages": [{"role": "user", "content": "hi"}], "options": {"temperature": 0.5}}
        result = schema.to_provider(data, Provider.openai)
        self.assertEqual(result["model"], "gpt-4")
        self.assertEqual(result["temperature"], 0.5)
        self.assertEqual(result["messages"], data["messages"])

    def test_output(self):
        schema = OpenAIOutputSchema()
        raw = {"model": "gpt-4", "choices": [{"message": {"content": "hi"}}], "usage": {"prompt_tokens": 5, "completion_tokens": 2}}
        result = schema.from_provider(raw, Provider.openai)
        self.assertEqual(result["message"]["content"], "hi")
        self.assertEqual(result["prompt_eval_count"], 5)

    def test_stream_output(self):
        schema = OpenAIOutputSchema()
        raw = {"model": "gpt-4", "choices": [{"delta": {"content": "hi"}}]}
        result = schema.from_provider(raw, Provider.openai)
        self.assertEqual(result["message"]["content"], "hi")


class TestClaudeConversion(unittest.TestCase):
    def test_input_with_thinking(self):
        schema = ClaudeInputSchema()
        data = {"model": "claude-3", "messages": [{"role": "user", "content": "hi"}], "options": {"reasoning_effort": "high"}}
        result = schema.to_provider(data, Provider.claude)
        self.assertIn("thinking", result)
        self.assertEqual(result["thinking"]["budget_tokens"], 16000)

    def test_output(self):
        schema = ClaudeOutputSchema()
        raw = {"model": "claude-3", "content": [{"type": "text", "text": "hi"}], "usage": {"input_tokens": 5, "output_tokens": 2}}
        result = schema.from_provider(raw, Provider.claude)
        self.assertEqual(result["message"]["content"], "hi")
        self.assertEqual(result["prompt_eval_count"], 5)


class TestGeminiConversion(unittest.TestCase):
    def test_input(self):
        schema = GeminiInputSchema()
        data = {"model": "gemini-pro", "messages": [{"role": "user", "content": "hi"}]}
        result = schema.to_provider(data, Provider.gemini)
        self.assertEqual(result["contents"][0]["parts"][0]["text"], "hi")

    def test_output(self):
        schema = GeminiOutputSchema()
        raw = {"candidates": [{"content": {"parts": [{"text": "hi"}]}}], "usageMetadata": {"promptTokenCount": 5, "candidatesTokenCount": 2}}
        result = schema.from_provider(raw, Provider.gemini)
        self.assertEqual(result["message"]["content"], "hi")
        self.assertEqual(result["prompt_eval_count"], 5)


class TestProviderEnum(unittest.TestCase):
    def test_all_providers(self):
        self.assertEqual(len([p for p in Provider]), 9)


if __name__ == "__main__":
    unittest.main(verbosity=2)