import unittest
from schemas import OllamaResponse, Provider


class TestOutput(unittest.TestCase):
    def test_openai_response(self):
        raw = {
            "model": "gpt-4",
            "choices": [{"message": {"role": "assistant", "content": "hello"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        }
        r = OllamaResponse.from_provider(raw, Provider.openai, model="gpt-4")
        self.assertEqual(r.model, "gpt-4")
        self.assertEqual(r.message.content, "hello")
        self.assertEqual(r.prompt_eval_count, 10)
        self.assertEqual(r.eval_count, 5)

    def test_claude_response(self):
        raw = {
            "model": "claude-3",
            "content": [
                {"type": "text", "text": "hello"},
                {"type": "thinking", "thinking": "hmm"},
            ],
            "role": "assistant",
            "usage": {"input_tokens": 8, "output_tokens": 3},
        }
        r = OllamaResponse.from_provider(raw, Provider.claude, model="claude-3")
        self.assertEqual(r.message.content, "hello")
        self.assertEqual(r.reasoning_content, "hmm")
        self.assertEqual(r.prompt_eval_count, 8)
        self.assertEqual(r.eval_count, 3)

    def test_gemini_response(self):
        raw = {
            "candidates": [{"content": {"parts": [{"text": "hi"}], "role": "model"}}],
            "usageMetadata": {"promptTokenCount": 4, "candidatesTokenCount": 2},
        }
        r = OllamaResponse.from_provider(raw, Provider.gemini, model="gemini-pro")
        self.assertEqual(r.message.content, "hi")
        self.assertEqual(r.prompt_eval_count, 4)
        self.assertEqual(r.eval_count, 2)

    def test_ollama_response_passthrough(self):
        raw = {
            "model": "llama3",
            "message": {"role": "assistant", "content": "yo"},
            "done": True,
            "prompt_eval_count": 3,
            "eval_count": 1,
        }
        r = OllamaResponse.from_provider(raw, Provider.ollama, model="llama3")
        self.assertEqual(r.message.content, "yo")
        self.assertEqual(r.prompt_eval_count, 3)

    def test_deepseek_reasoning_response(self):
        raw = {
            "model": "ds",
            "choices": [
                {"message": {"role": "assistant", "content": "answer", "reasoning_content": "think"}}
            ],
            "usage": {"prompt_tokens": 5, "completion_tokens": 10},
        }
        r = OllamaResponse.from_provider(raw, Provider.deepseek, model="ds")
        self.assertEqual(r.message.content, "answer")
        self.assertEqual(r.reasoning_content, "think")


if __name__ == "__main__":
    unittest.main()
