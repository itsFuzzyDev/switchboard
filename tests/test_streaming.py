import unittest
from schemas import OllamaStreamChunk, Provider


class TestStreaming(unittest.TestCase):
    def test_openai_stream_chunk(self):
        raw = {"choices": [{"delta": {"content": "hello"}, "finish_reason": None}]}
        c = OllamaStreamChunk.from_provider_chunk(raw, Provider.openai, model="gpt-4")
        self.assertEqual(c.message.content, "hello")
        self.assertFalse(c.done)

    def test_openai_stream_done(self):
        raw = {"choices": [{"delta": {}, "finish_reason": "stop"}]}
        c = OllamaStreamChunk.from_provider_chunk(raw, Provider.openai, model="gpt-4")
        self.assertTrue(c.done)

    def test_deepseek_stream_reasoning(self):
        raw = {"choices": [{"delta": {"reasoning_content": "hmm"}}]}
        c = OllamaStreamChunk.from_provider_chunk(raw, Provider.deepseek, model="ds")
        self.assertEqual(c.reasoning_content, "hmm")
        self.assertEqual(c.message.content, "")

    def test_claude_text_delta(self):
        raw = {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "hi"}}
        c = OllamaStreamChunk.from_provider_chunk(raw, Provider.claude, model="claude-3")
        self.assertEqual(c.message.content, "hi")
        self.assertFalse(c.done)

    def test_claude_thinking_delta(self):
        raw = {"type": "content_block_delta", "delta": {"type": "thinking_delta", "thinking": "think"}}
        c = OllamaStreamChunk.from_provider_chunk(raw, Provider.claude, model="claude-3")
        self.assertEqual(c.reasoning_content, "think")

    def test_gemini_stream_chunk(self):
        raw = {"candidates": [{"content": {"parts": [{"text": "world"}]}}]}
        c = OllamaStreamChunk.from_provider_chunk(raw, Provider.gemini, model="gemini-pro")
        self.assertEqual(c.message.content, "world")

    def test_ollama_stream_chunk(self):
        raw = {"model": "llama3", "message": {"role": "assistant", "content": "!"}, "done": False}
        c = OllamaStreamChunk.from_provider_chunk(raw, Provider.ollama, model="llama3")
        self.assertEqual(c.message.content, "!")
        self.assertFalse(c.done)

    def test_ollama_stream_done(self):
        raw = {"model": "llama3", "message": {"role": "assistant", "content": ""}, "done": True}
        c = OllamaStreamChunk.from_provider_chunk(raw, Provider.ollama, model="llama3")
        self.assertTrue(c.done)


if __name__ == "__main__":
    unittest.main()
