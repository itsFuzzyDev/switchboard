# Switchboard

A lightweight open router that accepts **Ollama-style** chat requests and transparently maps them to any supported provider, then normalizes the response back into **Ollama-style** output.

Built for a downstream project that needs one unified interface across many LLM backends.

---

## Supported providers

| Provider | Input format | Output format | Streaming |
|----------|-------------|---------------|-----------|
| OpenAI | Chat Completions | Chat Completions | ✅ |
| Ollama | Native | Native | ✅ |
| Google Gemini | `generateContent` | `generateContent` | ✅ |
| Groq | Chat Completions | Chat Completions | ✅ |
| Anthropic Claude | Messages API | Messages API | ✅ |
| DeepSeek | Chat Completions | Chat Completions | ✅ |
| Mistral AI | Chat Completions | Chat Completions | ✅ |
| OpenRouter | Chat Completions | Chat Completions | ✅ |
| xAI (Grok) | Chat Completions | Chat Completions | ✅ |

More providers can be added by writing a single converter + parser pair.

---

## How it works

```
┌─────────────────┐     to_provider()      ┌──────────────┐
│  OllamaRequest  │ ────────────────────▶ │  Provider X  │
│  (your input)   │                        │  (native API)│
└─────────────────┘                        └──────────────┘
                                                   │
┌─────────────────┐     from_provider()          │
│  OllamaResponse │ ◀────────────────────────────┘
│  (your output)  │
└─────────────────┘
```

1. You construct an `OllamaRequest` (model, messages, options, tools, format, etc.)
2. Call `request.to_provider(Provider.openai)` (or any provider) to get the native request dict
3. Send that dict to the provider's HTTP API
4. Feed the raw JSON response to `OllamaResponse.from_provider(raw, Provider.openai, model="...")`
5. You get back a uniform `OllamaResponse` regardless of which provider you called

The same flow works for streaming: use `OllamaStreamChunk.from_provider_chunk(raw_chunk, provider, model)` on every SSE/NDJSON chunk.

---

## Quick start

```python
from schemas import OllamaRequest, Provider, OllamaResponse

# 1. Build the request in Ollama style
req = OllamaRequest(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": "You are a concise assistant."},
        {"role": "user", "content": "What is 2 + 2?"},
    ],
    options={"temperature": 0.2, "max_tokens": 100},
)

# 2. Convert to the provider's native schema
openai_body = req.to_provider(Provider.openai)
# openai_body now looks like:
# {
#   "model": "gpt-4o",
#   "messages": [...],
#   "stream": False,
#   "temperature": 0.2,
#   "max_tokens": 100
# }

# 3. Send it (example with httpx/requests/aiohttp)
# response = requests.post("https://api.openai.com/v1/chat/completions", json=openai_body, headers=...)

# 4. Parse the raw JSON back into Ollama style
raw = response.json()
ollama_resp = OllamaResponse.from_provider(raw, Provider.openai, model="gpt-4o")
print(ollama_resp.message.content)   # "4"
print(ollama_resp.eval_count)       # completion tokens
```

---

## Streaming

```python
from schemas import OllamaStreamChunk, Provider

# For each SSE line you receive:
chunk = OllamaStreamChunk.from_provider_chunk(raw_line, Provider.openai, model="gpt-4o")
print(chunk.message.content, end="", flush=True)
# chunk.done is True on the final chunk
```

---

## Tools / function calling

```python
req = OllamaRequest(
    model="gpt-4o",
    messages=[{"role": "user", "content": "What's the weather in Paris?"}],
    tools=[{
        "name": "get_weather",
        "description": "Get current weather",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}
    }],
)

# OpenAI gets: { "type": "function", "function": { ... } }
# Claude gets:  { "name": ..., "input_schema": ... }
```

---

## JSON mode / structured output

```python
# Simple JSON mode
req = OllamaRequest(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Return a JSON object with a greeting"}],
    format="json",
)
# OpenAI receives: response_format = { "type": "json_object" }

# JSON Schema
req = OllamaRequest(
    model="gpt-4o",
    messages=[...],
    format={"type": "object", "properties": {"greeting": {"type": "string"}}, "required": ["greeting"]},
)
# OpenAI receives: response_format = { "type": "json_schema", "json_schema": { ... } }
```

---

## Adding a new provider

Three steps:

1. **Input converter** in `schemas/input.py`:
   - Write a function `_myprovider(r: OllamaRequest) -> dict`
   - Register it in `_CONVERTERS`

2. **Output parser** in `schemas/output.py`:
   - Write `_parse_myprovider_response(raw, model) -> OllamaResponse`
   - Write `_parse_myprovider_stream_chunk(raw, model) -> OllamaStreamChunk`
   - Register both in `_PARSERS` and `_STREAM_PARSERS`

3. Add the enum value to `Provider` in `schemas/input.py`.

---

## Project layout

```
schemas/
  __init__.py      # public exports
  input.py         # OllamaRequest + provider converters
  output.py        # OllamaResponse + OllamaStreamChunk + parsers
tests/
  test_input.py    # request conversion tests
  test_output.py   # response parsing tests
  test_streaming.py# stream chunk parsing tests
```

---

## Tests

```bash
PYTHONPATH=. python3 -m unittest discover -s tests -v
```

---

*More providers may be added as needed. The architecture is intentionally flat so each provider is just one function.*
