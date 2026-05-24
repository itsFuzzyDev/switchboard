# Switchboard

A schema-agnostic LLM router. Define your input/output format once, call any provider.

---

## Supported providers

| Provider | URL |
|----------|-----|
| OpenAI | `api.openai.com/v1` |
| Ollama | `localhost:11434` |
| Google Gemini | `generativelanguage.googleapis.com` |
| Groq | `api.groq.com/openai/v1` |
| Anthropic Claude | `api.anthropic.com/v1` |
| DeepSeek | `api.deepseek.com/v1` |
| Mistral AI | `api.mistral.ai/v1` |
| OpenRouter | `openrouter.ai/api/v1` |
| xAI (Grok) | `api.x.ai/v1` |

---

## How it works

```
Your data (native schema)  ──▶  schemas/openai/input.py  ──▶  OpenAI format
OpenAI response            ──▶  schemas/openai/output.py ──▶  Your data (native schema)
```

**Global schema:** Ollama by default. You speak Ollama-shaped in, get Ollama-shaped out, regardless of provider.

The `native/` folder holds your global schema. Change it if you need a different default shape.

> **Note:** The Ollama provider also maps to `native/`, so changing `native/` changes the Ollama provider's format too.

Each provider folder (`schemas/<provider>/`) has:
- `input.py`: native → provider format
- `output.py`: provider response → native format

---

## Install

```bash
pip install -e .
```

---

## Quick start

### `chat()` — full message list

```python
from switchboard import Switchboard

sb = Switchboard(provider="openai", api_key="sk-...")

resp = sb.chat(
    model="gpt-4",
    messages=[{"role": "user", "content": "Hi"}]
)

print(resp["message"]["content"])
```

### `generate()` — single prompt string

```python
resp = sb.generate(model="gpt-4", prompt="Write a haiku about routers")
print(resp["message"]["content"])
```

### Streaming

```python
for chunk in sb.chat(messages=[{"role": "user", "content": "Hi"}], stream=True):
    print(chunk["message"]["content"], end="", flush=True)
```

### Model list

```python
models = sb.list()
```

---

## Architecture

```
schemas/
  base.py           # transform(), get_path(), set_path()
  provider.py       # Provider enum
  native/           # your global schema (default: Ollama)
    input.py       # passthrough
    output.py      # passthrough
  openai/
    input.py       # native → OpenAI
    output.py      # OpenAI → native
  claude/
    input.py       # native → Claude
    output.py      # Claude → native
  gemini/
    input.py       # native → Gemini
    output.py      # Gemini → native
```

---

## Adding a new provider

### OpenAI-compatible (Groq, DeepSeek, Mistral, OpenRouter, xAI, etc.)

Add the endpoint to `_ENDPOINTS` in `switchboard/client.py` and include the provider in `_OPENAI_COMPATIBLE`. Reuses `schemas/openai/` automatically.

### Custom provider

1. Create `schemas/<name>/input.py` with a class ending in `InputSchema`
2. Create `schemas/<name>/output.py` with a class ending in `OutputSchema`
3. Auto-discovered on import

Each class needs:
- `to_provider(data, provider)` — native → provider format
- `from_provider(raw, provider)` — provider response → native

---

## Tests

```bash
PYTHONPATH=. python3 -m unittest discover -s tests -v
```
