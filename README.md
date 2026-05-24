# Switchboard

A schema-agnostic LLM router. Define your input/output format once, call any provider.

Built for projects that need one unified interface across many LLM backends.

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

**Global schema:** Ollama (by default) — you speak Ollama in, get Ollama out, regardless of which provider you call.

Each **provider folder** has:
- `input.py`: converts **native schema** → **that provider's native format**
- `output.py`: converts **that provider's response** → **native schema**

The `native/` folder is your home schema. Change it if you want a different default.

---

## Install

```bash
pip install -e .
```

---

## Quick start

```python
from switchboard import Switchboard

sb = Switchboard(provider="openai", api_key="sk-...")

# Your schema is Ollama by default
resp = sb.chat(
    model="gpt-4",
    messages=[{"role": "user", "content": "Hi"}]
)

# Response is Ollama-shaped
print(resp["message"]["content"])
```

Or use `.generate()` for a single prompt string:

```python
resp = sb.generate(model="gpt-4", prompt="Write a haiku about routers")
print(resp["message"]["content"])
```

---

## Streaming

```python
for chunk in sb.chat(messages=[{"role": "user", "content": "Hi"}], stream=True):
    print(chunk["message"]["content"], end="", flush=True)
```

---

## Model list

```python
from switchboard import Switchboard

sb = Switchboard(provider="openai", api_key="sk-...")

print(sb.list()) # gets you list of most/all models, requires api key on all except OpenRouter and Ollaam
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

**To add a new provider:** create a folder with `input.py` + `output.py`. Auto-discovered.

---

## Adding a new provider

1. Create `schemas/newprovider/input.py` with `InputSchema` class
2. Create `schemas/newprovider/output.py` with `OutputSchema` class
3. Done

Each class has:
- `to_provider(data, provider)`: convert your schema → provider format
- `from_provider(raw, provider)`: convert provider response → your schema

---

## Tests

```bash
PYTHONPATH=. python3 -m unittest discover -s tests -v
```
