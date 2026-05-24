from enum import Enum


class Provider(str, Enum):
    openai = "openai"
    ollama = "ollama"
    gemini = "gemini"
    groq = "groq"
    claude = "claude"
    deepseek = "deepseek"
    mistral = "mistral"
    openrouter = "openrouter"
    xai = "xai"