from schemas.provider import Provider
from schemas.base import (
    InputSchema, OutputSchema, transform, get_path, set_path,
    SwitchboardError, ApiError, SchemaError, ConfigurationError,
)
from schemas.registry import register, load
from schemas.native.input import OllamaInputSchema
from schemas.native.output import OllamaOutputSchema
from schemas.openai.input import OpenAIInputSchema
from schemas.openai.output import OpenAIOutputSchema
from schemas.claude.input import ClaudeInputSchema
from schemas.claude.output import ClaudeOutputSchema
from schemas.gemini.input import GeminiInputSchema
from schemas.gemini.output import GeminiOutputSchema

register("native", OllamaInputSchema, OllamaOutputSchema)
register("openai", OpenAIInputSchema, OpenAIOutputSchema)
register("claude", ClaudeInputSchema, ClaudeOutputSchema)
register("gemini", GeminiInputSchema, GeminiOutputSchema)

__all__ = [
    "Provider",
    "InputSchema", "OutputSchema",
    "transform", "get_path", "set_path",
    "SwitchboardError", "ApiError", "SchemaError", "ConfigurationError",
    "register", "load",
]
