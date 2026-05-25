from typing import Type
from schemas.base import InputSchema, OutputSchema, ConfigurationError

_REGISTRY: dict[str, tuple[Type[InputSchema], Type[OutputSchema]]] = {}


def register(name: str, input_cls: Type[InputSchema], output_cls: Type[OutputSchema]) -> None:
    _REGISTRY[name] = (input_cls, output_cls)


def load(name: str) -> tuple[InputSchema, OutputSchema]:
    try:
        in_cls, out_cls = _REGISTRY[name]
    except KeyError as exc:
        raise ConfigurationError(f"No schema registered for provider '{name}'") from exc
    return in_cls(), out_cls()
