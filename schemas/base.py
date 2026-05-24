from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any


def _parse_path(path: str) -> list[str]:
    parts, current = [], ""
    for c in path:
        if c in (".", "["):
            if current:
                parts.append(current)
                current = ""
        elif c == "]":
            parts.append(current)
            current = ""
        else:
            current += c
    if current:
        parts.append(current)
    return parts


def get_path(data: dict, path: str) -> Any:
    if not data:
        return None
    val = data
    for part in _parse_path(path):
        if val is None:
            return None
        if isinstance(val, list):
            if not part.isdigit():
                return None
            idx = int(part)
            val = val[idx] if idx < len(val) else None
        elif isinstance(val, dict):
            val = val.get(part)
        else:
            return None
    return val


def set_path(data: dict, path: str, value: Any) -> None:
    parts = _parse_path(path)
    node = data
    for part in parts[:-1]:
        if part.isdigit():
            if not isinstance(node, list):
                raise TypeError(f"Cannot index into {type(node).__name__} with path part {part!r}")
            idx = int(part)
            while len(node) <= idx:
                node.append({})
            node = node[idx]
        else:
            if part not in node:
                node[part] = {}
            node = node[part]

    last = parts[-1]
    if last.isdigit():
        if not isinstance(node, list):
            raise TypeError(f"Cannot index into {type(node).__name__} with path part {last!r}")
        idx = int(last)
        while len(node) <= idx:
            node.append(None)
        node[idx] = value
    else:
        node[last] = value


def transform(raw: dict, mapping: dict[str, str]) -> dict:
    """Apply mapping: {source_path: target_path}. Unmapped top-level keys go to extra."""
    out = {}
    consumed = set()
    for src, tgt in mapping.items():
        val = get_path(raw, src)
        if val is not None:
            set_path(out, tgt, val)
            # Only consume exact top-level keys
            if "." not in src and "[" not in src:
                consumed.add(src)
    extra = {k: v for k, v in raw.items() if k not in consumed}
    if extra:
        out["extra"] = extra
    return out


class InputSchema(ABC):
    @abstractmethod
    def to_provider(self, data: dict, provider: str) -> dict: ...


class OutputSchema(ABC):
    @abstractmethod
    def from_provider(self, raw: dict, provider: str) -> dict: ...
