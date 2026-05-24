from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any


def get_path(data: dict, path: str) -> Any:
    """Navigate nested dict: 'a.b.c' or 'items[0].name'."""
    if not data:
        return None
    parts, current = [], ""
    for c in path:
        if c == ".":
            if current:
                parts.append(current)
                current = ""
        elif c == "[":
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

    val = data
    for part in parts:
        if val is None:
            return None
        if isinstance(val, list):
            if not part.isdigit():
                return None
            val = val[int(part)] if int(part) < len(val) else None
        elif isinstance(val, dict):
            val = val.get(part)
        else:
            return None
    return val


def set_path(data: dict, path: str, value: Any) -> None:
    """Set nested value. Creates dicts as needed."""
    parts, current = [], ""
    for c in path:
        if c == ".":
            if current:
                parts.append(current)
                current = ""
        elif c == "[":
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

    node = data
    for part in parts[:-1]:
        if part.isdigit():
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
        idx = int(last)
        while len(node) <= idx:
            node.append(None)
        node[idx] = value
    else:
        node[last] = value


def transform(raw: dict, mapping: dict[str, str]) -> dict:
    """Apply mapping: {source_path: target_path}. Unmapped goes to extra."""
    out, consumed = {}, set()
    for src, tgt in mapping.items():
        val = get_path(raw, src)
        if val is not None:
            set_path(out, tgt, val)
            consumed.add(src)

    if consumed:
        out["extra"] = {k: v for k, v in raw.items() if k not in consumed}
    elif raw:
        out["extra"] = raw
    return out


class Schema(ABC):
    @abstractmethod
    def to_provider(self, data: dict, provider: str) -> dict: ...
    @abstractmethod
    def from_provider(self, raw: dict, provider: str) -> dict: ...