"""Structured representation of property and units."""

from dataclasses import dataclass, field
from typing import Any

@dataclass
class PropertyCache:
    value: Any
    units: str = field(default_factory=str)
    tags: list[str] = field(default_factory=list)