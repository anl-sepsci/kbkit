"""Structured representation of systems."""

from dataclasses import dataclass, field
from pathlib import Path 
from kbkit.core.properties import SystemProperties

@dataclass
class SystemMetadata:
    name: str
    kind: str
    path: Path
    props: SystemProperties
    rdf_path: Path = field(default_factory=str)
    tags: list[str] = field(default_factory=list)

    def has_rdf(self) -> bool:
        return bool(self.rdf_path)
