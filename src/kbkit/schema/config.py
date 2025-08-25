"""Structured representation of system environment."""

from dataclasses import dataclass
import logging
from pathlib import Path 
from kbkit.core.registry import SystemRegistry

@dataclass
class SystemConfig:
    base_path: Path
    pure_path: Path 
    ensemble: str
    cations: list[str]
    anions: list[str]
    registry: SystemRegistry
    logger: logging.Logger
    molecules: list[str]