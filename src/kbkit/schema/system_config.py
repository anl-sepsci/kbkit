"""Domain-level configuration object for system discovery and registry integration."""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

@dataclass
class SystemConfig:
    """
    Configuration container for system-level metadata and registry context.

    Encapsulates the environment required to discover, register, and analyze molecular systems
    across base and pure directories. Serves as a semantic anchor for ensemble-specific workflows.

    Notes
    -----
    - Designed to support reproducible system discovery and filtering.
    - Registry object should be preconfigured with semantic rules and discovery logic.
    - Logging is centralized to support contributor diagnostics and debugging.
    """

    base_path: Path
    pure_path: Path
    ensemble: str
    cations: list[str]
    anions: list[str]
    logger: logging.Logger
    molecules: list[str]
    registry: Optional[object] = None
