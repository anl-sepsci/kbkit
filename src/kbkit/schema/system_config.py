"""Domain-level configuration object for system discovery and registry integration."""

import logging
from dataclasses import dataclass
from pathlib import Path

from kbkit.core.system_registry import SystemRegistry


@dataclass
class SystemConfig:
    """
    Configuration container for system-level metadata and registry context.

    Encapsulates the environment required to discover, register, and analyze molecular systems
    across base and pure directories. Serves as a semantic anchor for ensemble-specific workflows.

    Attributes
    ----------
    base_path : Path
        Path to the directory containing mixed or ensemble systems.
    pure_path : Path
        Path to the directory containing pure component systems.
    ensemble : str
        Name or identifier for the ensemble (e.g., "NaCl_water").
    cations : list[str]
        List of cation species included in the ensemble.
    anions : list[str]
        List of anion species included in the ensemble.
    registry : SystemRegistry
        Registry object used to discover and organize system metadata.
    logger : logging.Logger
        Logger instance for structured diagnostics and workflow tracing.
    molecules : list[str]
        Full list of molecular species present in the ensemble.

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
    registry: SystemRegistry
    logger: logging.Logger
    molecules: list[str]
