"""Centralized configuration and registries."""

from kbkit.config.mplstyle import load_mplstyle
from kbkit.config.unit_registry import load_unit_registry

__all__ = ["load_mplstyle", "load_unit_registry"]
