"""Domain models and orchestration logic."""

import sys

from kbkit.core.system_loader import SystemLoader
from kbkit.core.system_properties import SystemProperties

__all__ = ["KBPipeline", "SystemLoader", "SystemProperties"]


def _safe_import():
    try:
        from kbkit.core.kb_pipeline import KBPipeline  # noqa: PLC0415

        sys.modules[__name__].KBPipeline = KBPipeline
    except ImportError:
        pass  # avoid circular import crash


_safe_import()
