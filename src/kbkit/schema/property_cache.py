"""
Structured representation of scalar properties with units and semantic tags.

Defines the PropertyCache dataclass, which encapsulates a value, its associated units,
and optional tags for downstream filtering, annotation, or metadata mapping.
Used in system registries, analysis pipelines, and config resolution.
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class PropertyCache:
    """
    Container for a scalar property with units and semantic annotations.

    Designed to store a value alongside its physical units and optional tags
    for classification, filtering, or metadata enrichment.

    Attributes
    ----------
    value : Any
        The raw property value (e.g., float, int, or derived object).
    units : str
        Units associated with the value (e.g., "kJ/mol", "nm", "mol/L").
    tags : list[str]
        Optional semantic labels for filtering or categorization
        (e.g., ["thermo", "mixture", "derived"]).

    Notes
    -----
    - Tags can be used to group properties by domain or analysis context.
    - Units are stored as strings for lightweight serialization and display.
    - Value type is intentionally flexible to support extensible use cases.
    """

    value: Any
    units: str = field(default_factory=str)
    tags: list[str] = field(default_factory=list)
