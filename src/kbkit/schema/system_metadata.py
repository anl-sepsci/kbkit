"""Structured representation of molecular simulation systems."""

from dataclasses import dataclass, field
from pathlib import Path

from kbkit.core.system_properties import SystemProperties


@dataclass
class SystemMetadata:
    """
    Semantic container for a molecular simulation system.

    Attributes
    ----------
    name : str
        System name, typically derived from directory or input file.
    kind : str
        System type, either "pure" or "mixture".
    path : Path
        Filesystem path to the system directory.
    props : SystemProperties
        Parsed properties including topology, thermodynamics, and metadata.
    rdf_path : Path, optional
        Path to RDF directory if available (used for structural analysis).
    tags : list[str], optional
        Optional semantic tags for filtering, grouping, or annotation.

    Methods
    -------
    has_rdf() -> bool
        Return True if an RDF path is defined and non-empty.

    Notes
    -----
    - Used by SystemRegistry, SystemConfig, and SystemAnalyzer to organize and filter systems.
    - Supports reproducible analysis by encapsulating both structure and metadata.
    """

    name: str
    kind: str
    path: Path
    props: SystemProperties
    rdf_path: Path = field(default_factory=Path)
    tags: list[str] = field(default_factory=list)

    def has_rdf(self) -> bool:
        """Return True if an RDF path is defined and non-empty."""
        return any(self.rdf_path.glob("*.xvg"))
