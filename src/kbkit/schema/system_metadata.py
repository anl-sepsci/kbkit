"""Structured representation of molecular simulation systems."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from kbkit.systems.properties import SystemProperties


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

    Notes
    -----
    - Used by :class:`~kbkit.systems.collection.SystemCollection` to organize and filter systems.
    - Supports reproducible analysis by encapsulating both structure and metadata.
    """

    name: str
    kind: str
    path: Path
    props: "SystemProperties"  # not Optional
    rdf_path: Path = field(default_factory=Path)

    def has_rdf(self) -> bool:
        """Return True if an RDF path is defined and contains RDF data files."""
        # Use str(self.rdf_path) == "." to detect the default empty Path()
        if not self.rdf_path or str(self.rdf_path) == ".":
            return False

        if not self.rdf_path.is_dir():
            return False

        return any(any(self.rdf_path.glob(pattern)) for pattern in ("*.xvg", "*.txt", "*.csv"))

    def is_pure(self) -> bool:
        """Return True if ``kind`` is a pure molecule."""
        return self.kind.lower() == "pure"
