"""
Structured representation of Kirkwood-Buff integrals and related RDF data.

Defines the KBIMetadata dataclass, which encapsulates radial distribution functions,
KBI curves, lambda integrals, and fitted values for a given molecular pair.
Used in structural analysis, mixture characterization, and thermodynamic inference.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray


@dataclass
class KBIMetadata:
    """
    Container for Kirkwood-Buff integral (KBI) analysis results for a molecular pair.

    Attributes
    ----------
    mols : tuple[str, str]
        Pair of molecule names (e.g., ("Na+", "Cl-")).
    r : NDArray[np.float64]
        Radial distance array (in nanometers).
    g : NDArray[np.float64]
        Radial distribution function (RDF) values.
    rkbi : NDArray[np.float64]
        Cumulative KBI integral over r.
    lam : NDArray[np.float64]
        Lambda integral values over r.
    lam_rkbi : NDArray[np.float64]
        Lambda-weighted KBI values.
    lam_fit : NDArray[np.float64]
        Fitted lambda integral curve.
    lam_rkbi_fit : NDArray[np.float64]
        Fitted lambda-weighted KBI curve.
    kbi : float
        Final Kirkwood-Buff integral value for the molecular pair.

    Notes
    -----
    - All arrays are assumed to be aligned over the same radial grid `r`.
    - Used in mixture analysis and thermodynamic property estimation.
    - Can be serialized or visualized for structural diagnostics.
    """

    mols: tuple[str, ...]
    r: NDArray[np.float64]
    g: NDArray[np.float64]
    rkbi: NDArray[np.float64]
    lam: NDArray[np.float64]
    lam_rkbi: NDArray[np.float64]
    lam_fit: NDArray[np.float64]
    lam_rkbi_fit: NDArray[np.floating[Any]]
    kbi: float
