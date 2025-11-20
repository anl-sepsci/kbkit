"""Structured representation of Kirkwood-Buff integrals and related RDF data."""

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray


@dataclass
class KBIMetadata:
    """
    Container for Kirkwood-Buff integral (KBI) analysis results for a molecular pair.

    Notes
    -----
    - All arrays are assumed to be aligned over the same radial grid `r`.
    """

    mols: tuple[str, ...]
    r: NDArray[np.float64]
    g: NDArray[np.float64]
    rkbi: NDArray[np.float64]
    scaled_rkbi: NDArray[np.float64]
    r_fit: NDArray[np.float64]
    scaled_rkbi_fit: NDArray[np.floating[Any]]
    scaled_rkbi_est: NDArray[np.floating[Any]]
    kbi_limit: float
