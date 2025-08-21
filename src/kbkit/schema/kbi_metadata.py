


from dataclasses import dataclass, field 
from numpy.typing import NDArray
import numpy as np

@dataclass
class KBIMetadata:
    mols: tuple[str, str]
    r: NDArray[np.float64]
    g: NDArray[np.float64]
    rkbi: NDArray[np.float64]
    lam: NDArray[np.float64]
    lam_rkbi: NDArray[np.float64]
    lam_fit: NDArray[np.float64]
    lam_rkbi_fit: NDArray[np.float64]
    kbi: float