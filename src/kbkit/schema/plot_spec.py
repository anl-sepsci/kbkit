"""Structure that contains data for plotting thermodynamic properties."""

from dataclasses import dataclass
from typing import Optional

import numpy as np
from numpy.typing import NDArray


@dataclass
class PlotSpec:
    """
    Specification for a single plot to be rendered by the Plotter.

    Attributes
    ----------
    x_data : NDArray[np.float64]
        The x-axis data for the plot.
    ylabel : str
        Label for the y-axis.
    filename : str
        Output filename for the saved plot.
    multi : bool, optional
        Whether the plot includes multiple y-series (e.g., stacked thermodynamic contributions).
    y_data : Optional[NDArray[np.float64]], optional
        Single y-axis data for simple plots.
    y_series : Optional[list[tuple[NDArray[np.float64], str, str, str]]], optional
        List of (y_data, color, marker, label) tuples for multi-series plots.
    fit_fns : Optional[dict[str, Callable[..., Any]]], optional
        Optional dictionary of molecule-specific fit functions for overlaying curves.
    """

    x_data: NDArray[np.float64]
    ylabel: str
    filename: str
    multi: bool = False
    y_data: Optional[NDArray[np.float64]] = None
    y_series: Optional[list[tuple[NDArray[np.float64], str, str, str]]] = None
    fit_fns: Optional[dict[str, np.poly1d]] = None
