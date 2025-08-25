"""
Load custom matplotlib style for consistent scientific visualization.

Provides a utility to apply a predefined `.mplstyle` file for presentation-ready plots.
Used across analysis and reporting modules to ensure visual consistency.
"""

from pathlib import Path

import matplotlib.pyplot as plt


def load_mplstyle() -> plt.style:
    """
    Apply the custom matplotlib style defined in `presentation.mplstyle`.

    Returns
    -------
    matplotlib.style
        The applied style object (side effect: sets global matplotlib style).

    Notes
    -----
    - Style file is expected to reside in the same directory as this module.
    - Used to standardize plot aesthetics across figures and notebooks.
    """
    style_path = Path(__file__).parent / "presentation.mplstyle"
    return plt.style.use(style_path)
