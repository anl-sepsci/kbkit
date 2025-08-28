"""Load custom matplotlib style for consistent scientific visualization."""

from pathlib import Path

import matplotlib.pyplot as plt


def load_mplstyle() -> None:
    """
    Apply the custom matplotlib style defined in `presentation.mplstyle`.

    Returns
    -------
    matplotlib.style
        The applied style object (side effect: sets global matplotlib style).
    """
    style_path = Path(__file__).parent / "presentation.mplstyle"
    return plt.style.use(style_path)
