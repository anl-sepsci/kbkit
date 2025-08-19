

import matplotlib.pyplot as plt 
from pathlib import Path

def load_mplstyle() -> plt.style:
    style_path = Path(__file__).parent / "presentation.mplstyle"
    return plt.style.use(style_path)


