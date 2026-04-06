"""Simple functions for formatting plt axes."""

import matplotlib.pyplot as plt
import numpy as np

def style_axes(
    axhandle: plt.axes,
    axes_linewidth: float = 0.5,
    gridon: bool = False,
    gridstyle: str = ":",
    gridcolor: str = "k",
    gridwidth: float = 0.5,
    gridalpha: float = 0.8,
    majorticklength: float = 8,
    majortickwidth: float = 0.5,
    minorticklength: float = 5,
    minortickwidth: float = 0.5,
    minorxticks: list[float] | np.ndarray | None = None,
    minoryticks: list[float] | np.ndarray | None = None,
):
    """Style axes, grid, and ticks for a given axes object."""
    for spine in axhandle.spines.values():
        spine.set_linewidth(axes_linewidth)
    if gridon:
        axhandle.grid(gridon, ls=gridstyle, color=gridcolor, lw=gridwidth, alpha=gridalpha)
    else:
        axhandle.grid(gridon)
    axhandle.tick_params(axis="both", length=majorticklength, width=majortickwidth)
    axhandle.tick_params(axis="both", length=minorticklength, width=minortickwidth, which="minor")
    if minorxticks is not None:
        axhandle.set_xticks(minor=True, ticks=minorxticks)
    if minoryticks is not None:
        axhandle.set_yticks(minor=True, ticks=minoryticks)


def style_legend(
    axhandle: plt.axes,
    ncol: int = 1,
    fontsize: float = 14,
    labelspacing: float = 0.2,
    linelength: float = 1.5,
    linewidth: float = 1,
    framealpha: float = 1,
    framewidth: float = 0.5,
    framecolor: str = "k",
    frameon: bool = True,
    facecolor: str = "white",
    rounding_size: float = 0.0,
    framepad: float = 0.1,
    **kwargs,
) -> None:
    """Style legend on axes object."""
    legend = axhandle.legend(
        ncol=ncol,
        framealpha=framealpha,
        fontsize=fontsize,
        edgecolor=framecolor,
        facecolor=facecolor,
        frameon=frameon,
        fancybox=False,
        labelspacing=labelspacing,
        handlelength=linelength,
        **kwargs,
    )
    legend.get_frame().set_boxstyle(pad=framepad, rounding_size=rounding_size)
    legend.get_frame().set_linewidth(framewidth)
    for line in legend.get_lines():
        line.set_linewidth(linewidth)
