"""Plotting support for time series energy properties."""

import copy
import warnings
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np

from kbkit.config.mplstyle import load_mplstyle
from kbkit.utils.format import ENERGY_ALIASES, format_unit_str, resolve_attr_key

warnings.filterwarnings("ignore")

if TYPE_CHECKING:
    from kbkit.systems.collection import SystemCollection
    from kbkit.systems.properties import SystemProperties

load_mplstyle()


class TimeseriesPlotter:
    """Plotting timeseries of energy properties for a given simulations.

    Parameters
    ----------
    props: SystemProperties
        SystemProperties object for a given molecular dynamics system.
    start_time: int
        Initial time for plotting.
    """

    def __init__(self, props: "SystemProperties", start_time: int = 0) -> None:
        # resets start time for plotting, but dont alter original
        self.props = copy.copy(props)
        self.props.start_time = start_time

    @classmethod
    def from_collection(cls, systems: "SystemCollection", system_name: str | int, start_time: int = 0) -> "TimeseriesPlotter":
        """Initialized `TimeseriesPlotter` from a :class:`~kbkit.systems.collection.SystemCollection` object.

        Parameters
        ----------
        collection: SystemCollection
            SystemCollection object for a given set of systems.
        system: str | int
            Name or index of system in SystemCollection.
        start_time: int
            Initial time for plotting.

        Returns
        -------
        TimeseriesPlotter
            Initialized TimeseriesPlotter object.
        """
        return cls(systems[system_name].props, start_time)

    def plot(
        self,
        name: str,
        units: str | None = None,
        show_avg: bool = True,
        figsize: tuple = (9, 4),
        xlabel: str = "Time (ns)",
        ylabel: str | None = None,
        title: str | None = None,
        ylim: tuple | None = None,
        xlim: tuple | None = None,
        color: str = "skyblue",
        alpha: float = 0.8,
        ls: str = "-",
        lw: float = 1,
        marker: str = "none",
        savepath: str | Path | None = None,
        show: bool = True,
    ):
        """
        Create a timeseries plot for a given energy property.

        Optionally, visualize the running average of the property and report average on figure legend.

        Parameters
        ----------
        name: str
            Name of property to plot.
        units: str, optional
            Units of desired property. If not provided, property will be displayed in default units.
        show_avg: bool, optional
            Add the running average and the averaged property to the figure.
        figsize: tuple, optional
            Size of the figure to display (height, width).
        xlabel: str, optional
            Label for x-axis.
        ylabel: str, optional
            Label for y-axis.
        title: str, optional
            Title label.
        ylim: tuple, optional
            Limits for y-axis.
        xlim: tuple, optional
            Limits for x-axis.
        color: str, optional
            Color to display timeseries.
        alpha: float, optional
            Transparency for timeseries.
        ls: str, optional
            Linestyle for timeseries.
        lw: float, optional
            Linewidth for timeseries.
        marker: str, optional
            Maker to display for timeseries.
        savepath: str | Path, optional
            Path to save figure.
        show: bool, optional
            Display the figure.
        """
        name = resolve_attr_key(name, ENERGY_ALIASES)
        units = units or self.props.energy[0].units[name]

        time, values = self.props.get(name=name, units=units, avg=False, time_series=True)

        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(time / 1000, values, c=color, alpha=alpha, ls=ls, lw=lw, marker=marker)

        if show_avg and len(time) > 0 and len(values) > 0:
            with np.errstate(divide="ignore", invalid="ignore"):
                run_avg = [np.mean(values[:i]) for i in range(len(values))]
                last = run_avg[-1]
                label = f"{last:.3f} ({units})" if last < 1 else f"{last:.0f} ({units})"
                ax.plot(time / 1000, run_avg, c="k", ls="-", lw=lw, label=label)
                ax.legend()

        if xlabel:
            ax.set_xlabel(xlabel)
        if ylabel:
            ax.set_ylabel(ylabel)
        else:
            ax.set_ylabel(f"{name.capitalize()} ({format_unit_str(units)})")
        if title:
            ax.set_title(title)
        if xlim:
            ax.set_xlim(xlim)
        if ylim:
            ax.set_ylim(ylim)

        if savepath:
            savepath = Path(savepath) if Path(savepath).is_file() else Path(savepath) / "energy.pdf"
            fig.savefig(savepath, dpi=100)

        if show:
            plt.show()
        else:
            plt.close()

        return fig, ax
