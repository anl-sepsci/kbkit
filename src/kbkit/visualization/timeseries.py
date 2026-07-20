"""Plotting support for time series energy properties."""

import copy
import warnings
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np

from kbkit.config.mplstyle import load_mplstyle
from kbkit.utils.format import ENERGY_ALIASES, format_unit_str, resolve_attr_key
from kbkit.visualization.format import style_axes, style_legend

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
    start: int
        Initial time for plotting.
    """

    def __init__(self, props: "SystemProperties", start: int = 0) -> None:
        # resets start time for plotting, but dont alter original
        self.props = copy.copy(props)
        self.props.start = start

    @classmethod
    def from_collection(
        cls, systems: "SystemCollection", system_name: str | int, start: int = 0
    ) -> "TimeseriesPlotter":
        """Initialized `TimeseriesPlotter` from a :class:`~kbkit.systems.collection.SystemCollection` object.

        Parameters
        ----------
        collection: SystemCollection
            SystemCollection object for a given set of systems.
        system: str | int
            Name or index of system in SystemCollection.
        start: int
            Initial time for plotting.

        Returns
        -------
        TimeseriesPlotter
            Initialized TimeseriesPlotter object.
        """
        return cls(systems[system_name].props, start)

    def plot(
        self,
        name: str,
        units: str | None = None,
        show_avg: bool = True,
        figsize: tuple = (9, 4),
        xlabel: str | None = None,
        ylabel: str | None = None,
        title: str | None = None,
        ylim: tuple | None = None,
        xlim: tuple | None = None,
        savepath: str | Path | None = None,
        show: bool = True,
        **kwargs,
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
        savepath: str | Path, optional
            Path to save figure.
        show: bool, optional
            Display the figure.
        """
        name = resolve_attr_key(name, ENERGY_ALIASES)
        units = units or self.props.energy[0].units[name]

        x_arr, values = self.props.get(name=name, units=units, avg=False, time_series=True)
        if self.props.energy[0]._x_key == "time":
            x_arr /= 1000  # convert from ps -> ns

        if xlabel is None:
            xlabel = "Time (ns)" if self.props.energy[0]._x_key == "time" else "Timestep"

        fig, ax = plt.subplots(figsize=figsize)
        style_axes(ax)
        ax.plot(x_arr, values, **kwargs)

        if show_avg and len(x_arr) > 0 and len(values) > 0:
            with np.errstate(divide="ignore", invalid="ignore"):
                run_avg = [np.mean(values[:i]) for i in range(len(values))]
            last = run_avg[-1]
            label = f"{last:.3f} {format_unit_str(units)}" if last < 1 else f"{last:.0f} {format_unit_str(units)}"
            ax.plot(x_arr, run_avg, c="k", ls="-", lw=1, label=label)
            style_legend(ax, ncol=1)

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
        else:
            ax.set_xlim(x_arr.min(), x_arr.max())

        if ylim:
            ax.set_ylim(ylim)

        if savepath:
            savepath = Path(savepath) if Path(savepath).is_file() else Path(savepath) / "energy.pdf"
            fig.savefig(savepath, dpi=100)

        if show:
            plt.show()
        else:
            plt.close()
