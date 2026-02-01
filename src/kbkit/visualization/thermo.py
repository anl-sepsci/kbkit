"""Plotting support for :class:`~kbkit.kbi.thermodynamics.KBThermo`."""

from itertools import combinations_with_replacement
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from cycler import cycler
from matplotlib.ticker import MultipleLocator

from kbkit.config.mplstyle import load_mplstyle
from kbkit.utils.format import format_unit_str

if TYPE_CHECKING:
    from kbkit.kbi.thermodynamics import KBThermo

load_mplstyle()

ARR_DIM_1 = 1
ARR_DIM_2 = 2
ARR_DIM_3 = 3
BINARY = 2
TERNARY = 3

class ThermoPlotter:
    """
    Plot properties from :class:`~kbkit.analysis.kb_thermo.KBThermo`.

    Parameters
    ----------
    thermo: KBThermo
        KBThermo object for a set of systems.
    molecule_map: dict[str, str]
        Dictionary mapping molecule names to desired names for figures.
    """

    def __init__(self, thermo: "KBThermo", molecule_map: dict[str, str] | None = None) -> None:
        self.thermo = thermo

        # create molecules; use names provided mapped to present molecule names
        molecules = self.thermo.systems.molecules

        # get names mapped
        self.molecule_map = molecule_map or {m: m for m in molecules}
        self.molecules = [self.molecule_map[mol] for mol in molecules]

    def plot(
        self,
        x: np.ndarray,
        y: np.ndarray,
        xlabel: str | None = None,
        ylabel: str | None = None,
        xlim: tuple | None = None,
        ylim: tuple | None = None,
        lw: float = 1.5,
        ls: str = "",
        marker: str = "o",
        cmap: str = "jet",
        figsize: tuple = (5, 4.5),
        savepath: str | Path | None = None,
        show: bool = True,
    ):
        """
        Create a plot for a given ``x`` and ``y`` arrays.

        Parameters
        ----------
        x: np.ndarray
            x-values to plot.
        y: np.ndarray
            y-values to plot.
        xlabel: str, optional
            Label for x-axis.
        ylabel: str, optional
            Label for y-axis.
        xlim: tuple, optional
            Limits for x-axis.
        ylim: tuple, optional
            Limits for y-axis.
        lw: float, optional
            Linewidth for timeseries.
        ls: str, optional
            Linestyle for timeseries.
        marker: str, optional
            Maker to display for timeseries.
        cmap: str, optional
            Matplotlib colormap.
        figsize: tuple, optional
            Size of the figure to display (height, width).
        savepath: str | Path, optional
            Path to save figure.
        show: bool, optional
            Display the figure.
        """
        n_colors = 5
        if y.ndim == ARR_DIM_2:
            n_colors = y.shape[1]
        elif y.ndim == ARR_DIM_3:
            combos = list(combinations_with_replacement(range(y.shape[1]), 2))
            n_colors = len(combos)

        cmap_obj = plt.get_cmap(cmap)
        colors = cmap_obj(np.linspace(0, 1, n_colors))

        fig, ax = plt.subplots(figsize=figsize)
        ax.set_prop_cycle(cycler(color=colors))

        if y.ndim == ARR_DIM_1:
            if x.ndim > 1:
                x = x[:, 0]
            ax.plot(x, y, lw=lw, ls=ls, marker=marker)

        elif y.ndim == ARR_DIM_2:
            ax.plot(x, y, lw=lw, ls=ls, marker=marker, label=self.molecules)

        elif y.ndim == ARR_DIM_3:
            if x.ndim > 1:
                x = x[:, 0]
            for i, j in combos:
                ax.plot(x, y[:, i, j], lw=lw, ls=ls, marker=marker, label=f"{self.molecules[i]}-{self.molecules[j]}")

        if xlabel:
            ax.set_xlabel(xlabel)
        if ylabel:
            ax.set_ylabel(ylabel)
        if xlim:
            ax.set_xlim(xlim)
        if ylim:
            ax.set_ylim(ylim)

        ax.legend()
        if savepath:
            fpath = Path(savepath)
            fpath = fpath if fpath.suffix else fpath / "thermo_property.pdf"
            fig.savefig(fpath, dpi=100)

        if show:
            plt.show()
        else:
            plt.close()
        return fig, ax

    def plot_property(
        self,
        name: str,
        units: str | None = None,
        ylabel: str | None = None,
        xmol: str | None = None,
        xlim: tuple | None = None,
        ylim: tuple | None = None,
        lw: float = 1.5,
        ls: str = "",
        marker: str = "o",
        cmap: str = "jet",
        figsize: tuple = (5, 4.5),
        savepath: str | Path | None = None,
        show: bool = True,
    ):
        """
        Create a plot for a given property name in KBThermo.

        Parameters
        ----------
        name: str
            Name of property to plot.
        units: str, optional
            Units of property.
        ylabel: str
            Label for y-axis.
        xmol: str, optional
            Name of molecule to use for x-axis.
        xlim: tuple, optional
            Limits for x-axis.
        ylim: tuple, optional
            Limits for y-axis.
        lw: float, optional
            Linewidth for timeseries.
        ls: str, optional
            Linestyle for timeseries.
        marker: str, optional
            Maker to display for timeseries.
        cmap: str, optional
            Matplotlib colormap.
        figsize: tuple, optional
            Size of the figure to display (height, width).
        savepath: str | Path, optional
            Path to save figure.
        show: bool, optional
            Display the figure.
        """
        x = self.thermo.systems.x
        values = getattr(self.thermo, name)
        try:
            values = values(units)
        except Exception:
            values = values()

        if values.ndim == ARR_DIM_2:
            if xmol is not None:
                x = x[:, self.thermo.systems.get_mol_index(xmol)]
        else:
            xmol = xmol or self.thermo.systems.molecules[0]
            x = x[:, self.thermo.systems.get_mol_index(xmol)]

        xlabel = r"$x_i$" if not xmol else rf"$x_{{{self.molecule_map[xmol]}}}$"
        ylabel = rf"{ylabel} ({format_unit_str(units)})" if units else ylabel

        if savepath:
            fpath = Path(savepath)
            fpath = fpath if fpath.suffix else fpath / f"{name.lower()}.pdf"

        return self.plot(x, values, xlabel, ylabel, xlim, ylim, lw, ls, marker, cmap, figsize, savepath, show)

    def plot_ternary(
        self,
        x: np.ndarray,
        y: np.ndarray,
        cbar_label: str | None = None,
        cmap: str = "jet",
        figsize: tuple = (8, 6),
        savepath: str | Path | None = None,
        show: bool = False,
    ):
        """
        Render a ternary system plot based on the ``x`` and ``y`` input.

        Parameters
        ----------
        x: np.ndarray
            x-values to plot.
        y: np.ndarray
            y-values to plot.
        cbar_label: str, optional
            Label for colorbar.
        cmap: str, optional
            Matplotlib colormap.
        figsize: tuple, optional
            Size of the figure to display (height, width).
        savepath: str | Path, optional
            Path to save figure.
        show: bool, optional
            Display the figure.
        """
        if x.shape[1] != TERNARY:
            raise ValueError("This is not a ternary system!")
        if y.ndim > 1:
            raise ValueError("Ternary plotting is only available for 1D y-values.")

        xtext, ytext, ztext = self.molecules
        a, b, c = x[:, 0], x[:, 1], x[:, 2]

        valid_mask = (a >= 0) & (b >= 0) & (c >= 0) & ~np.isnan(y) & ~np.isinf(y)
        a = a[valid_mask]
        b = b[valid_mask]
        c = c[valid_mask]
        values = y[valid_mask]

        fig, ax = plt.subplots(figsize=figsize, subplot_kw={"projection": "ternary"})
        ax.set_aspect(25)
        tp = ax.tricontourf(a, b, c, values, cmap=cmap, alpha=1, edgecolors="none", levels=40)  # type: ignore
        cbar_label = cbar_label or ""
        fig.colorbar(tp, ax=ax, aspect=25, label=cbar_label)

        ax.set_tlabel(xtext)  # type: ignore[attr-defined]
        ax.set_llabel(ytext)  # type: ignore[attr-defined]
        ax.set_rlabel(ztext)  # type: ignore[attr-defined]

        # Add grid lines on top
        ax.grid(True, which="major", linestyle="-", linewidth=1, color="k")

        ax.taxis.set_major_locator(MultipleLocator(0.10))  # type: ignore[attr-defined]
        ax.laxis.set_major_locator(MultipleLocator(0.10))  # type: ignore[attr-defined]
        ax.raxis.set_major_locator(MultipleLocator(0.10))  # type: ignore[attr-defined]

        if savepath:
            fpath = Path(savepath)
            fpath = fpath if fpath.suffix else fpath / "ternary_property.pdf"
            fig.savefig(fpath, dpi=100)

        if show:
            plt.show()
        else:
            plt.close()
        return fig, ax

    def plot_property_ternary(
        self,
        name: str,
        units: str | None = None,
        cmap: str = "jet",
        figsize: tuple = (8, 6),
        savepath: str | Path | None = None,
        show: bool = False,
    ):
        """
        Render a ternary system plot based on the property name.

        Parameters
        ----------
        name: str
            Property to plot.
        units: str, optional
            Units of property.
        cbar_label: str, optional
            Label for colorbar.
        cmap: str, optional
            Matplotlib colormap.
        figsize: tuple, optional
            Size of the figure to display (height, width).
        savepath: str | Path, optional
            Path to save figure.
        show: bool, optional
            Display the figure.
        """
        x = self.thermo.systems.x
        y = getattr(self.thermo, name)
        try:
            y = y(units)
        except Exception:
            y = y()

        name = name.replace("_", " ")
        cbar_label = f"{name} ({format_unit_str(units)})" if units else name

        return self.plot_ternary(
            x=x, y=y, cbar_label=cbar_label, cmap=cmap, figsize=figsize, savepath=savepath, show=show
        )

    def plot_activity_coef_deriv_fits(
        self,
        xlim: tuple | None = None,
        ylim: tuple | None = None,
        lw: float = 2.5,
        ls: str = "",
        marker: str = "o",
        cmap: str = "jet",
        figsize: tuple = (5, 4.5),
        savepath: str | Path | None = None,
        show: bool = True,
    ):
        """Plot the fits to activity coefficient derivatives, for the polynomial ``integration_type``.

        Parameters
        ----------
        xlim: tuple, optional
            Limits for x-axis.
        ylim: tuple, optional
            Limits for y-axis.
        lw: float, optional
            Linewidth for timeseries.
        ls: str, optional
            Linestyle for timeseries.
        marker: str, optional
            Maker to display for timeseries.
        cmap: str, optional
            Matplotlib colormap.
        figsize: tuple, optional
            Size of the figure to display (height, width).
        savepath: str | Path, optional
            Path to save figure.
        show: bool, optional
            Display the figure.
        """
        x = self.thermo.systems.x
        values = self.thermo.dlngamma_dxi()
        n_colors = values.shape[1]
        cmap_obj = plt.get_cmap(cmap)
        colors = cmap_obj(np.linspace(0, 1, n_colors))

        fig, ax = plt.subplots(figsize=figsize)
        ax.set_prop_cycle(cycler(color=colors))
        ax.plot(x, values, lw=lw, ls=ls, marker=marker, label=self.molecules)

        # now add fit fns
        for _, meta in self.thermo.activity_metadata.by_types["derivative"].items():
            if not meta.has_fn:
                continue
            ax.plot(meta.x_eval, meta.y_eval, c="k", lw=1.5, ls="-")

        ax.legend()
        ax.set_xlabel(r"$x_i$")
        ax.set_ylabel(r"$\partial \ln \gamma_i / \partial x_i$")
        if xlim:
            ax.set_xlim(xlim)
        if ylim:
            ax.set_ylim(ylim)

        if savepath:
            fpath = Path(savepath)
            fpath = fpath if fpath.suffix else fpath / "activity_coef_deriv_fits.pdf"
            fig.savefig(fpath, dpi=100)

        if show:
            plt.show()
        else:
            plt.close()
        return fig, ax

    def plot_binary_mixing(
        self,
        xmol,
        xlim: tuple | None = None,
        ylim: tuple | None = None,
        lw: float = 1.5,
        ls: str = "",
        marker: str = "o",
        cmap: str = "jet",
        figsize: tuple = (5, 4.5),
        savepath: str | Path | None = None,
        show: bool = True,
    ):
        """
        Plots the contributions to Gibbs mixing free energy for binary mixtures.

        Parameters
        ----------
        xmol: str
            Molecule for x-axis.
        xlim: tuple, optional
            Limits for x-axis.
        ylim: tuple, optional
            Limits for y-axis.
        lw: float, optional
            Linewidth for timeseries.
        ls: str, optional
            Linestyle for timeseries.
        marker: str, optional
            Maker to display for timeseries.
        cmap: str, optional
            Matplotlib colormap.
        figsize: tuple, optional
            Size of the figure to display (height, width).
        savepath: str | Path, optional
            Path to save figure.
        show: bool, optional
            Display the figure.
        """
        # plot mixing properties
        fig, ax = plt.subplots(figsize=figsize)
        cmap_obj = plt.get_cmap(cmap)
        colors = cmap_obj(np.linspace(0, 1, 5))
        xmol_mix = xmol or self.thermo.systems.molecules[0]
        xi = self.thermo.systems.x[:, self.thermo.systems.get_mol_index(xmol_mix)]
        ax.set_prop_cycle(cycler(color=colors))
        ax.plot(xi, self.thermo.h_mix(), lw=lw, ls=ls, marker=marker, label=r"$\Delta H_{mix}$")
        ax.plot(xi, -self.thermo.temperature() * self.thermo.s_ex(), lw=lw, ls=ls, marker=marker, label=r"$-TS^{EX}$")
        ax.plot(xi, self.thermo.g_ex(), lw=lw, ls=ls, marker=marker, label=r"$G^{EX}$")
        ax.plot(xi, -self.thermo.temperature() * self.thermo.g_id() / self.thermo.temperature(), lw=lw, ls=ls, marker=marker, label=r"$-TS^{id}$")
        ax.plot(xi, self.thermo.g_mix(), lw=lw, ls=ls, marker=marker, label=r"$\Delta G_{mix}$")
        ax.set_xlabel(rf"$x_{{{xmol_mix}}}$")
        unit_str = format_unit_str("kJ/mol")
        ax.set_ylabel(fr"Thermodynamic Properties ({unit_str})")
        ax.legend()

        if xlim:
            ax.set_xlim(xlim)
        if ylim:
            ax.set_ylim(ylim)

        if savepath:
            fpath = Path(savepath)
            fpath = fpath if fpath.suffix else fpath / "thermodyanmic_mixing_properties.pdf"
            fig.savefig(fpath, dpi=100)

        if show:
            plt.show()
        else:
            plt.close()
        return fig, ax

    def make_figures(
        self,
        savepath: str | Path,
        xmol: str | None = None,
        cmap: str = "jet",
    ) -> None:
        """
        Create default figures for a binary or ternary system.

        Parameters
        ----------
        xmol: str, optional
            Molecule for x-axis.
        cmap: str, optional
            Matplotlib colormap.
        savepath: str | Path, optional
            File location to save figure.
        """
        savepath = Path(savepath)

        # plot kbis
        self.plot_property(
            name="kbi",
            units="cm^3/mol",
            xmol=xmol,
            ylabel=r"$G_{ij}^{\infty}$",
            cmap=cmap,
            savepath=Path(savepath) / "kbi.pdf",
            show=False,
        )

        # plot activity coeffs.
        if self.thermo.activity_integration_type == "polynomial":
            self.plot_activity_coef_deriv_fits(
                cmap=cmap, savepath=Path(savepath) / "ln_activity_coef_deriv_fits.pdf", show=False
            )
        else:
            self.plot_property(
                name="dlngamma_dxi",
                xmol=xmol,
                ylabel=r"$\partial \ln \gamma_i / \partial x_i$",
                cmap=cmap,
                savepath=Path(savepath) / "ln_activity_coef_derivs.pdf",
                show=False,
            )

        self.plot_property(
            name="lngamma",
            xmol=xmol,
            ylabel=r"$\ln \gamma_i$",
            cmap=cmap,
            savepath=Path(savepath) / "activity_coef.pdf",
            show=False,
        )

        # plot structure factors
        self.plot_property(
            name="s0_ij",
            xmol=xmol,
            ylabel=r"$\hat{S}_{ij}(0)$",
            cmap=cmap,
            savepath=Path(savepath) / "partial_structure_factors.pdf",
            show=False,
        )

        system_types = {BINARY: "BINARY", TERNARY: "TERNARY"}
        if self.thermo.systems.n_i not in system_types:
            return

        sys_type = system_types[self.thermo.systems.n_i]
        if sys_type == "BINARY":
            # plot structure factors
            self.plot_property(
                name="det_H",
                units="kJ/mol",
                xmol=xmol,
                ylabel=r"$|H|$",
                cmap=cmap,
                savepath=Path(savepath) / "hessian_determinant.pdf",
                show=False,
            )

            # plot mixing
            self.plot_binary_mixing(
                xmol=xmol,
                cmap=cmap,
                savepath=Path(savepath) / "thermodynamic_mixing_properties.pdf",
                show=False,
            )

        else:
            self.plot_property(
                name="det_H",
                units="kJ/mol",
                cmap=cmap,
                savepath=Path(savepath) / "hessian_determinant.pdf",
                show=False,
            )

            for prop in ("h_mix", "s_ex", "g_ex", "g_id", "g_mix"):
                self.plot_property_ternary(
                    name=prop,
                    units="kJ/mol/K" if prop.startswith("s_") else "kJ/mol",
                    cmap=cmap,
                    savepath=Path(savepath) / f"{prop}.pdf",
                    show=False,
                )
