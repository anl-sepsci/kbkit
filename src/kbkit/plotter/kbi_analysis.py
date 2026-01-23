"""Plotting support for KBI convergence and extrapolation."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from kbkit.config.mplstyle import load_mplstyle
from kbkit.config.unit_registry import load_unit_registry
from kbkit.schema.property_result import PropertyResult
from kbkit.utils.format import format_unit_str

load_mplstyle()


class KBIAnalysisPlotter:
    """
    Visualize KBI convergence and extrapolation to the thermodynamic limit.

    Paramters
    ---------
    kbi: PropertyResult
        PropertyResult object containing KBI values and :class:`~kbkit.schema.kbi_metadata.KBIMetadata` for analysis.
    molecule_map: dict[str, str]
        Dictionary mapping molecules from simulation to desired labels in legend.
    """

    def __init__(self, kbi: PropertyResult, molecule_map: dict[str, str] | None = None) -> None:
        self.result = kbi
        self.metadata = self.result.metadata
        self.molecule_map = molecule_map
        self.ureg = load_unit_registry()
        self.Q_ = self.ureg.Quantity

    def _convert_kbi(self, array: np.ndarray, units: str = "cm^3/mol") -> np.ndarray:
        """Convert KBI metrics to the desired units."""
        return self.Q_(array, self.result.units).to(units).magnitude

    def plot(self, system_name: str, units: str = "cm^3/mol", savepath: str | None = None, show: bool = True):
        """Plot KBI analysis for a given system.

        This is a 1x3 subplot showing RDFs, corrected running KBI, and extrapolation of KBI to the thermodynamic limit for all molecular pairs in the system.

        Parameters
        ----------
        system_name: str
            System to plot.
        units: str, optional
            Units of KBIs to display.
        savepath: str, optional
            Path to save figure to. Will not save if this is ignored.
        show: bool, optional
            Display the figure.
        """
        fig, ax = plt.subplots(1, 3, figsize=(12, 3.6))
        for _, meta in self.metadata[system_name].items():
            r = meta.r
            g = meta.g
            rkbi = self._convert_kbi(meta.rkbi, units)
            scaled_rkbi = self._convert_kbi(meta.scaled_rkbi, units)
            scaled_rkbi_est = self._convert_kbi(meta.scaled_rkbi_est, units)
            molmap = self.molecule_map or {m: m for m in meta.mols}
            ax[0].plot(r, g, lw=2.5, label="-".join(molmap[mol] for mol in meta.mols))
            ax[1].plot(r, rkbi, lw=2.5)
            ax[2].plot(r, scaled_rkbi, lw=2.5)
            ax[2].plot(meta.r_fit, scaled_rkbi_est, lw=3, ls="--", c="k")

        ax[0].set_xlabel(r"$r$ [$nm$]")
        ax[1].set_xlabel(r"$R$ [$nm$]")
        ax[2].set_xlabel(r"$R$ [$nm$]")
        ax[0].set_ylabel("g(r)")
        ax[1].set_ylabel(f"G$_{{ij}}^R$ [{format_unit_str('cm^3/mol')}]")
        ax[2].set_ylabel(f"$R$ $G_{{ij}}^R$ [$nm$ {format_unit_str('cm^3/mol')}]")
        ax[0].legend(loc="best", ncol=1, fontsize="small", frameon=True)
        if savepath:
            savepath = Path(savepath)
            fpath = savepath if not savepath.is_dir() else savepath / f"{system_name}.pdf"
            plt.savefig(fpath, dpi=100)
        if show:
            plt.show()
        else:
            plt.close()

    def plot_all(self, units: str = "cm^3/mol", savepath: str | None = None, show: bool = True):
        """
        Plot KBI analysis subplots for all systems present in :class:`~kbkit.schema.kbi_metadata.KBIMetadata`.

        Parameters
        ----------
        units: str, optional
            Units for KBI in figures.
        savepath: str, optional
            Path to save figures to.
        show: bool, optional
            Show all figures.
        """
        savepath = Path(savepath)
        savepath = savepath if savepath.is_dir() else savepath.parent

        # plot all systems
        for sys, _ in self.metadata.items():
            fpath = savepath / f"{sys}.pdf"
            self.plot(system_name=sys, units=units, savepath=fpath, show=show)
