"""Plotting support for KBI convergence and extrapolation."""

import itertools
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from kbkit.config.mplstyle import load_mplstyle
from kbkit.schema.property_result import PropertyResult
from kbkit.utils.format import format_unit_str
from kbkit.visualization.format import style_axes, style_legend

load_mplstyle()


class KBIAnalysisPlotter:
    """
    Visualize KBI convergence and extrapolation to the thermodynamic limit.

    Parameters
    ----------
    kbi: PropertyResult
        PropertyResult object containing KBI values for analysis.
    """

    def __init__(self, kbi: PropertyResult) -> None:
        self.result = kbi
        self.metadata = self.result.metadata

    def plot_rkbis(self, cmap: str | None = None, add_kbi_value: bool = True, savepath: str | Path | None = None):
        """Plot KBI analysis plots for all systems present.

        Parameters
        ----------
        cmap: str, optional
            Matplotlib colormap name.
        add_kbi_value: bool, optional
            Add horizontal value for KBI with ``weight_type='geometric'``.
        savepath: str | Path, optional
            Path to save figures to.
        """
        if savepath:
            savepath = Path(savepath) if Path(savepath).is_dir() else Path(savepath).parent

        if self.metadata is None:
            return

        for sys, mol_dict in self.metadata.items():
            for mols, kbi_obj in mol_dict.items():
                filepath = savepath / f"{sys}_{'_'.join(mols)}.pdf" if savepath is not None else None
                if kbi_obj.weight_type == "geometric":
                    kbi_obj.plot_kbi_compare_extrapolation(
                        weight_types=[kbi_obj.weight_type], add_kbi_value=add_kbi_value, cmap=cmap, filepath=filepath
                    )
                else:
                    kbi_obj.plot_kbi_compare(
                        weight_types=[kbi_obj.weight_type], add_kbi_value=add_kbi_value, cmap=cmap, filepath=filepath
                    )

    def plot_composition(
        self,
        x: np.ndarray,
        molecules: list[str],
        xlab: str,
        units: str = "cm^3/mol",
        cmap: str = "jet",
        savepath: str | Path | None = None,
        show: bool = True,
        **kwargs,
    ):
        """
        Plot KBIs as a function of composition for all systems present.

        Parameters
        ----------
        x: np.ndarray
            Composition to plot (1D array).
        molecules: list[str]
            List of molecules corresponding to shape of KBI array.
        xlab: str
            x-label for plot.
        show_err: bool, optional
            Show errorbar on plot.
        units: str, optional
            Units for KBI in figures.
        cmap: str, optional
            Matplotlib colormap.
        savepath: str | Path, optional
            Path to save figures to.
        show: bool, optional
            Show all figures.
        """
        kbi_res = self.result.to(units)
        kbi_shape = kbi_res.value.shape
        molecules = list(molecules)

        combos = list(itertools.product(range(kbi_shape[1]), range(kbi_shape[2])))
        unique_combos = [(i, j) for (i, j) in combos if i <= j]
        colors = plt.colormaps.get(cmap, "jet")(np.linspace(0, 1, len(unique_combos)))

        _, ax = plt.subplots(1, 1, figsize=(5, 4))
        for c, (i, j) in enumerate(unique_combos):
            label = f"{molecules[i]}-{molecules[j]}"
            ax.plot(x, kbi_res.value[:, i, j], color=colors[c], label=label, **kwargs)

        ax.set_xlabel(xlab)
        ax.set_ylabel(rf"$G_{{ij}}^\infty$ ({format_unit_str(units)})")
        ax.set_xlim(-0.05, 1.05)
        style_legend(ax, ncol=1)
        style_axes(ax, minorxticks=np.arange(0, 1, 0.1))

        if savepath:
            fpath = Path(savepath) if not Path(savepath).is_dir() else Path(savepath) / "kbis_composition.pdf"
            plt.savefig(fpath, dpi=100)
        if show:
            plt.show()
        else:
            plt.close()
