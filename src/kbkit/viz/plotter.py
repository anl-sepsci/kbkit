"""Plotting support for Kirkwood-Buff Analysis."""

import os
import warnings
from itertools import combinations_with_replacement
from pathlib import Path
from typing import Any, Callable, List, Tuple, TypedDict

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator

from kbkit.analysis.kb_thermo import KBThermo
from kbkit.utils.format import format_unit_str
from kbkit.config.mplstyle import load_mplstyle

load_mplstyle() # load figure config file
warnings.filterwarnings("ignore")

BINARY_SYSTEM = 2
TERNAY_SYSTEM = 3


class _SinglePlotSpec(TypedDict):
    # a class for the single-data plot
    x_data: np.ndarray
    y_data: np.ndarray
    ylabel: str
    filename: str
    fit_fns: dict[str, Callable[..., Any]] | None


class _MultiPlotSpec(TypedDict):
    # class for multi-series plot
    x_data: np.ndarray
    y_series: List[Tuple[np.ndarray, str, str, str]]
    ylabel: str
    filename: str
    multi: bool


class Plotter:
    r"""
    A class for plotting results from KB analysis (:class:`kbkit.kb.kb_thermo.KBThermo`).

    Parameters
    ----------
    kb_obj: KBThermo
        Instance of KBThermo.
    x_mol: str, optional
        Molecule to use for labeling x-axis in figures for binary systems. Defaults to first element in molecule list.
    molecule_map: dict[str, str], optional.
        Dictionary of molecule ID in topology mapped to molecule names for figure labeling. Defaults to using molecule names in topology.
    """

    def __init__(
        self,
        kb_obj: KBThermo,
        molecule_map: dict[str, str],
        x_mol: str = "",
    ) -> None:
        self.kb = kb_obj
        self.x_mol = x_mol
        self._setup_folders()
        self.molecule_map = molecule_map

    def _setup_folders(self) -> None:
        # create folders for figures if they don't exist
        self.kb_dir = Path(os.path.join(self.kb.base_path, "kb_analysis"))
        self.sys_dir = Path(os.path.join(self.kb_dir, "system_figures"))
        for path in (self.kb_dir, self.sys_dir):
            if not path.exists():
                os.mkdir(path)

    @property
    def molecule_map(self) -> dict[str, str]:
        """dict[str, str]: Dictionary mapping molecule ID in topology file to names for figure labels."""
        if not isinstance(self._molecule_map, dict):
            raise TypeError(f"Type for map: type({type(self._molecule_map)}) is not dict.")
        return self._molecule_map

    @molecule_map.setter
    def molecule_map(self, mapped: dict[str, str]) -> None:
        # if not specified fall back on molecule name in topology file
        if not mapped:
            mapped = {mol: mol for mol in self.kb.unique_molecules}

        # check that all molecules are defined in map
        found_mask = np.array([mol not in self.kb.unique_molecules for mol in mapped])
        if any(found_mask):
            missing_mols = np.fromiter(mapped.keys(), dtype=str)[found_mask]
            raise ValueError(
                f"Molecules missing from molecule_map: {', '.join(missing_mols)}. "
                f"Available molecules: {', '.join(self.kb.unique_molecules)}"
            )

        self._molecule_map = mapped

    @property
    def x_mol(self) -> str:
        """str: Molecule to use for x-axis labels in 2D plots."""
        if not isinstance(self._x_mol, str):
            raise TypeError(f"Type for mol: type({type(self._x_mol)}) is not str.")
        return self._x_mol

    @x_mol.setter
    def x_mol(self, mol: str) -> None:
        # if not specified default to first molecule in list
        if not mol:
            self._x_mol = self.kb.unique_molecules[0]

        # check if mol is in unique molecules
        if mol not in self.kb.unique_molecules:
            raise ValueError(f"Molecule {mol} not in available molecules: {', '.join(self.kb.unique_molecules)}")

        self._x_mol = mol

    @property
    def unique_names(self) -> list[str]:
        """list: Names of molecules to use in figure labels."""
        return [self.molecule_map[mol] for mol in self.kb.unique_molecules]

    @property
    def _x_idx(self) -> int:
        # get index of x_mol in kb.unique_molecules
        return self.kb._mol_idx(self.x_mol)

    def _get_rdf_colors(self, cmap: str = "jet") -> dict[str, dict[str, tuple[float, ...]]]:
        # create a colormap mapping pairs of molecules with a color
        if "_color_dict" not in self.__dict__:
            # Collect all unique unordered molecule pairs across systems
            all_pairs: set[tuple[str, ...]] = set()
            for system in self.kb.system_properties:
                try:
                    mol_ids = self.kb.system_properties[system].topology.molecules
                    pairs = combinations_with_replacement(mol_ids, 2)
                    all_pairs.update(tuple(sorted(p)) for p in pairs)
                except Exception as e:
                    print(f"Error processing system '{system}': {e}")

            # Assign unique colors to each pair
            all_pairs_list: list[tuple[str, ...]] = list(all_pairs)
            all_pairs_list = sorted(all_pairs_list)
            n_pairs = len(all_pairs_list)
            try:
                colormap = plt.cm.get_cmap(cmap, n_pairs)
            except Exception as e:
                print(f"Error creating colormap '{cmap}': {e}")
                colormap = plt.cm.get_cmap("jet", n_pairs)

            color_map = {}
            for i, pair in enumerate(all_pairs_list):
                try:
                    color_map[pair] = colormap(i)
                except Exception as e:
                    print(f"Error assigning color for pair {pair}: {e}")
                    color_map[pair] = (0, 0, 0, 1)  # fallback to black

            # Build nested dict color_dict[mol_i][mol_j]
            color_dict: dict[str, dict[str, tuple[float, ...]]] = {}
            for mol_i, mol_j in all_pairs:
                color = color_map.get((mol_i, mol_j), (0, 0, 0, 1))
                color_dict.setdefault(mol_i, {})[mol_j] = color
                color_dict.setdefault(mol_j, {})[mol_i] = color

            self._color_dict = color_dict

        return self._color_dict

    def plot_system_kbi_analysis(
        self,
        system: str,
        units: str = "",
        alpha: float = 0.6,
        cmap: str = "jet",
        show: bool = False,
    ) -> None:
        """
        Plot KBI analysis results for a specific system. Creates a 1 x 3 subplot showing RDFs and KBIs including fit to the thermodynamic limit for all unique molecule pairs.

        Parameters
        ----------
        system: str
            System name to plot.
        units: str, optional
            Units for KBI calculation. Default is 'cm^3/mol'.
        alpha: float, optional
            Transparency for lines in plot. Default is 0.6.
        cmap: str, optional
            Matplotlib colormap. Default is 'jet'.
        show: bool, optional
            Display figure. Default is False.
        """
        # add legend to above figure.
        color_dict = self._get_rdf_colors(cmap=cmap)
        kbi_system_dict = self.kb.kbi_dict().get(system, {})
        units = "cm^3/mol" if units == "" else units

        fig, ax = plt.subplots(1, 3, figsize=(12, 4))
        for mols, mol_dict in kbi_system_dict.items():
            mol_i, mol_j = mols.split("-")
            color = color_dict.get(mol_i, {}).get(mol_j)

            rkbi = self.kb.Q_(mol_dict["rkbi"], "nm^3/molecule").to(units).magnitude
            lkbi = self.kb.Q_(mol_dict["lambda_kbi"], "nm^3/molecule").to(units).magnitude
            lkbi_fit = self.kb.Q_(mol_dict["lambda_kbi_fit"], "nm^3/molecule").to(units).magnitude
            kbi_inf = self.kb.Q_(mol_dict["kbi_inf"], "nm^3/molecule").to(units).magnitude

            ax[0].plot(mol_dict["r"], mol_dict["g"], lw=3, c=color, alpha=alpha, label=mols)
            ax[1].plot(
                mol_dict["r"],
                rkbi,
                lw=3,
                c=color,
                alpha=alpha,
                label=f"G$_{{ij}}^R$: {rkbi[-1]:.4g}",
            )
            ax[2].plot(
                mol_dict["lambda"],
                lkbi,
                lw=3,
                c=color,
                alpha=alpha,
                label=rf"G$_{{ij}}^\infty$: {kbi_inf:.4g}",
            )
            ax[2].plot(mol_dict["lambda_fit"], lkbi_fit, ls="--", lw=4, c="k")

        ax[0].set_xlabel("r / nm")
        ax[1].set_xlabel("r / nm")
        ax[2].set_xlabel(r"$\lambda$")
        ax[0].set_ylabel("g(r)")
        ax[1].set_ylabel(f"G$_{{ij}}^R$ / {format_unit_str(units)}")
        ax[2].set_ylabel(rf"$\lambda$ G$_{{ij}}^R$ / {format_unit_str(units)}")
        ax[0].legend(
            loc="lower center",
            bbox_to_anchor=(0.5, 1.01),
            ncol=1,
            fontsize="small",
            fancybox=True,
            shadow=True,
        )
        ax[1].legend(
            loc="lower center",
            bbox_to_anchor=(0.5, 1.01),
            ncol=2,
            fontsize="small",
            fancybox=True,
            shadow=True,
        )
        ax[2].legend(
            loc="lower center",
            bbox_to_anchor=(0.5, 1.01),
            ncol=2,
            fontsize="small",
            fancybox=True,
            shadow=True,
        )
        plt.savefig(os.path.join(self.sys_dir, f"{system}_rdfs_kbis.png"))
        if show:
            plt.show()
        else:
            plt.close()

    def plot_rdf_kbis(self, units: str = "cm^3/mol", show: bool = False) -> None:
        """
        For each system, create a plot (:meth:`plot_system_kbi_analysis`) showing KBI analysis for each molecular pair.

        Parameters
        ----------
        units: str, optional
            Units to plot KBI in. Default is 'cm^3/mol'.
        show: bool, optional
            Display figures. Default is False.
        """
        for system in self.kb.systems:
            self.plot_system_kbi_analysis(system, units=units, show=show)

    def plot_system_rdf(
        self,
        system: str,
        xlim: tuple[float, float] = (0.0, 0.0),
        ylim: tuple[float, float] = (0.0, 0.0),
        line: bool = False,
        cmap: str = "jet",
        alpha: float = 0.6,
        show: bool = True,
    ) -> None:
        """
        Plot all RDFs for a specific system with inset zoom.

        Parameters
        ----------
        system: str
            System name to plot.
        xlim: tuple, optional
            Limits for inset zoom x-axis. Default (4,5).
        ylim: tuple, optional
            Limits for inset zoom y-axis. Default (0.99,1.01).
        line: bool, optional
            Add line at y=1 to show deviation. Default False.
        cmap: str, optional
            Matplotlib colormap. Default 'jet'.
        alpha: float, optional
            Transparency of lines. Default 0.6.
        show: bool, optional
            Display figure. Default True.
        """
        # set up main fig/axes
        fig, main_ax = plt.subplots(figsize=(5, 4))
        main_ax.set_box_aspect(0.6)
        xlim = (4, 5) if any(xlim) != 0 else xlim
        ylim = (0.99, 1.01) if any(ylim) != 0 else ylim
        inset_ax = main_ax.inset_axes(
            (0.65, 0.12, 0.3, 0.3),  # [x, y, width, height] w.r.t. axes
            xlim=xlim,
            ylim=ylim,  # sets viewport &amp; tells relation to main axes
            # xticklabels=[], yticklabels=[]
        )
        inset_ax.tick_params(axis="x", labelsize=11)
        inset_ax.tick_params(axis="y", labelsize=11)

        color_dict = self._get_rdf_colors(cmap=cmap)
        kbi_system_dict = self.kb.kbi_dict().get(system, {})

        for mols, mol_dict in kbi_system_dict.items():
            mol_i, mol_j = mols.split("-")
            color = color_dict.get(mol_i, {}).get(mol_j)

            # add plot content
            for ax in main_ax, inset_ax:
                ax.plot(mol_dict["r"], mol_dict["g"], c=color, alpha=alpha, label=mols)  # first example line

            # add zoom leaders
            main_ax.indicate_inset_zoom(inset_ax, edgecolor="black")

        if line:
            inset_ax.axhline(1.0, c="k", ls="--", lw=1.5)

        main_ax.set_xlabel("r / nm")
        main_ax.set_ylabel("g(r)")
        main_ax.legend(
            loc="lower center",
            bbox_to_anchor=(0.5, 1.01),
            ncol=2,
            fontsize="small",
            fancybox=True,
            shadow=True,
        )
        if show:
            plt.show()
        else:
            plt.close()

    def plot_kbis(self, units: str = "cm^3/mol", cmap: str = "jet", show: bool = False) -> None:
        """
        Plot KBI values in the thermodynamic limit as a function of composition.

        Parameters
        ----------
        units: str, optional
            Units for KBI calculation. Default is 'cm^3/mol'.
        cmap: str, optional
            Matplotlib colormap. Default is 'jet'.
        show: bool, optional
            Display figure. Default is False.
        """
        color_dict = self._get_rdf_colors(cmap=cmap)
        fig, ax = plt.subplots(1, 1, figsize=(5, 4))
        legend_info = {}
        for mol_dict in self.kb.kbi_dict().values():
            for mols in mol_dict:
                mol_i, mol_j = mols.split("-")
                i, j = [self.kb._mol_idx(mol) for mol in (mol_i, mol_j)]
                color = color_dict.get(mol_i, {}).get(mol_j)
                kbi = self.kb.kbi_mat()[:, i, j]
                kbi = self.kb.Q_(kbi, "nm^3/molecule").to(units).magnitude
                line = ax.scatter(self.kb.mol_fr[:, self._x_idx], kbi, c=color, marker="s", lw=1.8, label=mols)
                if mols not in legend_info:
                    legend_info[mols] = line
        lines = list(legend_info.values())
        labels = list(legend_info.keys())
        ax.legend(
            lines,
            labels,
            loc="lower center",
            bbox_to_anchor=(0.5, 1.01),
            ncol=2,
            fontsize="small",
            fancybox=True,
            shadow=True,
        )
        ax.set_xlim(-0.05, 1.05)
        ax.set_xticks(ticks=np.arange(0, 1.1, 0.1))
        ax.set_xlabel(f"x$_{{{self.molecule_map[self.x_mol]}}}$")
        ax.set_ylabel(rf"G$_{{ij}}^{{\infty}}$ / {format_unit_str(units)}")
        plt.savefig(self.kb_dir + f"/composition_kbi_{units.replace('^', '').replace('/', '_')}.png")
        if show:
            plt.show()
        else:
            plt.close()

    def _get_plot_spec(self, prop: str, energy_units: str = "kJ/mol") -> _SinglePlotSpec | _MultiPlotSpec:
        # get the figure specifications for a given property
        if prop in ["lngamma", "dlngamma", "lngamma_fits", "dlngamma_fits"]:
            # Handle the properties that return a _SinglePlotSpec
            return _SinglePlotSpec(
                x_data=self.kb.mol_fr,
                y_data=self.kb.lngammas() if prop.endswith("gamma") else self.kb.dlngammas_dxs(),
                ylabel=r"$\ln \gamma_{i}$"
                if prop.endswith("gamma")
                else r"$\partial \ln(\gamma_{i})$ / $\partial x_{i}$",
                filename=f"{prop}.png",  # Simplified for example
                fit_fns={mol: self.kb.lngamma_fn(mol) for mol in self.kb.unique_molecules} if "fits" in prop else None,
            )

        elif prop in ["mixing", "excess"]:
            # Handle the properties that return a _MultiPlotSpec
            y_series_list = [
                (self.kb.hmix(energy_units), "violet", "s", r"$\Delta H_{mix}$"),
                (-self.kb.temperature() * self.kb.se(energy_units), "limegreen", "o", r"$-TS^E$"),
            ]
            if prop == "mixing":
                y_series_list.extend(
                    [
                        (self.kb.gid(energy_units), "darkorange", "<", r"$G^{id}$"),
                        (self.kb.gm(energy_units), "mediumblue", "^", r"$\Delta G_{mix}$"),
                    ]
                )
            else:  # prop == excess
                y_series_list.append((self.kb.ge(energy_units), "mediumblue", "^", r"$G^E$"))

            return _MultiPlotSpec(
                x_data=self.kb.mol_fr[:, self._x_idx],
                y_series=y_series_list,
                ylabel=rf"Contributions to $\Delta G_{{mix}}$ / {format_unit_str(energy_units)}"
                if prop == "mixing"
                else f"Excess Properties / {format_unit_str(energy_units)}",
                filename=f"gibbs_{'mixing' if prop == 'mixing' else 'excess'}_contributions.png",
                multi=True,
            )

        elif prop in ["i0", "det_h"]:
            # Handle other single-data plots
            return _SinglePlotSpec(
                x_data=self.kb.mol_fr[:, self._x_idx],
                y_data=self.kb.i0(units="1/cm") if prop == "i0" else self.kb.det_hessian(units=energy_units),
                ylabel=f"I$_0$ / {format_unit_str('cm^{-1}')}"
                if prop == "i0"
                else f"$|H_{{ij}}|$ / {format_unit_str(energy_units)}",
                filename=f"saxs_{'I0' if prop == 'i0' else 'det_hessian'}.png",
                fit_fns=None,
            )
        else:
            raise ValueError(f"Unknown property: '{prop}'")

    def _render_binary_plot(
        self,
        spec: _SinglePlotSpec | _MultiPlotSpec,
        ylim: tuple[float, float] = (0.0, 0.0),
        show: bool = True,
        cmap: str = "jet",
        marker: str = "o",
    ) -> None:
        # create a binary plot for a given property
        fig, ax = plt.subplots(figsize=(5, 4))

        # Check the type of the spec object using a runtime check
        if type(spec) is _MultiPlotSpec:
            x = spec["x_data"]
            for y_data, color, mk, label in spec["y_series"]:
                if isinstance(x, (list, np.ndarray)) and isinstance(y_data, (list, np.ndarray)):
                    ax.scatter(x, y_data, c=color, marker=mk, label=label)
                else:
                    raise TypeError(
                        f"Incompatible data type for plotting, x-data type({type(spec['x_data'])}); y-data type({type(y_data)})."
                    )

            ax.legend(
                loc="lower center",
                bbox_to_anchor=(0.5, 1.01),
                ncol=2,
                fontsize="small",
                fancybox=True,
                shadow=True,
            )

        elif type(spec) is _SinglePlotSpec:
            x_data = spec["x_data"]
            y_data = spec["y_data"]

            if y_data.ndim == 1:  # single y_data for single component
                ax.scatter(x_data, y_data, c="mediumblue", marker=marker)

            else:  # for many components
                colors = plt.cm.get_cmap(cmap)(np.linspace(0, 1, self.kb.n_comp))
                fit_fns = spec.get("fit_fns", None)

                for i, mol in enumerate(self.kb.unique_molecules):
                    xi = x_data[:, self._x_idx] if self.kb.n_comp == BINARY_SYSTEM else x_data[:, i]
                    yi = y_data[:, i]
                    ax.scatter(xi, yi, c=[colors[i]], marker=marker, label=self.molecule_map[mol])

                    if fit_fns is not None:
                        fit = fit_fns[mol]
                        xfit = np.arange(0, 1.01, 0.01)
                        ax.plot(xfit, fit(xfit), c=colors[i], lw=2)

                ax.legend(
                    loc="lower center",
                    bbox_to_anchor=(0.5, 1.01),
                    ncol=2,
                    fontsize="small",
                    fancybox=True,
                    shadow=True,
                )

        ax.set_xlabel(f"x$_{{{self.molecule_map[self.x_mol]}}}$" if self.kb.n_comp == BINARY_SYSTEM else "x$_i$")
        ax.set_ylabel(spec["ylabel"])
        ax.set_xlim(-0.05, 1.05)
        ax.set_xticks(np.arange(0, 1.1, 0.1))

        if any(ylim) != 0:
            ax.set_ylim(*ylim)
        elif "multi" not in spec and "y_data" in spec:
            y_max, y_min = np.nanmax(spec["y_data"]), np.nanmin(spec["y_data"])
            pad = 0.1 * (y_max - y_min) if y_max != y_min else 0.05
            y_lb = 0 if spec["y_data"].ndim == 1 else -0.05
            ax.set_ylim(min([y_lb, y_min - pad]), max([0.05, y_max + pad]))

        plt.savefig(os.path.join(self.kb_dir, str(spec["filename"])))
        if show:
            plt.show()
        else:
            plt.close()

    def _render_ternary_plot(
        self,
        property_name: str,
        energy_units: str = "kJ/mol",
        cmap: str = "jet",
        show: bool = False,
    ) -> None:
        # create a ternary plot for a given property
        _map = {
            "ge": self.kb.ge(energy_units),
            "gm": self.kb.gm(energy_units),
            "hmix": self.kb.hmix(energy_units),
            "se": self.kb.se(energy_units),
            "i0": self.kb.i0("1/cm"),
            "det_h": self.kb.det_hessian(energy_units),
        }
        arr = np.asarray(_map[property_name])
        xtext, ytext, ztext = self.unique_names
        a, b, c = self.kb.mol_fr[:, 0], self.kb.mol_fr[:, 1], self.kb.mol_fr[:, 2]

        valid_mask = (a >= 0) & (b >= 0) & (c >= 0) & ~np.isnan(arr) & ~np.isinf(arr)
        a = a[valid_mask]
        b = b[valid_mask]
        c = c[valid_mask]
        values = arr[valid_mask]

        fig, ax = plt.subplots(figsize=(8, 6), subplot_kw={"projection": "ternary"})
        ax.set_aspect(25)
        tp = ax.tricontourf(a, b, c, values, cmap=cmap, alpha=1, edgecolors="none", levels=40)  # type: ignore
        fig.colorbar(tp, ax=ax, aspect=25, label=f"{property_name} / kJ mol$^{-1}$")

        ax.set_tlabel(xtext)  # type: ignore[attr-defined]
        ax.set_llabel(ytext)  # type: ignore[attr-defined]
        ax.set_rlabel(ztext)  # type: ignore[attr-defined]

        # Add grid lines on top
        ax.grid(True, which="major", linestyle="-", linewidth=1, color="k")

        ax.taxis.set_major_locator(MultipleLocator(0.10))  # type: ignore[attr-defined]
        ax.laxis.set_major_locator(MultipleLocator(0.10))  # type: ignore[attr-defined]
        ax.raxis.set_major_locator(MultipleLocator(0.10))  # type: ignore[attr-defined]

        plt.savefig(os.path.join(self.kb_dir, f"ternary_{property_name}.png"))
        if show:
            plt.show()
        else:
            plt.close()

    def available_properties(self) -> None:
        r"""Print out the available properties to plot with :meth:`plot_property`."""
        print(
            "Properties: ",
            [
                "kbi",
                "lngamma",
                "dlngamma",
                "lngamma_fits",
                "dlngamma_fits",
                "excess",
                "mixing",
                "gm",
                "ge",
                "hmix",
                "se",
                "i0",
                "det_h",
            ],
        )

    def plot(
        self,
        prop: str,
        system: str = "",
        units: str = "",
        cmap: str = "jet",
        marker: str = "o",
        xlim: tuple[float, float] = (0.0, 0.0),
        ylim: tuple[float, float] = (0.0, 0.0),
        show: bool = True,
    ) -> None:
        r"""
        Master plot function. Handles property selection, data prep, and plotting.

        Automatically determines the correct type of plot.

        Parameters
        ----------
        prop: str
            Which property to plot? Options include:
                - '`rdf`': (System required) Radial distribution function for all pairwise interactions.
                - '`kbi`': KBI as a function of composition
                - '`lngamma`': Activity coefficients for each molecule.
                - '`dlngamma`': Derivative of activity coefficients with respect to mol fraction of each molecule.
                - '`lngamma_fits`': Activity coefficient function.
                - '`dlngamma_fits`': Fit of polynomial function to activity coefficient derivative.
                - '`excess`': (Binary systems only) Excess thermodynamic properties as a function of composition.
                - '`mixing`': (Binary systems only) Mixing thermodynamic properties as a function of composition.
                - '`gm`': Gibbs free energy of mixing.
                - '`ge`': Gibbs excess free energy.
                - '`hmix`': Mixing enthalpy.
                - '`se`': Excess entropy.
                - '`i0`': SAXS intensity as q :math:`\rightarrow` 0.
                - '`det_h`': Determinant of Hessian.
        system: str, optional
            System to plot for RDF/KBI analysis of specific system (default None).
        units: str, optional
            Units for plotting. If `thermo_property` is '`kbi`', units refer to KBI values (default 'cm^3/mol'), otherwise units refer to energy (default 'kJ/mol').
        cmap: str, otpional
            Matplotlib colormap (default 'jet').
        marker: str, optional
            Marker shape for scatterplots (default 'o').
        xlim: list, optional
            For specific system, x-axis limits of zoomed in RDF.
        ylim: list, optional
            For specific system, y-axis limits of zoomed in RDF; otherwise: y-axis in binary and activity coefficient plots.
        show: bool, optional
            Display figure (default True).
        """
        prop_key = prop.lower()
        if prop_key not in self.available_properties():
            raise ValueError(f"Property {prop_key} not valid.")
        
        energy_units = units if units else "kJ/mol"
        kbi_units = units if units else "cm^3/mol"

        if system:
            # plot system rdfs
            if prop_key == "rdf":
                self.plot_system_rdf(system=system, xlim=xlim, ylim=ylim, line=True, cmap=cmap, show=show)

            # plot system kbis
            elif prop_key == "kbi":
                self.plot_system_kbi_analysis(system=system, units=kbi_units, cmap=cmap, show=show)

            else:
                print("WARNING: Invalid plot option specified! System specific include rdf and kbi.")

        elif prop_key == "kbi":
            self.plot_kbis(units=kbi_units, cmap=cmap, show=show)

        elif self.kb.n_comp == BINARY_SYSTEM or prop_key in {
            "lngamma",
            "dlngamma",
            "lngamma_fits",
            "dlngamma_fits",
        }:
            spec = self._get_plot_spec(prop_key, energy_units=energy_units)
            self._render_binary_plot(spec, marker=marker, ylim=ylim, cmap=cmap, show=show)

        elif self.kb.n_comp == TERNAY_SYSTEM and prop_key in {"gm", "ge", "hmix", "se", "i0", "det_h"}:
            self._render_ternary_plot(property_name=prop_key, energy_units=energy_units, cmap=cmap, show=show)

        elif self.kb.n_comp > TERNAY_SYSTEM:
            print(
                f"WARNING: plotter does not support {prop_key} for more than 3 components. ({self.kb.n_comp} components detected.)"
            )

    def make_figures(self, energy_units: str = "kJ/mol") -> None:
        r"""
        Create all figures for Kirkwood-Buff analysis.

        Parameters
        ----------
        energy_units: str
            Energy units for calculations. Default is 'kJ/mol'.
        """
        # create figure for rdf/kbi analysis
        self.plot_rdf_kbis(show=False)
        # plot KBI as a function of composition
        self.plot_kbis(units="cm^3/mol", show=False)

        # create figures for properties independent of component number
        for thermo_prop in ["lngamma", "dlngamma", "i0", "det_h"]:
            self.plot(prop=thermo_prop, units=energy_units, show=False)

        # plot polynomial fits to activity coefficient derivatives if polynomial integration is performed
        if self.kb.gamma_integration_type == "polynomial":
            for thermo_prop in ["lngamma_fits", "dlngamma_fits"]:
                self.plot(prop=thermo_prop, units=energy_units, show=False)

        # for binary systems plot mixing and excess energy contributions
        if self.kb.n_comp == BINARY_SYSTEM:
            for thermo_prop in ["mixing", "excess"]:
                self.plot(prop=thermo_prop, units=energy_units, show=False)

        # for ternary system plot individual energy contributions on separate figure
        elif self.kb.n_comp == TERNAY_SYSTEM:
            for thermo_prop in ["ge", "gm", "hmix", "se"]:
                self.plot(prop=thermo_prop, units=energy_units, show=False)

        else:
            print(f"WARNING: plotter does not support more than 3 components. ({self.kb.n_comp} components detected.)")
