"""Plotting support for Kirkwood-Buff Analysis."""

import os
import warnings
from itertools import combinations_with_replacement

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator
from numpy.typing import NDArray

from kbkit.config.mplstyle import load_mplstyle
from kbkit.schema.plot_spec import PlotSpec
from kbkit.utils.format import format_unit_str

load_mplstyle()
warnings.filterwarnings("ignore")

BINARY_SYSTEM = 2
TERNARY_SYSTEM = 3


class Plotter:
    r"""
    A class for plotting results from Kirkwood-Buff analysis (:class:`kbkit.kb.kb_thermo.KBThermo`).

    Parameters
    ----------
    pipeline: KBPipeline
        Instance of KBThermo.
    molecule_map: dict[str, str], optional.
        Dictionary of molecule ID in topology mapped to molecule names for figure labeling. Defaults to using molecule names in topology.
    x_mol: str, optional
        Molecule to use for labeling x-axis in figures for binary systems. Defaults to first element in molecule list.
    img_type: str, optional
        Type of image file. Defaults to PDF.
    save_dir: str, optional
        Location for saving figures. Defaults to ``base_path`` in Pipeline.
    """

    def __init__(
        self,
        pipeline,
        molecule_map: dict[str, str],
        x_mol: str = "",
        img_type: str = ".pdf",
        save_dir: str | None = None,
    ) -> None:
        # data pipeline containing results from analysis
        self.pipe = pipeline

        # create a dict of properties to plot to eliminate nested structures
        self.property_map = self.pipe.results

        self.x_mol = x_mol
        self.molecule_map = molecule_map
        self.img_type = img_type.lower() if img_type.startswith('.') else f".{img_type.lower()}"
        if self.img_type not in (".png", ".pdf", ".jpg"):
            raise ValueError(f"Invalid image type ({self.img_type}) detected! Options include: .pdf, .jpg, .png")

        self.base_path = save_dir or self.pipe.config.base_path
        self._setup_folders(str(self.base_path))


    def _setup_folders(self, base_path: str) -> None:
        # create folders for figures if they don't exist
        self.thermo_dir = os.path.join(base_path, "kb_analysis")
        self.sys_dir = os.path.join(self.thermo_dir, "system_figures")
        for path in (self.thermo_dir, self.sys_dir):
            if not os.path.exists(path):
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
            mapped = {mol: mol for mol in self.pipe.state.unique_molecules}

        # check that all molecules are defined in map
        found_mask = np.array([mol not in mapped for mol in self.pipe.state.unique_molecules])
        if any(found_mask):
            missing_mols = np.array(list(mapped.keys()))[found_mask]
            raise ValueError(
                f"Molecules missing from molecule_map: {', '.join(missing_mols)}. "
                f"Available molecules: {', '.join(self.pipe.state.unique_molecules)}"
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
            self._x_mol = self.pipe.state.unique_molecules[0]

        # check if mol is in unique molecules
        if mol not in self.pipe.state.unique_molecules:
            raise ValueError(
                f"Molecule {mol} not in available molecules: {', '.join(self.pipe.state.unique_molecules)}"
            )

        self._x_mol = mol

    @property
    def unique_names(self) -> list[str]:
        """list: Names of molecules to use in figure labels."""
        return [self.molecule_map[mol] for mol in self.pipe.state.unique_molecules]

    @property
    def _x_idx(self) -> int:
        # get index of x_mol in kb.unique_molecules
        return self.pipe.state.unique_molecules.index(self.x_mol)

    def _get_rdf_colors(self, cmap: str = "jet") -> dict[str, dict[str, tuple[float, ...]]]:
        # create a colormap mapping pairs of molecules with a color
        if "_color_dict" not in self.__dict__:
            # Collect all unique unordered molecule pairs across systems
            all_pairs: set[tuple[str, ...]] = set()
            for meta in self.pipe.config.registry:
                mol_ids = meta.props.topology.molecules
                pairs = combinations_with_replacement(mol_ids, 2)
                all_pairs.update(tuple(sorted(p)) for p in pairs)

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

    def _convert_kbi(self, value, units="cm^3/mol") -> NDArray[np.float64]:
        """Conver KBI metadata value to desired units."""
        converted = self.pipe.state.Q_(value, "nm^3/molecule").to(units)
        return np.asarray(converted.magnitude)

    def plot_system_kbi_analysis(
        self,
        system: str,
        units: str = "",
        alpha: float = 0.6,
        cmap: str = "jet",
        show: bool = False,
    ):
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
        kbi_meta = self.pipe.kbi_calculator.kbi_metadata.get(system)
        if kbi_meta is None:
            raise ValueError(f"No KBIs for sytem: {system}")
        units = "cm^3/mol" if units == "" else units

        fig, ax = plt.subplots(1, 3, figsize=(12, 3.6))
        for meta in kbi_meta:
            mol_i, mol_j = meta.mols
            color = color_dict.get(mol_i, {})[mol_j]

            rkbi = self._convert_kbi(meta.rkbi)
            scaled_rkbi = self._convert_kbi(meta.scaled_rkbi)
            scaled_rkbi_est = self._convert_kbi(meta.scaled_rkbi_est)

            label = f"{self.molecule_map[mol_i]}-{self.molecule_map[mol_j]}"
            ax[0].plot(meta.r, meta.g, lw=2.5, c=color, alpha=alpha, label=label)
            ax[1].plot(
                meta.r,
                rkbi,
                lw=2.5,
                c=color,
                alpha=alpha,
            )
            ax[2].plot(
                meta.r,
                scaled_rkbi,
                lw=2.5,
                c=color,
                alpha=alpha,
            )
            ax[2].plot(meta.r_fit, scaled_rkbi_est, ls="--", lw=3, c="k")

        ax[0].set_xlabel(r"$r$ [$nm$]")
        ax[1].set_xlabel(r"$R$ [$nm$]")
        ax[2].set_xlabel(r"$R$ [$nm$]")
        ax[0].set_ylabel("g(r)")
        ax[1].set_ylabel(f"G$_{{ij}}^R$ [{format_unit_str(units)}]")
        ax[2].set_ylabel(f"$R$ $G_{{ij}}^R$ [$nm$ {format_unit_str(units)}]")
        ax[0].legend(loc="best", ncol=1, fontsize="small", frameon=True)
        plt.savefig(os.path.join(self.sys_dir, f"{system}_rdfs_kbis.{self.img_type}"))
        if show:
            plt.show()
        else:
            plt.close()
        return fig, ax

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
        for system, _kbi_meta in self.pipe.kbi_calculator.kbi_metadata.items():
            self.plot_system_kbi_analysis(system, units=units, show=show)

    def plot_kbis(self, units: str = "cm^3/mol", cmap: str = "jet", show: bool = False):
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
        fig, ax = plt.subplots(1, 1, figsize=(5, 4.5))
        legend_info = {}
        # get list of SystemMeta for each system
        for s, system_meta in enumerate(self.pipe.config.registry):
            # get kbi meta for system
            meta_list = self.pipe.kbi_calculator.kbi_metadata.get(system_meta.name, [])
            if len(meta_list) < 1:
                continue
            # then iterate through all metas
            for meta in meta_list:
                mol_i, mol_j = meta.mols
                color = color_dict.get(mol_i, {})[mol_j]
                kbi = self._convert_kbi(meta.kbi_limit)
                label = f"{self.molecule_map[mol_i]}-{self.molecule_map[mol_j]}"
                line = ax.scatter(self.pipe.state.mol_fr[s, self._x_idx], kbi, c=color, marker="o", lw=1.8, label=label)
                if meta.mols not in legend_info:
                    legend_info[label] = line
        lines = list(legend_info.values())
        labels = list(legend_info.keys())
        ax.legend(lines, labels, loc="best", ncol=1, fontsize="small", frameon=True)
        ax.set_xlim(-0.05, 1.05)
        ax.set_xticks(ticks=np.arange(0, 1.1, 0.1))
        ax.set_xlabel(f"x$_{{{self.molecule_map[self.x_mol]}}}$")
        ax.set_ylabel(rf"G$_{{ij}}^{{\infty}}$ [{format_unit_str(units)}]")
        plt.savefig(self.thermo_dir + f"/composition_kbi_{units.replace('^', '').replace('/', '_')}.{self.img_type}")
        if show:
            plt.show()
        else:
            plt.close()
        return fig, ax

    def _get_plot_spec(self, prop: str) -> PlotSpec:
        """
        Generate a PlotSpec object for a given thermodynamic property.

        Parameters
        ----------
        prop : str
            The name of the property to plot (e.g., 'lngammas', 'mixing', 'i0').

        Returns
        -------
        PlotSpec
            A fully populated PlotSpec object containing data and metadata for rendering.
        """
        if prop in ["lngammas", "dlngammas", "lngammas_fits", "dlngammas_fits"]:
            fits = None
            xfit = None
            if prop.endswith("fits"):
                if self.pipe.thermo.gamma_integration_type == "polynomial":
                    xfit = np.arange(0, 1.01, 0.01, dtype=np.float64)
                    fit_fns = (
                        self.pipe.thermo._lngamma_fn_dict
                        if prop == "lngammas_fits"
                        else self.pipe.thermo._dlngamma_fn_dict
                    )
                    fits = {mol: fn(xfit) for mol, fn in fit_fns.items()}
                elif self.pipe.thermo.gamma_integration_type == "numerical":
                    dlng = self.pipe.thermo.dlngammas_dxs.value
                    xfit = np.array(self.pipe.state.mol_fr[:, self._x_idx], dtype=np.float64)
                    fits = {}
                    for j, molj in enumerate(self.pipe.state.unique_molecules):
                        xj = self.pipe.state.mol_fr[:, j]
                        dlng[np.where(xj == 1)[0][0], j] = 0
                        fits[molj] = dlng[:, j]

            return PlotSpec(
                x_data=self.pipe.state.mol_fr,
                y_data=self.property_map["lngammas"] if "dln" not in prop else self.property_map["dlngammas_dxs"],
                ylabel=r"$\ln \gamma_{i}$" if "dln" not in prop else r"$\partial \ln(\gamma_{i})$ / $\partial x_{i}$",
                filename=f"{prop}.{self.img_type}",
                xfit=xfit,
                fits=fits,
                multi=False,
            )

        elif prop in ["mixing", "excess"]:
            y_series_list = [
                (self.pipe.state.mixture_enthalpy(), "violet", "s", r"$\Delta H_{mix}$"),
                (-self.pipe.state.temperature() * self.property_map["s_ex"], "limegreen", "o", r"$-TS^E$"),
            ]
            if prop == "mixing":
                y_series_list += [
                    (self.property_map["g_id"], "darkorange", "<", r"$G^{id}$"),
                    (self.property_map["g_mix"], "mediumblue", "^", r"$\Delta G_{mix}$"),
                ]
            else:
                y_series_list.append((self.property_map["g_ex"], "mediumblue", "^", r"$G^E$"))

            return PlotSpec(
                x_data=self.pipe.state.mol_fr[:, self._x_idx],
                y_series=y_series_list,
                ylabel=rf"Contributions to $\Delta G_{{mix}}$ [{format_unit_str('kJ/mol')}]"
                if prop == "mixing"
                else f"Excess Properties [{format_unit_str('kJ/mol')}]",
                filename=f"gibbs_{'mixing' if prop == 'mixing' else 'excess'}_contributions.{self.img_type}",
                multi=True,
            )

        elif prop in ["i0", "hessian_determinant"]:
            return PlotSpec(
                x_data=self.pipe.state.mol_fr[:, self._x_idx],
                y_data=self.property_map["i0"] if prop == "i0" else self.property_map["hessian_determinant"],
                ylabel=f"I$_0$ [{format_unit_str('cm^{-1}')}]"
                if prop == "i0"
                else f"$|H_{{ij}}|$ [{format_unit_str('kJ/mol')}]",
                filename=f"saxs_{'I0' if prop == 'i0' else 'hessian_determinant'}.{self.img_type}",
                multi=False,
            )

        else:
            raise ValueError(f"Unknown property: '{prop}'")

    def _render_binary_plot(
        self,
        spec: PlotSpec,
        ylim: tuple[float, float] = (0.0, 0.0),
        show: bool = True,
        cmap: str = "jet",
        marker: str = "o",
    ):
        """
        Render a binary system plot based on the provided PlotSpec.

        Parameters
        ----------
        spec : PlotSpec
            The plot specification containing data and metadata.
        ylim : tuple[float, float], optional
            Manual y-axis limits. If (0.0, 0.0), limits are auto-scaled.
        show : bool, optional
            Whether to display the plot interactively.
        cmap : str, optional
            Colormap used for multi-component plots.
        marker : str, optional
            Marker style for scatter plots.
        """
        fig, ax = plt.subplots(figsize=(5, 4.5))

        if spec.multi and spec.y_series:
            for y_data, color, mk, label in spec.y_series:
                ax.scatter(spec.x_data, y_data, c=color, marker=mk, label=label)
            ax.legend(loc="best", ncol=1, fontsize="small", frameon=True)

        elif spec.y_data is not None:
            if spec.y_data.ndim == 1:
                ax.scatter(spec.x_data, spec.y_data, c="mediumblue", marker=marker)
            else:
                colors = plt.cm.get_cmap(cmap)(np.linspace(0, 1, self.pipe.state.n_comp))
                for i, mol in enumerate(self.pipe.state.unique_molecules):
                    xi = spec.x_data[:, self._x_idx] if self.pipe.state.n_comp == BINARY_SYSTEM else spec.x_data[:, i]
                    yi = spec.y_data[:, i]
                    ax.scatter(xi, yi, c=[colors[i]], marker=marker, label=self.molecule_map[mol])

                    if spec.fits is not None and spec.xfit is not None:
                        ax.plot(spec.xfit, spec.fits[mol], c=colors[i], lw=2)

                ax.legend(loc="best", ncol=1, fontsize="small", frameon=True)

        ax.set_xlabel(
            f"x$_{{{self.molecule_map[self.x_mol]}}}$" if self.pipe.state.n_comp == BINARY_SYSTEM else "x$_i$"
        )
        ax.set_ylabel(spec.ylabel)
        ax.set_xlim(-0.05, 1.05)
        ax.set_xticks(np.arange(0, 1.1, 0.1))

        if any(ylim) != 0:
            ax.set_ylim(*ylim)
        elif spec.y_data is not None:
            y_finite = spec.y_data[np.isfinite(spec.y_data)]
            if len(y_finite) > 0:
                y_max, y_min = np.nanmax(y_finite), np.nanmin(y_finite)
                pad = 0.1 * (y_max - y_min) if y_max != y_min else 0.05
                y_lb = 0 if spec.y_data.ndim == 1 else -0.05
                ax.set_ylim(min([y_lb, y_min - pad]), max([0.05, y_max + pad]))

        plt.savefig(os.path.join(self.thermo_dir, spec.filename))
        if show:
            plt.show()
        else:
            plt.close()
        return fig, ax

    def _render_ternary_plot(
        self,
        property_name: str,
        cmap: str = "jet",
        show: bool = False,
    ):
        """
        Render a ternary system plot based on the provided PlotSpec.

        Parameters
        ----------
        property_name : str
            Single property to plot on ternary figure.
        cmap : str, optional
            Colormap used for multi-component plots.
        show : bool, optional
            Whether to display the plot interactively.
        """
        arr = np.asarray(self.property_map[property_name])
        xtext, ytext, ztext = self.unique_names
        a, b, c = (
            self.pipe.state.mol_fr[:, 0],
            self.pipe.state.mol_fr[:, 1],
            self.pipe.state.mol_fr[:, 2],
        )

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

        plt.savefig(os.path.join(self.thermo_dir, f"ternary_{property_name}.{self.img_type}"))
        if show:
            plt.show()
        else:
            plt.close()
        return fig, ax

    def available_properties(self) -> list[str]:
        r"""Print out the available properties to plot with :meth:`plot_property`."""
        properties = [
            "kbi",
            "lngammas",
            "dlngammas",
            "lngammas_fits",
            "dlngammas_fits",
            "excess",
            "mixing",
            "g_mix",
            "g_ex",
            "h_mix",
            "s_ex",
            "i0",
            "hessian_determinant",
        ]
        return properties

    def plot(
        self,
        prop: str,
        system: str = "",
        cmap: str = "jet",
        marker: str = "o",
        ylim: tuple[float, float] = (0.0, 0.0),
        show: bool = True,
    ):
        r"""
        Master plot function. Handles property selection, data prep, and plotting.

        Automatically determines the correct type of plot.

        Parameters
        ----------
        prop: str
            Which property to plot? Options include:
                - '`rdf`': (System required) Radial distribution function for all pairwise interactions.
                - '`kbi`': KBI as a function of composition
                - '`lngammas`': Activity coefficients for each molecule.
                - '`dlngammas`': Derivative of activity coefficients with respect to mol fraction of each molecule.
                - '`lngammas_fits`': Activity coefficient function.
                - '`dlngammas_fits`': Fit of polynomial function to activity coefficient derivative.
                - '`excess`': (Binary systems only) Excess thermodynamic properties as a function of composition.
                - '`mixing`': (Binary systems only) Mixing thermodynamic properties as a function of composition.
                - '`g_mix`': Gibbs free energy of mixing.
                - '`g_ex`': Gibbs excess free energy.
                - '`h_mix`': Mixing enthalpy.
                - '`s_ex`': Excess entropy.
                - '`i0`': SAXS intensity as q :math:`\rightarrow` 0.
                - '`hessian_determinant`': Determinant of Hessian.
        system: str, optional
            System to plot for RDF/KBI analysis of specific system (default None).
        cmap: str, otpional
            Matplotlib colormap (default 'jet').
        marker: str, optional
            Marker shape for scatterplots (default 'o').
        ylim: list, optional
            For specific system, y-axis limits of zoomed in RDF; otherwise: y-axis in binary and activity coefficient plots.
        show: bool, optional
            Display figure (default True).
        """
        # get prop key
        prop_key = ""
        for p in self.available_properties():
            if (prop.lower().startswith(p)) or (p.startswith(prop.lower())):
                prop_key = p 
                break
        if len(prop_key) == 0:
            raise ValueError(f"Property {prop_key} not valid. Options include: {self.available_properties()}")

        if system:
            # plot system kbis
            if prop_key == "kbi":
                return self.plot_system_kbi_analysis(system=system, units="cm^3/mol", cmap=cmap, show=show)

            else:
                print("WARNING: Invalid plot option specified! System specific include rdf and kbi.")

        elif prop_key == "kbi":
            return self.plot_kbis(units="cm^3/mol", cmap=cmap, show=show)

        elif self.pipe.state.n_comp == BINARY_SYSTEM or prop_key in {
            "lngammas",
            "dlngammas",
            "lngammas_fits",
            "dlngammas_fits",
        }:
            spec = self._get_plot_spec(prop_key)
            return self._render_binary_plot(spec, marker=marker, ylim=ylim, cmap=cmap, show=show)

        elif self.pipe.state.n_comp == TERNARY_SYSTEM and prop_key in {"g_mix", "g_ex", "h_mix", "s_ex", "i0", "hessian_determinant"}:
            return self._render_ternary_plot(property_name=prop_key, cmap=cmap, show=show)

        elif self.pipe.state.n_comp > TERNARY_SYSTEM:
            print(
                f"WARNING: plotter does not support {prop_key} for more than 3 components. ({self.pipe.state.n_comp} components detected.)"
            )

    def make_figures(self) -> None:
        r"""Create all figures for Kirkwood-Buff analysis."""
        # create figure for rdf/kbi analysis
        self.plot_rdf_kbis(show=False)
        # plot KBI as a function of composition
        self.plot_kbis(units="cm^3/mol", show=False)

        # create figures for properties independent of component number
        for thermo_prop in ["lngammas", "dlngammas", "i0", "hessian_determinant"]:
            self.plot(prop=thermo_prop, show=False)

        # plot polynomial fits to activity coefficient derivatives if polynomial integration is performed
        if self.pipe.thermo.gamma_integration_type == "polynomial":
            for thermo_prop in ["lngammas_fits", "dlngammas_fits"]:
                self.plot(prop=thermo_prop, show=False)

        # for binary systems plot mixing and excess energy contributions
        if self.pipe.state.n_comp == BINARY_SYSTEM:
            for thermo_prop in ["mixing", "excess"]:
                self.plot(prop=thermo_prop, show=False)

        # for ternary system plot individual energy contributions on separate figure
        elif self.pipe.state.n_comp == TERNARY_SYSTEM:
            for thermo_prop in ["g_ex", "g_mix", "h_mix", "s_ex"]:
                self.plot(prop=thermo_prop, show=False)

        else:
            print(
                f"WARNING: plotter does not support more than 3 components. ({self.pipe.state.n_comp} components detected.)"
            )
