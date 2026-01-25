r"""
Pipeline module for automated Kirkwood-Buff (KB) thermodynamic analysis.

This module provides a high-level workflow that coordinates all major `KBKit` components—-`SystemCollection`, `KBThermo`, `SystemProperties`, and `KBICalculator`—-to compute thermodynamic properties across a composition series directly from simulation outputs.

The pipeline expects a directory structure containing simulation results for each composition point.
At each of these composition points, the pipeline:

1. Builds a set of systems at constant temperature using: :class:`~kbkit.systems.collection.SystemCollection`.
2. :class:`~kbkit.systems.collection.SystemCollection` computes topology and energy properties as a function of mole fractions.
3. Computes pairwise Kirkwood-Buff integrals using :class:`~kbkit.kbi.calculator.KBICalculator`.
4. Computes KBI-derived thermodynamic properties and structure factors using :class:`~kbkit.kbi.thermodynamics.KBThermo`.

Composition-Grid Requirements
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Different thermodynamic quantities place different demands on the composition grid.
In KBKit, these fall into two distinct categories:

**1. Quantities that *require* an evenly spaced composition grid**
(first derivatives of the Gibbs free energy)

These properties depend on **integration** of derivatives of the Gibbs free energy and therefore require a composition series that spans the **entire mole-fraction domain** with **approximately uniform spacing**.
This ensures stable integration and physically meaningful results.

Properties in this category include:
    - activity coefficients (γᵢ),
    - excess Gibbs-energy-related quantities that rely on integrating activity coefficients (i.e., decoupling enthalpic and entropic contributions).

A well-distributed composition grid is essential for these quantities.

**2. Quantities that do *not* require evenly spaced compositions**
(second derivatives of the Gibbs free energy)

These properties are computed **directly from the KB integrals** and do *not* depend on the spacing or coverage of the composition grid.
Uneven, sparse, or clustered composition points are acceptable as long as the KBIs themselves are well converged.

Properties in this category include:
    - stability metrics (Hessian of :math:`\Delta G_{mix}`),
    - structure factors,
    - any quantity derived directly from the KBI matrix that doesn't rely on activity coefficients or excess Gibbs energy contributions.

Requirements for automated thermodynamic analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
- A composition series with one simulation directory per composition point.
- Each directory must contain:
    * a structure file (.gro),
    * an energy file (.edr),
    * a subdirectory containing RDF files (.xvg) for each pairwise interaction.
- Pure-component simulations are required for:
    * mixing enthalpy,
    * excess molar volume,
    * decoupling enthalpic and entropic contributions.


The pipeline stores all intermediate objects for reproducibility and supports high-throughput mixture sweeps and automated KB analysis.
"""

from functools import cached_property
from typing import TYPE_CHECKING, Literal

from kbkit.kbi.calculator import KBICalculator
from kbkit.kbi.thermodynamics import KBThermo
from kbkit.schema.property_result import PropertyResult
from kbkit.systems.collection import SystemCollection
from kbkit.utils.validation import validate_path

if TYPE_CHECKING:
    from kbkit.visualization import KBIAnalysisPlotter, ThermoPlotter, TimeseriesPlotter


class Pipeline:
    """
    High-level workflow manager for running automated KBKit thermodynamic analysis across a composition series.

    Pipeline loads simulation data, constructs `SystemCollection` objects, computes KB integrals, and evaluates thermodynamic properties using `KBThermo`.
    It provides a reproducible interface for mixture sweeps and KB-based analysis.

    Parameters
    ----------
    pure_path : str or Path
        Path to pure component directory.
    pure_systems: list[str]
        List of pure systems to include.
    base_path : str or Path
        Path to base system directory.
    base_systems : list[str], optional
        Explicit list of system names to include.
    rdf_dir: str, optional
        Explicit directory name that contains rdf files.
    start_time : int, optional
        Start time for time-averaged properties.
    include_mode: str, optional
        Optional string to filter files (.edr, .gro, .top) if multiple are found of a given type.
    ignore_convergence_errors : bool, optional
        If True, ingnores convergence errors and forces KBI calculations to skip entire systems with non-converged RDFs.
    rdf_convergence_thresholds: tuple[float, float], optional
        Thresholds for convergence requirements of RDF tail.
    rdf_tail_length: float, optional
        Length of RDF tail (nm) to use for convergence evaluation & KBI corrections. If this is set, no iteration to find maximum length for RDF convergence will be performed.
    kbi_correct_rdf_convergence: bool, optional
        Whether to correct RDF for excess/depletion, i.e., Ganguly correction.
    kbi_apply_damping: bool, optional
        Whether to apply damping function to correlation function, i.e., Kruger correction.
    kbi_extrapolate_thermodynamic_limit: bool, optional
        Whether to extrapolate KBI value to the thermodynamic limit.
    activity_integration_type: str, optional
        Method for performing integration of activity coefficient derivatives.
    activity_polynomial_degree: int, optional
        Polynomial degree for fitting activity coefficient derivatives, if ``activity_integration_type`` is `polynomial`.
    molecule_map: dict[str, str], optional
        Dictionary mapping molecule names to desired molecule labels in figures.
    """

    def __init__(
        self,
        base_path: str | None = None,
        base_systems: list[str] | None = None,
        pure_path: str | None = None,
        pure_systems: list[str] | None = None,
        rdf_dir: str = "",
        start_time: int = 10000,
        include_mode: str = "npt",
        ignore_convergence_errors: bool = False,
        rdf_convergence_thresholds: tuple = (1e-3, 1e-2),
        rdf_tail_length: float | None = None,
        kbi_correct_rdf_convergence: bool = True,
        kbi_apply_damping: bool = True,
        kbi_extrapolate_thermodynamic_limit: bool = True,
        activity_integration_type: Literal["numerical", "polynomial"] = "numerical",
        activity_polynomial_degree: int = 5,
        molecule_map: dict[str, str] | None = None,
    ) -> None:
        self.base_path = base_path
        self.base_systems = base_systems
        self.pure_path = pure_path
        self.pure_systems = pure_systems
        self.rdf_dir = rdf_dir
        self.start_time = start_time
        self.include_mode = include_mode
        self.ignore_convergence_errors = ignore_convergence_errors
        self.rdf_convergence_thresholds = rdf_convergence_thresholds
        self.rdf_tail_length = rdf_tail_length
        self.kbi_correct_rdf_convergence = kbi_correct_rdf_convergence
        self.kbi_apply_damping = kbi_apply_damping
        self.kbi_extrapolate_thermodynamic_limit = kbi_extrapolate_thermodynamic_limit
        self.activity_integration_type = activity_integration_type
        self.activity_polynomial_degree = int(activity_polynomial_degree)
        self.molecule_map = molecule_map

    def _build_systems(self) -> SystemCollection:
        """Build SystemCollection object."""
        return SystemCollection.load(
            base_path=self.base_path,
            base_systems=self.base_systems,
            pure_path=self.pure_path,
            pure_systems=self.pure_systems,
            rdf_dir=self.rdf_dir,
            start_time=self.start_time,
            include_mode=self.include_mode,
        )

    @cached_property
    def systems(self) -> SystemCollection:
        """SystemCollection: Configuration for a thermodynamic state, includes topology and energy properties."""
        return self._build_systems()

    @cached_property
    def calculator(self) -> KBICalculator:
        """KBICalculator: Calculator for KBIs as a function of composition."""
        return KBICalculator(
            systems=self.systems,
            ignore_convergence_errors=self.ignore_convergence_errors,
            convergence_thresholds=self.rdf_convergence_thresholds,
            tail_length=self.rdf_tail_length,
            correct_rdf_convergence=self.kbi_correct_rdf_convergence,
            apply_damping=self.kbi_apply_damping,
            extrapolate_thermodynamic_limit=self.kbi_extrapolate_thermodynamic_limit,
        )

    @property
    def kbi_res(self) -> PropertyResult:
        """PropertyResult: Compute KBI result object."""
        return self.calculator.kbi(units="cm^3/mol")

    @cached_property
    def thermo(self) -> KBThermo:
        """KBThermo: KBI-derived thermodynamic quantities."""
        return KBThermo(
            systems=self.systems,
            kbi=self.kbi_res,
            activity_integration_type=self.activity_integration_type,
            activity_polynomial_degree=self.activity_polynomial_degree,
        )

    @cached_property
    def results(self) -> dict[str, PropertyResult]:
        """Dictionary of :class:`~kbkit.schema.property_result.PropertyResult` with mapped names and values."""
        res = {}
        # add properties from KBThermo
        res.update({k: v for k, v in self.thermo.results.items() if isinstance(v, PropertyResult)})
        # compute system properties
        for prop, units in self.systems.units.items():
            if "time" in prop.lower():
                continue
            res[f"simulated_{prop.lower().replace('-', '_')}"] = self.systems.simulated_property(
                name=prop, units=units, avg=True
            )
            res[f"ideal_{prop.lower().replace('-', '_')}"] = self.systems.ideal_property(
                name=prop, units=units, avg=True
            )
            res[f"excess_{prop.lower().replace('-', '_')}"] = self.systems.excess_property(
                name=prop, units=units, avg=True
            )
        return res

    def timeseries_plotter(self, system: str, start_time: int = 0) -> "TimeseriesPlotter":
        """TimeseriesPlotter: Plotter for visualizing property timeseries."""
        return self.systems.timeseries_plotter(system, start_time)

    @property
    def kbi_plotter(self) -> "KBIAnalysisPlotter":
        """KBIAnalysisPlotter: Plotter for visualizing KBI convergence and extrapolation."""
        return self.calculator.kbi_plotter(molecule_map=self.molecule_map)

    @property
    def thermo_plotter(self) -> "ThermoPlotter":
        """ThermoPlotter: Plotter for visualizing KBI and derived thermodynamic properties as a function of composition."""
        return self.thermo.plotter(molecule_map=self.molecule_map)

    def make_figures(
        self,
        xmol: str | None = None,
        cmap: str = "jet",
        savepath: str | None = None,
    ) -> None:
        """Make KBI analysis figures for all systems and default figures from ThermoPlotter.

        Parameters
        ----------
        xmol: str, optional
            Molecule name for x-axis.
        cmap: str, optional
            Matplotlib colormap.
        savepath: str, optional
            Parent path for saving figures.
        """
        # get savepath
        if not savepath:
            base_path = self.systems.mixtures[0].path.parent
            fig_path = base_path / "kb_analysis"
        else:
            fig_path = validate_path(savepath)
        fig_path.mkdir(parents=True, exist_ok=True)

        # plot all kbi_analysis
        kbi_savepath = fig_path / "system_figures"
        kbi_savepath.mkdir(parents=True, exist_ok=True)
        self.kbi_plotter.plot_all(units="cm^3/mol", savepath=str(kbi_savepath), show=False)

        # plot thermo figures
        self.thermo_plotter.make_figures(xmol=xmol, cmap=cmap, savepath=str(fig_path))
