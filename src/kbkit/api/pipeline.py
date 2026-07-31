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
    start : int, optional
        Starting point for when data should be used. GROMACS: time (ps), LAMMPS: timestep (i.e., for 1 fs steps, start=1000--start at 1 ps).
    include_mode: str, optional
        Optional string to filter files (.edr, .gro, .top) if multiple are found of a given type.
    weight_type: str, optional
        Type of weight function for finite-volume corrections. Options: ('none','u0','u1','u2','geometric')
    r_position: tuple[float,float], optional
        Range of `r` values to include for linear fit. Values outside this range will be excluded.
    maximize_r2: bool, optional
        Search through valid positions to find the `r` values that maximize :math:`R^2` with a `r` range greater than 1.0 nm, and that include the last 10% of data (this ensures some range in the beginning is not selected).
    r2_threshold: float, optional
        Set a :math:`R^2` threshold to satisfy `KBIConvergence`.
    min_r_range: float, optional
        Minimum range of `r` values to be required for `KBIConvergence`.
    raise_on_convergence_error : bool, optional
        Only applied for ``weight_type='geometric'``, for linear extrapolation to thermodynamic limit.
        If True, raises KBIConvergenceError when convergence checks fail.
        If False, returns NaN and prints warning. Default: True.
    force: bool, optional
        Only applied for ``weight_type='geometric'``. If KBIConvergenceError is raised, prints warning and returns KBI for ``weight_type='u2'``.
    activity_integration_type: str, optional
        Method for performing integration of activity coefficient derivatives.
    activity_polynomial_degree: int, optional
        Polynomial degree for fitting activity coefficient derivatives, if ``activity_integration_type`` is `polynomial`.
    reference_states: dict[str, int], optional
        For setting asymmetrical reference states for solutes, i.e., for :math:`x_i = 0`. Mapped to molecule name in topology. Defaults to pure component (if not specified).
    molecule_map: dict[str, str], optional
        Dictionary mapping molecule names to desired molecule labels in figures.
    charges: dict[str, int], optional
        Optional charge dictionary for ions. Mapping residue names to their corresponding charge.
    """

    def __init__(
        self,
        base_path: str | None = None,
        base_systems: list[str] | None = None,
        pure_path: str | None = None,
        pure_systems: list[str] | None = None,
        rdf_dir: str = "",
        start: int = 10000,
        include_mode: str = "npt",
        weight_type: Literal["none", "u0", "u1", "u2", "geometric"] = "geometric",
        r_positions: tuple[float, float] | None = None,
        maximize_r2: bool = True,
        min_r_range: float = 1.0,
        r2_threshold: float = 0.99,
        errors: Literal["raise", "warn", "ignore"] = "warn",
        force: bool = False,
        activity_integration_type: Literal["numerical", "polynomial"] = "numerical",
        activity_polynomial_degree: int = 5,
        reference_states: dict[str, int] | None = None,
        molecule_map: dict[str, str] | None = None,
        charges: dict[str, int] | None = None,
    ) -> None:
        self.base_path = base_path
        self.base_systems = base_systems
        self.pure_path = pure_path
        self.pure_systems = pure_systems
        self.rdf_dir = rdf_dir
        self.start = start
        self.include_mode = include_mode
        self.weight_type = weight_type
        self.r_positions = r_positions
        self.maximize_r2 = maximize_r2
        self.min_r_range = min_r_range
        self.r2_threshold = r2_threshold
        self.errors = errors
        self.force = force
        self.activity_integration_type = activity_integration_type
        self.activity_polynomial_degree = int(activity_polynomial_degree)
        self.reference_states = reference_states
        self.molecule_map = molecule_map
        self.charges = charges

    @cached_property
    def systems(self) -> SystemCollection:
        """SystemCollection: Configuration for a thermodynamic state, includes topology and energy properties."""
        return SystemCollection.load(
            base_path=self.base_path,
            base_systems=self.base_systems,
            pure_path=self.pure_path,
            pure_systems=self.pure_systems,
            rdf_dir=self.rdf_dir,
            start=self.start,
            include_mode=self.include_mode,
            charges=self.charges,
        )

    @cached_property
    def calculator(self) -> KBICalculator:
        """KBICalculator: Calculator for KBIs as a function of composition."""
        return KBICalculator(
            systems=self.systems,
            weight_type=self.weight_type,
            r_positions=self.r_positions,
            maximize_r2=self.maximize_r2,
            min_r_range=self.min_r_range,
            r2_threshold=self.r2_threshold,
            force=self.force,
            errors=self.errors,
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
            reference_states=self.reference_states,
        )

    @cached_property
    def results(self) -> dict[str, PropertyResult]:
        """dict[str, PropertyResults]: Property result objects for KBI-derived and properties measured directly from MD simultions."""
        res_dict = {}
        res_dict.update(self.systems.results)
        res_dict.update(self.thermo.results)
        return res_dict

    def timeseries_plotter(self, system: str, start: int = 0) -> "TimeseriesPlotter":
        """Plotter for visualizing property timeseries.

        Returns
        -------
        TimeseriesPlotter
        """
        return self.systems.timeseries_plotter(system, start)

    @property
    def kbi_plotter(self) -> "KBIAnalysisPlotter":
        """KBIAnalysisPlotter: Plotter for visualizing KBI convergence and extrapolation."""
        return self.calculator.kbi_plotter()

    @property
    def thermo_plotter(self) -> "ThermoPlotter":
        """ThermoPlotter: Plotter for visualizing KBI and derived thermodynamic properties as a function of composition."""
        return self.thermo.plotter(molecule_map=self.molecule_map)

    def make_figures(self, xmol: str | None = None, cmap: str = "jet", savepath: str | None = None, **kwargs) -> None:
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
        self.kbi_plotter.plot_rkbis(savepath=str(kbi_savepath))

        # plot thermo figures
        self.thermo_plotter.make_figures(xmol=xmol, cmap=cmap, savepath=str(fig_path), **kwargs)
