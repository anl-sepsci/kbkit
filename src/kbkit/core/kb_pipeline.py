"""High-level orchestration layer for running thermodynamic analysis workflows."""

from dataclasses import fields
from functools import cached_property

import numpy as np
from numpy.typing import NDArray

from kbkit.analysis.kb_thermo import KBThermo
from kbkit.analysis.system_state import SystemState
from kbkit.calculators.kbi_calculator import KBICalculator
from kbkit.core.system_loader import SystemLoader
from kbkit.schema.thermo_property import ThermoProperty
from kbkit.schema.thermo_state import ThermoState


class KBPipeline:
    """
    A pipeline for performing Kirkwood-Buff analysis of molecular simulations.

    Parameters
    ----------
    pure_path : str
        The path where pure component systems are located. Defaults to a 'pure_components' directory next to the base path if empty string.
    pure_systems: list[str]
        System names for pure component directories.
    base_path : str
        The base path where the systems are located. Defaults to the current working directory if empty string.
    base_systems : list, optional
        A list of base systems to include. If not provided, it will automatically detect systems in the base path.
    rdf_dir : str, optional
        The directory where RDF files are located within each system directory. If empty, it will search in the system directory itself. (default: "").
    ensemble : str, optional
        The ensemble type for the systems, e.g., 'npt', 'nvt'. (default: 'npt').
    cations : list, optional
        A list of cation names to consider for salt pairs. (default: []).
    anions : list, optional
        A list of anion names to consider for salt pairs. (default: []).
    start_time : int, optional
        The starting time for analysis, used in temperature and enthalpy calculations. (default: `0`).
    verbose : bool, optional
        If True, enables verbose output during processing. (default: False).
    use_fixed_r : bool, optional
        If True, uses a fixed cutoff radius for KBI calculations. (default: True).
    ignore_convergence_errors: bool, optional
        If True, will ignore the error that RDF is not converged and perform calculations with NaN values for not converged system. (default: False).
    rdf_convergence_threshold: float, optional
        Set the threshold for a converged RDF. (default: `0.005`).
    gamma_integration_type : str, optional
        The type of integration to use for gamma calculations. (default: 'numerical').
    gamma_polynomial_degree : int, optional
        The degree of the polynomial to fit for gamma calculations if using polynomial integration. (default: `5`).

    Attributes
    ----------
    config: SystemConfig
        SystemConfig object for SystemState analysis.
    state: SystemState
        SystemState object for systems as a function of composition at single temperature.
    kbi_calc: KBICalculator
        KBICalculator object for performing KBI calculations.
    thermo: KBThermo
        KBThermo object for computing thermodynamic properties from KBIs.
    results: ThermoState
        ThermoState object containing results from KBThermo and SystemState.
    """

    def __init__(
        self,
        pure_path: str,
        pure_systems: list[str],
        base_path: str,
        base_systems: list[str] | None = None,
        rdf_dir: str = "",
        ensemble: str = "npt",
        cations: list[str] | None = None,
        anions: list[str] | None = None,
        start_time: int = 0,
        verbose: bool = False,
        use_fixed_r: bool = True,
        ignore_convergence_errors: bool = False,
        rdf_convergence_threshold: float = 0.005,
        gamma_integration_type: str = "numerical",
        gamma_polynomial_degree: int = 5,
    ) -> None:
        self.pure_path = pure_path
        self.pure_systems = pure_systems
        self.base_path = base_path
        self.base_systems = base_systems
        self.rdf_dir = rdf_dir
        self.ensemble = ensemble
        self.cations = cations or []
        self.anions = anions or []
        self.start_time = start_time
        self.verbose = verbose
        self.use_fixed_r = use_fixed_r
        self.ignore_convergence_errors = ignore_convergence_errors
        self.rdf_convergence_threshold = rdf_convergence_threshold
        self.gamma_integration_type = gamma_integration_type
        self.gamma_polynomial_degree = gamma_polynomial_degree

        # initialize property attribute
        self.properties: list[ThermoProperty] = []

    def run(self) -> None:
        """
        Executes the full Kirkwood-Buff Integral (KBI) calculation pipeline.

        This method orchestrates the entire process, including:

        1.  Loading system configurations using :class:`~kbkit.core.system_loader.SystemLoader`.
        2.  Building the system state using :class:`~kbkit.analysis.system_state.SystemState`.
        3.  Initializing the KBI calculator using :class:`~kbkit.calculators.kbi_calculator.KBICalculator`.
        4.  Computing the KBI matrix.
        5.  Creating the thermodynamic model using :class:`~kbkit.analysis.kb_thermo.KBThermo`.
        6.  Generating :class:`~kbkit.schema.thermo_property.ThermoProperty` objects.
        7.  Mapping properties into a structured thermodynamic state.

        This is the primary entry point for running the entire KBI-based
        thermodynamic analysis.

        Returns
        -------
        None
            The results of the pipeline are stored in the `results` attribute
            of this object. Use :meth:`results` to access them.

        Notes
        -----
        The pipeline's progress is logged using the logger initialized within
        :class:`~kbkit.core.system_loader.SystemLoader`.
        """
        loader = SystemLoader(verbose=self.verbose)
        self.logger = loader.logger

        self.logger.info("Building SystemConfig...")
        self.config = loader.build_config(
            pure_path=self.pure_path,
            pure_systems=self.pure_systems,
            base_path=self.base_path,
            base_systems=self.base_systems,
            rdf_dir=self.rdf_dir,
            ensemble=self.ensemble,
            cations=self.cations,
            anions=self.anions,
            start_time=self.start_time,
        )

        self.logger.info("Building SystemState...")
        self.state = SystemState(self.config)

        self.logger.info("Initializing KBICalculator")
        self.kbi_calc = KBICalculator(
            state=self.state,
            use_fixed_r=self.use_fixed_r,
            ignore_convergence_errors=self.ignore_convergence_errors,
            rdf_convergence_threshold=self.rdf_convergence_threshold,
        )
        self.logger.info("Calculating KBIs")
        kbi_matrix = self.kbi_calc.run()

        self.logger.info("Creating KBThermo...")
        self.thermo = KBThermo(
            state=self.state,
            kbi_matrix=kbi_matrix,
            gamma_integration_type=self.gamma_integration_type,
            gamma_polynomial_degree=self.gamma_polynomial_degree,
        )

        self.logger.info("Generating ThermoProperty objects...")
        self.properties = self._compute_properties()

        self.logger.info("Mapping ThermoProperty obejcts into ThermoState...")
        self._results = self._build_thermo_state(self.properties)

        self.logger.info("Pipeline sucessfully built!")

    @cached_property
    def results(self) -> ThermoState:
        """ThermoState object containing all computed thermodynamic properties."""
        if not hasattr(self, "_results"):
            self.run()  # no attribute detected, run the pipeline
        return self._results

    def _compute_properties(self) -> list[ThermoProperty]:
        """Compute ThermoProperties for all attributes of interest."""
        properties = self.thermo.computed_properties()
        properties.append(ThermoProperty(name="molecules", value=self.state.unique_molecules, units=""))
        properties.append(ThermoProperty(name="mol_fr", value=self.state.mol_fr, units=""))
        properties.append(ThermoProperty(name="temperature", value=self.state.temperature(units="K"), units="K"))
        properties.append(ThermoProperty(name="volume", value=self.state.volume(units="nm^3"), units="nm^3"))
        properties.append(
            ThermoProperty(
                name="molar_volume", value=self.state.molar_volume(units="nm^3/molecule"), units="nm^3/molecule"
            )
        )
        properties.append(ThermoProperty(name="n_electrons", value=self.state.n_electrons, units="electron/molecule"))
        properties.append(ThermoProperty(name="h_mix", value=self.state.h_mix(units="kJ/mol"), units="kJ/mol"))
        properties.append(
            ThermoProperty(name="volume_bar", value=self.state.volume_bar(units="nm^3/molecule"), units="nm^3")
        )
        properties.append(
            ThermoProperty(
                name="molecule_rho", value=self.state.molecule_rho(units="molecule/nm^3"), units="molecule/nm^3"
            )
        )
        properties.append(ThermoProperty(name="molecule_counts", value=self.state.molecule_counts, units="molecule"))
        return properties

    def _build_thermo_state(self, props: list[ThermoProperty]) -> ThermoState:
        """Build a ThermoState object for easy property access."""
        prop_map = {p.name: p for p in props}
        state_kwargs = {}
        for field in fields(ThermoState):
            if field.name not in prop_map:
                raise ValueError(f"Missing ThermoProperty for '{field.name}'.")
            state_kwargs[field.name] = prop_map[field.name]
        return ThermoState(**state_kwargs)

    def convert_units(self, name: str, target_units: str) -> NDArray[np.float64]:
        """Get thermodynamic property in desired units.

        Parameters
        ----------
        name: str
            Property to convert units for.
        target_units: str
            Desired units of the property.

        Returns
        -------
        np.ndarray
            Property in converted units.
        """
        meta = self.results.get(name)

        value = meta.value
        units = meta.units
        if len(units) == 0:
            raise ValueError("This is a unitlesss property!")

        try:
            converted = self.state.Q_(value, units).to(target_units)
            return np.asarray(converted.magnitude)
        except Exception as e:
            raise ValueError(f"Could not convert units from {units} to {target_units}") from e

    def available_properties(self) -> list[str]:
        """Get list of available thermodynamic properties from `KBThermo` and `SystemState`."""
        return list(self.results.to_dict().keys())
