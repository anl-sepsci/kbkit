"""High-level orchestration layer for running thermodynamic analysis workflows."""

from dataclasses import fields
from functools import cached_property
from typing import Any, Union

import numpy as np
from numpy.typing import NDArray

from kbkit.analysis.kb_thermo import KBThermo
from kbkit.analysis.kbi_calculator import KBICalculator
from kbkit.analysis.system_state import SystemState
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
    kbi_calculator: KBICalculator
        KBICalculator object for performing KBI calculations.
    thermo: KBThermo
        KBThermo object for computing thermodynamic properties from KBIs.
    thermo_state: ThermoState
        ThermoState object containing results from KBThermo and SystemState.
    results: dict[str, NDArray[np.float64]]
        Dictionary of attributes and their corresponding values in ThermoState object.
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
        5.  Creating the thermodynamic state using :class:`~kbkit.analysis.kb_thermo.KBThermo`.

        This is the primary entry point for running the entire KBI-based
        thermodynamic analysis.

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
        self.kbi_calculator = KBICalculator(
            state=self.state,
            use_fixed_r=self.use_fixed_r,
            ignore_convergence_errors=self.ignore_convergence_errors,
            rdf_convergence_threshold=self.rdf_convergence_threshold,
        )
        self.logger.info("Calculating KBIs")
        kbi_matrix = self.kbi_calculator.run(apply_electrolyte_correction=True)

        self.logger.info("Creating KBThermo...")
        self.thermo = KBThermo(
            state=self.state,
            kbi_matrix=kbi_matrix,
            gamma_integration_type=self.gamma_integration_type,
            gamma_polynomial_degree=self.gamma_polynomial_degree,
        )

        self.logger.info("Pipeline sucessfully built!")

    @cached_property
    def thermo_state(self) -> ThermoState:
        """:class:`~kbkit.schema.thermo_state.ThermoState` object containing all computed thermodynamic properties, in :class:`~kbkit.schema.thermo_property.ThermoProperty` objects."""
        self.logger.info("Generating ThermoProperty objects...")
        self.properties = self._compute_properties()

        self.logger.info("Mapping ThermoProperty obejcts into ThermoState...")
        return self._build_thermo_state(self.properties)

    @cached_property
    def results(self) -> dict[Any, Any]:
        """Dictionary of :class:`~kbkit.schema.thermo_state.ThermoState` with mapped names and values."""
        return self.thermo_state.to_dict()

    def get(self, name: str) -> Union[list[str], NDArray[np.float64]]:
        r"""Extract the property value from :class:`~kbkit.schema.thermo_state.ThermoState`."""
        return self.thermo_state.get(name).value

    def _compute_properties(self) -> list[ThermoProperty]:
        """Compute :class:`~kbkit.schema.thermo_property.ThermoProperty` for all attributes of interest."""
        return self.thermo.computed_properties() + self.state.computed_properties()

    def _build_thermo_state(self, props: list[ThermoProperty]) -> ThermoState:
        """Build a :class:`~kbkit.schema.thermo_state.ThermoState` object for easy property access."""
        prop_map = {p.name: p for p in props}
        state_kwargs = {}
        for field in fields(ThermoState):
            if field.name not in prop_map:
                raise ValueError(f"Missing ThermoProperty for '{field.name}'.")
            state_kwargs[field.name] = prop_map[field.name]
        return ThermoState(**state_kwargs)

    def convert_units(self, name: str, units: str) -> NDArray[np.float64]:
        """Get thermodynamic property in desired units.

        Parameters
        ----------
        name: str
            Property to convert units for.
        units: str
            Desired units of the property.

        Returns
        -------
        np.ndarray
            Property in converted units.
        """
        meta = self.thermo_state.get(name)

        value = meta.value
        initial_units = meta.units
        if len(initial_units) == 0:
            raise ValueError("This is a unitlesss property!")
        elif isinstance(value, dict):
            raise TypeError("Could not convert values from type dict. Values must be list or np.ndarray.")

        try:
            converted = self.state.Q_(value, initial_units).to(units)
            return np.asarray(converted.magnitude)
        except Exception as e:
            raise ValueError(f"Could not convert units from {units} to {units}") from e

    def available_properties(self) -> list[str]:
        """Get list of available thermodynamic properties from `KBThermo` and `SystemState`."""
        return list(self.thermo_state.to_dict().keys())
