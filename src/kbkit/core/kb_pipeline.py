"""High-level orchestration layer for running thermodynamic analysis workflows."""

from dataclasses import fields

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
    ensemble : str, optional
        The ensemble type for the systems, e.g., 'npt', 'nvt'. Defaults to 'npt'.
    cations : list, optional
        A list of cation names to consider for salt pairs. Defaults to an empty list.
    anions : list, optional
        A list of anion names to consider for salt pairs. Defaults to an empty list.
    start_time : int, optional
        The starting time for analysis, used in temperature and enthalpy calculations. Defaults to 0.
    verbose : bool, optional
        If True, enables verbose output during processing. Defaults to False.
    use_fixed_r : bool, optional
        If True, uses a fixed cutoff radius for KBI calculations. Defaults to True.
    gamma_integration_type : str, optional
        The type of integration to use for gamma calculations. Defaults to 'numerical'.
    gamma_polynomial_degree : int, optional
        The degree of the polynomial to fit for gamma calculations if using polynomial integration. Defaults to 5.

    Attributes
    ----------
    state: SystemState
        Initialized SystemState object for systems as a function of composition at single temperature.
    calculator: KBICalculator
        Initialized KBICalculator object for performing KBI calculations.
    thermo: KBThermo
        Initialized KBThermo object for computing thermodynamic properties from KBIs.
    """

    def __init__(
        self,
        pure_path: str,
        pure_systems: list[str],
        base_path: str,
        base_systems: list[str] | None = None,
        ensemble: str = "npt",
        cations: list[str] | None = None,
        anions: list[str] | None = None,
        start_time: int = 0,
        verbose: bool = False,
        use_fixed_r: bool = True,
        gamma_integration_type: str = "numerical",
        gamma_polynomial_degree: int = 5,
    ) -> None:
        # build configuration
        loader = SystemLoader(verbose=verbose)
        self.config = loader.build_config(
            pure_path=pure_path,
            pure_systems=pure_systems,
            base_path=base_path,
            base_systems=base_systems,
            ensemble=ensemble,
            cations=cations or [],
            anions=anions or [],
            start_time=start_time,
        )

        # get composition state
        self.state = SystemState(self.config)

        # create KBI calculator
        self.calculator = KBICalculator(state=self.state, use_fixed_r=use_fixed_r)
        kbi_matrix = self.calculator.calculate()

        # create thermo object
        self.thermo = KBThermo(
            state=self.state,
            kbi_matrix=kbi_matrix,
            gamma_integration_type=gamma_integration_type,
            gamma_polynomial_degree=gamma_polynomial_degree,
        )

        # initialize property attribute
        self.properties: list[ThermoProperty] = []

    def run(self) -> ThermoState:
        r"""Calculate thermodynamic properties from Kirkwood-Buff theory :class:`KBThermo`."""
        # 1. Generate ThermoProperty objects
        self.properties = self._compute_properties()

        # 2. Map them into a ThermoState
        return self._build_state(self.properties)

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

    def _build_state(self, props: list[ThermoProperty]) -> ThermoState:
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
        meta = self._extract_property_object(name)

        value = meta.value
        units = meta.units
        if len(units) == 0:
            raise ValueError("This is a unitlesss property!")

        try:
            converted = self.state.Q_(value, units).to(target_units)
            return np.asarray(converted.magnitude)
        except Exception as e:
            raise ValueError(f"Could not convert units from {units} to {target_units}") from e

    def _extract_property_object(self, name: str) -> ThermoProperty:
        """Get ThermoProperty object corresponding to `name` of property."""
        if len(self.properties) == 0:
            self.properties = self._compute_properties()
        for meta in self.properties:
            if meta.name.lower() == name.lower():
                return meta
        raise ValueError(
            f"No property exists with name: {name}.\nAvailable thermodynamic properties: {self.available_properties()}"
        )

    def available_properties(self) -> list[str]:
        """Get list of available thermodynamic properties from `KBThermo`."""
        if len(self.properties) == 0:
            self.properties = self._compute_properties()
        return [p.name.lower() for p in self.properties]

    def to_dict(self) -> dict[str, NDArray[np.float64]]:
        """Create a dictionary of properties calculated from :class:`KBThermo`."""
        value_dict: dict[str, NDArray[np.float64]] = {}
        value_dict["mol_fr"] = self.state.mol_fr
        value_dict.update({meta.name: meta.value for meta in self.thermo.computed_properties()})
        return value_dict
