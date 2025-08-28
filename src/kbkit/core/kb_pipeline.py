"""High-level orchestration layer for running thermodynamic analysis workflows."""

import numpy as np
from numpy.typing import NDArray

from kbkit.analysis.kb_thermo import KBThermo
from kbkit.analysis.system_state import SystemState
from kbkit.calculators.kbi_calculator import KBICalculator
from kbkit.core.system_loader import SystemLoader


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
    start_time : int, optional
        The starting time for analysis, used in temperature and enthalpy calculations. Defaults to 0.
    ensemble : str, optional
        The ensemble type for the systems, e.g., 'npt', 'nvt'. Defaults to 'npt'.
    cations : list, optional
        A list of cation names to consider for salt pairs. Defaults to an empty list.
    anions : list, optional
        A list of anion names to consider for salt pairs. Defaults to an empty list.
    gamma_integration_type : str, optional
        The type of integration to use for gamma calculations. Defaults to 'numerical'.

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
        gamma_integration_type: str = "numerical",
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
        self.calculator = KBICalculator(self.state)
        kbi_matrix = self.calculator.calculate()

        # create thermo object
        self.thermo = KBThermo(self.state, kbi_matrix)

        # get property for activity coefficient integration
        self.gamma_integration_type = gamma_integration_type

    def run(self) -> None:
        r"""Run Kirkwood-Buff analysis via :class:`kbkit.analysis.thermo.KBThermo`."""
        self.thermo.build_cache(self.gamma_integration_type)

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
        if name.lower() not in self.thermo._cache:
            try:
                self.run()
            except KeyError as e:
                raise KeyError(f"Property {name.lower()} not in cache. Check that property has been computed.") from e

        value = self.thermo._cache[name.lower()].value
        units = self.thermo._cache[name.lower()].units
        try:
            converted = self.state.Q_(value, units).to(target_units)
            return np.asarray(converted.magnitude)
        except Exception as e:
            raise ValueError(f"Could not convert units from {units} to {target_units}") from e

    def to_dict(self) -> dict[str, NDArray[np.float64]]:
        """Create a dictionary of properties calculated from :class:`kbkit.kb.kb_thermo.KBThermo`."""
        value_dict: dict[str, NDArray[np.float64]] = {}
        value_dict["mol_fr"] = self.state.mol_fr
        value_dict.update({name: meta.value for name, meta in self.thermo._cache.items()})
        return value_dict
