"""
High-level orchestration layer for running thermodynamic analysis workflows.

This module coordinates system discovery, RDF/KBI parsing, and thermodynamic matrix construction.
Intended for use in CLI tools, notebooks, or automated pipelines.
"""

from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from kbkit.analysis.kb_thermo import KBThermo
from kbkit.analysis.kbi_calculator import KBICalculator
from kbkit.analysis.system_features import SystemFeatures
from kbkit.core.system_loader import SystemLoader


class KBPipeline:
    """
    A pipeline for performing Kirkwood-Buff analysis of molecular simulations.

    Parameters
    ----------
    base_path : str, optional
        The base path where the systems are located. Defaults to the current working directory.
    pure_path : str, optional
        The path where pure component systems are located. Defaults to a 'pure_components' directory next to the base path.
    system_names : list, optional
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
    """

    def __init__(
        self,
        base_path: str | Path | None = None,
        pure_path: str | Path | None = None,
        ensemble: str = "npt",
        cations: list[str] | None = None,
        anions: list[str] | None = None,
        start_time: int = 0,
        system_names: list[str] | None = None,
        verbose: bool = False,
        gamma_integration_type: str = "numerical",
    ):
        # build configuration
        loader = SystemLoader(verbose=verbose)
        self.config = loader.build_config(
            base_path=base_path,
            pure_path=pure_path,
            ensemble=ensemble,
            cations=cations or [],
            anions=anions or [],
            start_time=start_time,
            system_names=system_names,
        )

        # get composition analyzer
        self.analyzer = SystemFeatures(self.config)

        # create KBI calculator
        self.calculator = KBICalculator(self.config, self.analyzer)
        kbi_matrix = self.calculator.get_corrected_kbi_matrix()

        # create thermo object
        self.thermo = KBThermo(self.analyzer, kbi_matrix)

        # get property for activity coefficient integration
        self.gamma_integration_type = gamma_integration_type

    def run(self) -> None:
        r"""Run Kirkwood-Buff analysis via :class:`kbkit.analysis.thermo.KBThermo`."""
        lngammas = self.thermo.lngammas(self.gamma_integration_type)
        self.thermo.gm(lngammas)
        self.thermo.se(lngammas)
        self.thermo.i0()
        self.thermo.det_hessian()
        self.thermo.isothermal_compressability()

    def convert_units(self, name: str, target_units: str) -> NDArray[np.float64]:
        """Get thermodynamic property in desired units."""
        if name.lower() not in self.thermo._cache:
            try:
                self.run()
            except KeyError as e:
                raise KeyError(f"Property {name.lower()} not in cache. Check that property has been computed.") from e

        value = self.thermo._cache[name.lower()].value
        units = self.thermo._cache[name.lower()].units
        try:
            converted = self.analyzer.Q_(value, units).to(target_units)
            return np.asarray(converted.magnitude)
        except Exception as e:
            raise ValueError(f"Could not convert units from {units} to {target_units}") from e

    def to_dict(self) -> dict[str, NDArray[np.float64]]:
        """Create a dictionary of properties calculated from :class:`kbkit.kb.kb_thermo.KBThermo`."""
        value_dict: dict[str, NDArray[np.float64]] = {}
        value_dict["mol_fr"] = self.analyzer.mol_fr
        value_dict.update({name: meta.value for name, meta in self.thermo._cache.items()})
        return value_dict
