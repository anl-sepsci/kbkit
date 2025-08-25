"""
Provides a unified interface for extracting molecular and system-level properties from GROMACS input files.

This module wraps the gro/top/edr parsers and exposes structured access to quantities like temperature, pressure,
volume, and molecular composition. It serves as the foundation for downstream analysis modules.
"""

from pathlib import Path

from kbkit.utils.logging import get_logger
from kbkit.utils.format import resolve_units
from kbkit.config.unit_registry import load_unit_registry
from kbkit.data.property_resolver import ENERGY_ALIASES, get_gmx_unit, resolve_attr_key
from kbkit.parsers.edr_file import EdrFileParser
from kbkit.parsers.top_file import TopFileParser
from kbkit.parsers.gro_file import GroFileParser
from kbkit.utils.file_resolver import FileResolver

class SystemProperties:
    def __init__(self, system_path: str | Path, ensemble: str = "npt", start_time: int = 0, verbose: bool = False) -> None:
        self.system_path = Path(system_path) 
        self.ensemble = ensemble.lower()
        self.start_time = start_time
        self.logger = get_logger(f"{__name__}.{self.__class__.__name__}", verbose=verbose)
        self.ureg = load_unit_registry()  # Load the unit registry for unit conversions
        self.Q_ = self.ureg.Quantity

        # set up file resolver
        self.file_resolver = FileResolver(self.system_path, self.ensemble, self.logger)

        # File discover and parser setup
        role_map = {
            "topology": ("single", TopFileParser),
            "structure": ("single", GroFileParser),
            "energy": ("multi", EdrFileParser),
        }

        for attr, (mode, parser_cls) in role_map.items():
            try:
                if mode == "single":
                    filepath = self.file_resolver.get_file(attr)
                elif mode == "multi":
                    filepath = self.file_resolver.get_all(attr)
                setattr(self, attr, parser_cls(filepath, verbose=verbose))
            except FileNotFoundError:
                if verbose:
                    self.logger.warning(f"No file(s) found for role '{attr}' in {self.system_path}")
                setattr(self, attr, None)
    
    @property
    def file_registry(self) -> dict[str, str | list[str]]:
        """str | list[str]: Registry of GROMACS file types and their values."""
        return {
            "top": self.topology.top_path,
            "gro": self.structure.gro_path,
            "edr": self.energy.edr_path,
        }

    def _get_average_property(
        self, name: str, start_time: float = 0, units: str = "", return_std: bool = False
    ) -> float | tuple[float, float]:
        """Returns the average property from .edr file."""

        prop = resolve_attr_key(name, ENERGY_ALIASES)
        gmx_units = get_gmx_unit(prop)
        units = resolve_units(units, gmx_units)
        start_time = start_time if start_time > 0 else self.start_time
        self.logger.debug(f"Fetching average property '{prop}' from .edr starting at {start_time}s with units '{units}'")

        if not self.energy.has_property(prop):
            if prop == "volume":
                try:
                    self.logger.info("Using .gro file to estimate volume since .edr lacks volume data")
                    vol = self.structure.calculate_box_volume()
                    vol_converted = float(self.Q_(vol, gmx_units).to(units).magnitude)                    
                    return (vol_converted, 0.0) if return_std else vol_converted
                except ValueError as e:
                    self.logger.error(f"Volume estimation from .gro file failed: {e}")
                    raise ValueError(f"Alternative volume calculation from .gro file failed: {e}") from e
            else:
                raise ValueError(f"GROMACS .edr file {self.file_registry['edr']} does not contain property: {prop}.")

        result = self.energy.average_property(name=prop, start_time=start_time, return_std=return_std)

        if return_std:
            avg_val, std_val = result
            avg_converted = self.Q_(avg_val, gmx_units).to(units).magnitude
            std_converted = self.Q_(std_val, gmx_units).to(units).magnitude
            return float(avg_converted), float(std_converted)
        else:
            avg_converted = self.Q_(result, gmx_units).to(units).magnitude
            return float(avg_converted)
        

    def heat_capacity(self, units: str = "") -> float:
        """Compute the heat capacity of the system."""
        self.logger.debug(f"Calculating heat capacity with units '{units}'")
        prop = resolve_attr_key("heat_capacity", ENERGY_ALIASES)
        gmx_units = get_gmx_unit(prop)
        cap = self.energy.heat_capacity(nmol=self.topology.total_molecules)
        units = resolve_units(units, gmx_units)
        unit_corr = self.Q_(cap, gmx_units).to(units).magnitude
        return float(unit_corr)

    def enthalpy(self, start_time: float = 0, units: str = "") -> float:
        """Compute the enthalpy of the system at a specified start time."""
        self.logger.debug(f"Calculating enthalpy from U, P, V at {start_time}s with units '{units}'")
        start_time = start_time if start_time > 0 else self.start_time

        U = self._get_average_property("potential", start_time=start_time, units="kJ/mol")
        P = self._get_average_property("pressure", start_time=start_time, units="kPa")
        V = self._get_average_property("volume", start_time=start_time, units="m^3")
        
        H = (U + P * V) / self.topology.total_molecules # convert to per molecule
        units = resolve_units(units, "kJ/mol")
        unit_corr = self.Q_(H, "kJ/mol").to(units).magnitude
        return float(unit_corr)

    
    def get(self, name: str, start_time: float = 0, units: str = "", std: bool = False) -> float | tuple[float, float]:
        """Fetch any available GROMACS property by name, with alias resolution and optional standard deviation."""
        name = resolve_attr_key(name, ENERGY_ALIASES)
        self.logger.debug(f"Requested property '{name}' with std={std}, units='{units}', start_time={start_time}")
        start_time = start_time if start_time > 0 else self.start_time

        if name == "heat_capacity":
            return self.heat_capacity(units=units)
        elif name == "enthalpy":
            return self.enthalpy(start_time=start_time, units=units)
        
        return self._get_average_property(name=name, start_time=start_time, units=units, return_std=std)


    

