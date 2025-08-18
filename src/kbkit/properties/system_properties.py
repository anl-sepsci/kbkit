"""Provide a unified interface to compute topology, electron counts, and GROMACS properties."""

from natsort import natsorted
from pathlib import Path

from kbkit.utils.logging import get_logger
from kbkit.config import load_unit_registry
from kbkit.data import energy_aliases, get_gmx_unit, resolve_attr_key
from kbkit.properties import EdrFileParser, TopFileParser, GroFileParser


class SystemProperties:
    def __init__(self, syspath: str, ensemble: str = "npt", verbose: bool = False) -> None:
        self.syspath = Path(syspath) 
        self.ensemble = ensemble.lower()
        self.logger = get_logger(f"{__name__}.{self.__class__.__name__}", verbose=verbose)
        self.ureg = load_unit_registry()  # Load the unit registry for unit conversions
        self.Q_ = self.ureg.Quantity

        # File discover and parser setup
        file_map = {
            ".top": ("top", TopFileParser),
            ".gro": ("gro", GroFileParser),
            ".edr": ("edr", EdrFileParser),
        }

        for suffix, (attr, parser_cls) in file_map.items():
            filepath = self._find_file(suffix)
            setattr(self, attr, parser_cls(filepath, verbose=verbose))


    def find_file(self, suffix: str) -> str | list[str]:
        """Find files in `syspath` that contain `suffix` and `ensemble`."""
        try:
            files = list(self.syspath.glob(f"*{self.ensemble}*{suffix}"))
            filtered = [f for f in files if not any(x in f.name for x in ("init", "eqm"))]

            if not filtered:
                self.logger.error(f"No {suffix} files found.")
                raise ValueError(f"No {suffix} files found.")

            if suffix == ".edr":
                return natsorted(filtered)

            if len(filtered) > 1:
                self.logger.warning(f"Multiple {suffix} files found: {filtered}. Using {filtered[0]}.")
            return str(filtered[0])

        except Exception as e:
            raise RuntimeError(
                f"Error finding files in '{self.syspath}' with suffix '{suffix}' and ensemble '{self.ensemble}': {e}"
            ) from e
        
    
    @property
    def file_registry(self) -> dict[str, str | list[str]]:
        """str | list[str]: Registry of GROMACS file types and their values."""
        return {
            "top": self.top.top_path,
            "gro": self.gro.gro_path,
            "edr": self.edr.edr_path,
        }
    
    def _resolve_units(self, requested: str, default: str) -> str:
        """Return the requested unit if provided, otherwise fall back to the default."""
        return requested if requested else default


    def _get_average_property(
        self, name: str, start_time: float = 0, units: str = "", return_std: bool = False
    ) -> float | tuple[float, float]:
        """Returns the average property from .edr file."""

        prop = resolve_attr_key(name, energy_aliases)
        gmx_units = get_gmx_unit(prop)
        units = self._resolve_units(units, gmx_units)
        self.logger.debug(f"Fetching average property '{prop}' from .edr starting at {start_time}s with units '{units}'")

        if not self.edr.has_property(prop):
            if prop == "volume":
                try:
                    self.logger.info("Using .gro file to estimate volume since .edr lacks volume data")
                    vol = self.gro.calculate_box_volume()
                    vol_converted = float(self.Q_(vol, gmx_units).to(units).magnitude)                    
                    return (vol_converted, 0.0) if return_std else vol_converted
                except ValueError as e:
                    self.logger.error(f"Volume estimation from .gro file failed: {e}")
                    raise ValueError(f"Alternative volume calculation from .gro file failed: {e}") from e
            else:
                raise ValueError(f"GROMACS .edr file {self.file_registry['edr']} does not contain property: {prop}.")

        result = self.edr.average_property(name=prop, start_time=start_time, return_std=return_std)

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
        prop = resolve_attr_key("heat_capacity", energy_aliases)
        gmx_units = get_gmx_unit(prop)
        cap = self.edr.heat_capacity(nmol=self.top.total_molecules)
        units = self._resolve_units(units, gmx_units)
        unit_corr = self.Q_(cap, gmx_units).to(units).magnitude
        return float(unit_corr)

    def enthalpy(self, start_time: float = 0, units: str = "") -> float:
        """Compute the enthalpy of the system at a specified start time."""
        self.logger.debug(f"Calculating enthalpy from U, P, V at {start_time}s with units '{units}'")

        U = self._get_average_property("potential", start_time=start_time, units="kJ/mol")
        P = self._get_average_property("pressure", start_time=start_time, units="kPa")
        V = self._get_average_property("volume", start_time=start_time, units="m^3")
        
        H = (U + P * V) / self.top.total_molecules # convert to per molecule
        units = self._resolve_units(units, "kJ/mol")
        unit_corr = self.Q_(H, "kJ/mol").to(units).magnitude
        return float(unit_corr)

    
    def get(self, name: str, start_time: float = 0, units: str = "", std: bool = False) -> float | tuple[float, float]:
        """Fetch any available GROMACS property by name, with alias resolution and optional standard deviation."""
        name = resolve_attr_key(name, energy_aliases)
        self.logger.debug(f"Requested property '{name}' with std={std}, units='{units}', start_time={start_time}")


        if name == "heat_capacity":
            return self.heat_capacity(units=units)
        elif name == "enthalpy":
            return self.enthalpy(start_time=start_time, units=units)
        
        return self._get_average_property(name=name, start_time=start_time, units=units, return_std=std)


    

