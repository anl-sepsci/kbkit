"""Unified interface for extracting molecular and system-level properties from GROMACS input files."""

from pathlib import Path
from typing import Union

from kbkit.config.unit_registry import load_unit_registry
from kbkit.data.property_resolver import ENERGY_ALIASES, get_gmx_unit, resolve_attr_key
from kbkit.parsers.edr_file import EdrFileParser
from kbkit.parsers.gro_file import GroFileParser
from kbkit.parsers.top_file import TopFileParser
from kbkit.utils.file_resolver import FileResolver
from kbkit.utils.format import resolve_units
from kbkit.utils.logging import get_logger


class SystemProperties:
    """
    Interface for accessing thermodynamic and structural properties of a GROMACS system.

    Combines topology (.top), structure (.gro), and energy (.edr) files into a unified
    property accessor. Supports alias resolution, unit conversion, and ensemble-aware
    file discovery.

    Parameters
    ----------
    system_path : str or Path
        Path to the system directory containing GROMACS files.
    ensemble : str, optional
        Ensemble name used to refine file matching (e.g., "npt", "nvt").
    start_time : int, optional
        Default start time (in ps) for time-averaged properties.
    verbose : bool, optional
        If True, enables detailed logging output.

    Attributes
    ----------
    topology: TopFileParser | GroFileParser
        GROMACS topology file parser (.gro or .top files).
    energy: EdrFileParser
        GROMACS energy file parser (.edr files).
    """

    def __init__(
        self, system_path: str | Path, ensemble: str = "npt", start_time: int = 0, verbose: bool = False
    ) -> None:
        self.system_path = Path(system_path)
        self.ensemble = ensemble.lower()
        self.start_time = start_time
        self.verbose = verbose
        self.logger = get_logger(f"{__name__}.{self.__class__.__name__}", verbose=verbose)
        self.ureg = load_unit_registry()  # Load the unit registry for unit conversions
        self.Q_ = self.ureg.Quantity

        # Set up file resolver
        self.file_resolver = FileResolver(self.system_path, self.ensemble, self.logger)

        # discover files and setup parsers
        self._discover_files()

    def _discover_files(self) -> None:
        """Discover files and initialize their parsers."""
        # log which files were discovered
        discovered_files = {"topology": False, "structure": False, "energy": False}
        try:
            top_file = self.file_resolver.get_file("topology")
            discovered_files["topology"] = True
        except FileNotFoundError:
            self.logger.warning(f"Topology file not found in '{self.system_path}'.")

        try:
            structure_files = self.file_resolver.get_all("structure")
            discovered_files["structure"] = True
        except FileNotFoundError:
            self.logger.warning(f"Structure file not found in '{self.system_path}'.")

        try:
            energy_files = self.file_resolver.get_all("energy")
            discovered_files["energy"] = True
        except FileNotFoundError as e:
            self.logger.error(f"Energy file(s) not found in '{self.system_path}'")
            raise FileNotFoundError(f"Energy file(s) not found in '{self.system_path}'") from e

        self.energy = EdrFileParser(energy_files, verbose=self.verbose)

        # now debug so that at least one structure/topology is found
        # Declare the type once, before the conditional
        self.topology: Union[GroFileParser, TopFileParser]

        if discovered_files["structure"]:
            for file in structure_files:
                self.topology = GroFileParser(file, verbose=self.verbose)
                if any(len(mol) > 1 for mol in self.topology.molecules):
                    break
        elif discovered_files["topology"]:
            self.topology = TopFileParser(top_file, verbose=self.verbose)
        else:
            raise FileNotFoundError(f"No topology or structure file found in '{self.system_path}'")

    @property
    def file_registry(self) -> dict[str, Path | list[Path]]:
        """
        Return a registry of resolved GROMACS file paths.

        Returns
        -------
        dict[str, str or list[str]]
            Dictionary mapping file types ("top", "edr") to their paths.
        """
        return {
            "top": self.topology.filepath,
            "edr": self.energy.edr_path,
        }

    def _get_average_property(
        self, name: str, start_time: float = 0, units: str = "", return_std: bool = False
    ) -> float | tuple[float, float]:
        """
        Compute the average (and optionally standard deviation) of a property from .edr or fallback sources.

        Parameters
        ----------
        name : str
            Property name to extract (e.g., "temperature", "volume").
        start_time : float, optional
            Time (in ps) after which data should be included.
        units : str, optional
            Desired output units.
        return_std : bool, optional
            If True, also return the standard deviation.

        Returns
        -------
        float or tuple[float, float]
            Average value, or (average, std) if `return_std` is True.
        """
        prop = resolve_attr_key(name, ENERGY_ALIASES)
        gmx_units = get_gmx_unit(prop)
        units = resolve_units(units, gmx_units)
        start_time = start_time if start_time > 0 else self.start_time
        self.logger.debug(
            f"Fetching average property '{prop}' from .edr starting at {start_time}s with units '{units}'"
        )

        if not self.energy.has_property(prop):
            if prop == "volume":
                try:
                    self.logger.info("Using .gro file to estimate volume since .edr lacks volume data")
                    vol = self.topology.box_volume
                    vol_converted = float(self.Q_(vol, gmx_units).to(units).magnitude)
                    return (vol_converted, 0.0) if return_std else vol_converted
                except ValueError as e:
                    self.logger.error(f"Volume estimation from .gro file failed: {e}")
                    raise ValueError(f"Alternative volume calculation from .gro file failed: {e}") from e
            else:
                raise ValueError(
                    f"GROMACS .edr file '{self.file_registry['edr']}' does not contain property: '{prop}'."
                )

        result = self.energy.average_property(name=prop, start_time=start_time, return_std=return_std)
        if isinstance(result, tuple):
            avg_val, std_val = result
            avg_converted = self.Q_(avg_val, gmx_units).to(units).magnitude
            std_converted = self.Q_(std_val, gmx_units).to(units).magnitude
            return float(avg_converted), float(std_converted)
        else:
            avg_converted = self.Q_(result, gmx_units).to(units).magnitude
            return float(avg_converted)

    def _heat_capacity(self, start_time: float = 0, units: str = "") -> float:
        """
        Compute the heat capacity of the system.

        Parameters
        ----------
        units : str, optional
            Desired output units (default: kJ/mol/K).

        Returns
        -------
        float
            Heat capacity in the requested units.
        """
        self.logger.debug(f"Calculating heat capacity with units '{units}'")
        prop = resolve_attr_key("heat_capacity", ENERGY_ALIASES)
        gmx_units = get_gmx_unit(prop)
        start_time = start_time if start_time > 0 else self.start_time

        cap = self.energy.heat_capacity(nmol=self.topology.total_molecules, start_time=start_time)
        units = resolve_units(units, gmx_units)
        return float(self.Q_(cap, gmx_units).to(units).magnitude)

    def _isothermal_compressiblity(self, start_time: float = 0, units: str = "") -> float:
        """
        Compute the isothermal compressibility of the system.

        Parameters
        ----------
        units : str, optional
            Desired output units (default: 1/kPa).

        Returns
        -------
        float
            Isothermal compressibility in the requested units.
        """
        self.logger.debug(f"Calculating isothermal compressibility with units '{units}'")
        prop = resolve_attr_key("isothermal_compressibility", ENERGY_ALIASES)
        gmx_units = get_gmx_unit(prop)
        start_time = start_time if start_time > 0 else self.start_time

        kappa = self.energy.isothermal_compressibility(start_time=start_time)
        units = resolve_units(units, gmx_units)
        return float(self.Q_(kappa, gmx_units).to(units).magnitude)

    def _enthalpy(self, start_time: float = 0, units: str = "") -> float:
        r"""
        Compute the enthalpy of the system (:math:`H`) from potential energy (:math:`U`).

        .. math::
            H = U + P \cdot V

        Parameters
        ----------
        start_time : float, optional
            Time (in ps) after which data should be included.
        units : str, optional
            Desired output units (default: kJ/mol).

        Returns
        -------
        float
            Enthalpy in the requested units.
        """
        self.logger.debug(f"Calculating enthalpy from U, P, V at {start_time}s with units '{units}'")
        start_time = start_time if start_time > 0 else self.start_time

        U = self._get_average_property("potential", start_time=start_time, units="kJ/mol", return_std=False)
        P = self._get_average_property("pressure", start_time=start_time, units="kPa", return_std=False)
        V = self._get_average_property("volume", start_time=start_time, units="m^3", return_std=False)

        U = float(U[0]) if isinstance(U, tuple) else U
        P = float(P[0]) if isinstance(P, tuple) else P
        V = float(V[0]) if isinstance(V, tuple) else V

        H = (U + P * V) / self.topology.total_molecules  # convert to per molecule
        units = resolve_units(units, "kJ/mol")
        return float(self.Q_(H, "kJ/mol").to(units).magnitude)

    def get(self, name: str, start_time: float = 0, units: str = "", std: bool = False) -> float | tuple[float, float]:
        """
        Fetch any available GROMACS property by name, with alias resolution and optional standard deviation.

        Parameters
        ----------
        name : str
            Property name to extract (e.g., "pressure", "enthalpy").
        start_time : float, optional
            Time (in ps) after which data should be included.
        units : str, optional
            Desired output units.
        std : bool, optional
            If True, also return the standard deviation.

        Returns
        -------
        float or tuple[float, float]
            Property value, or (value, std) if `std` is True.
        """
        name = resolve_attr_key(name, ENERGY_ALIASES)
        self.logger.debug(f"Requested property '{name}' with std={std}, units='{units}', start_time={start_time}")
        start_time = start_time if start_time > 0 else self.start_time

        if name == "heat_capacity":
            return self._heat_capacity(start_time=start_time, units=units)
        elif name == "enthalpy":
            return self._enthalpy(start_time=start_time, units=units)
        elif name == "isothermal_compressibility":
            return self._isothermal_compressiblity(start_time=start_time, units=units)

        return self._get_average_property(name=name, start_time=start_time, units=units, return_std=std)
