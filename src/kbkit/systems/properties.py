"""Unified interface for extracting molecular and system-level properties from GROMACS input files."""

import inspect
from functools import cached_property
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from kbkit.config.unit_registry import load_unit_registry
from kbkit.io import EnergyParser, TopologyParser
from kbkit.utils.format import ENERGY_ALIASES, resolve_attr_key
from kbkit.utils.validation import validate_path
from kbkit.visualization.timeseries import TimeseriesPlotter


class SystemProperties:
    """
    Interface for accessing thermodynamic and structural properties of a GROMACS system.

    Combines topology (.top), structure (.gro), and energy (.edr) files into a unified property accessor.
    Supports alias resolution, unit conversion, and ensemble-aware file discovery.

    Parameters
    ----------
    system_path : str or Path, optional
        Path to the system directory containing GROMACS files.
    include : str, optional
        String to include in file name for valid file. Only used if multiple files are found with the same suffix.
    energy: str or Path, optional
        Path for an energy file. Supported filetypes: ".edr", ".log", ".lammps"
    topology: str or Path, optional
        Path for a topology file. Supported filetypes: ".top", ".gro", ".lmp"
    start : int, optional
        Starting point for when data should be used. GROMACS: time (ps), LAMMPS: timestep (i.e., for 1 fs steps, start=1000--start at 1 ps).

    Attributes
    ----------
    energy: list[Path]
        List of paths to energy files.
    topology: list[Path]
        List of paths to topology files.

    .. note::
        - Defaults to looking at files/paths directly specified.
        - If files are not specified or do not exist, a ``system_path`` is required to locate the files with necessary suffix.
    """

    def __init__(
        self,
        path: str | None = None,
        include: str = "",
        energy: str | None = None,
        topology: str | None = None,
        start: int = 0,
    ) -> None:
        self.start = start

        # setup registry for unit conversions
        self.ureg = load_unit_registry()  # Load the unit registry for unit conversions
        self.Q_ = self.ureg.Quantity

        # validate system paths
        self.parent = validate_path(path) if path is not None else None

        # first try to resolve files if specified, otherwise search for files in path.
        self.energy_files = self._get_files(
            path=path, filename=energy, suffixes=[".edr", ".lammps", ".log"], include=include
        )

        self.topology_files = self._get_files(
            path=path, filename=topology, suffixes=[".gro", ".top", ".lmp"], include=include
        )

        # update system-path if not defined and all files have same parent
        if self.parent is None:
            energy_parents = [f.parent for f in self.energy_files]
            topo_parents = [f.parent for f in self.topology_files]
            parents = np.unique([energy_parents + topo_parents])
            if len(parents) == 1:
                self.parent = parents[0]

    @staticmethod
    def _get_abspath(path: str, filename: str) -> Path:
        """
        Get absolute filepath.

        Parameters
        ----------
        path: str
            Parent path containing files.
        filename: str
            Path to file.

        Returns
        -------
        Path
            Path object to valid file.
        """
        filepath = Path(filename).resolve()
        if filepath.is_file():
            return filepath

        if path is not None:
            abspath = Path(f"{path}/{filename}").resolve()
            if abspath.is_file():
                return abspath

        raise FileNotFoundError("File does not exist!")

    @staticmethod
    def _get_files(path: str, filename: str, suffixes: list[str], include: str) -> list[Path]:
        """Get files for a suffix, priorty given in the order of suffixes.

        Parameters
        ----------
        path: str
            Parent path containing files.
        filename: str
            Path to file.
        suffixes: str
            File types to iterate through and search for. (i.e., `.edr`, `.gro`, `.top`)
        include: str, optional
            String to filter files by. Will only incorporate if more than one file of the desired suffix is found.

        Returns
        -------
        list[Path]
            List of path objects containing files of a valid suffix.
        """
        if filename is not None:
            try:
                return [SystemProperties._get_abspath(path, filename)]
            except FileNotFoundError:
                pass

        if path is None:
            raise ValueError("path is required to find unspecified files!")

        for suffix in suffixes:
            files = SystemProperties._find_files_in_path(suffix=suffix, include=include, path=path)
            if len(files) > 0:
                return files

        raise FileNotFoundError(f"No files found with any of the suffixes: {suffixes} in path: {path}.")

    @staticmethod
    def _find_files_in_path(
        suffix: str,
        path: str | Path | None = None,
        include: str = "",
        exclude: list[str] | None = None,
    ) -> list[Path]:
        """
        Get list of files with a given suffix in system directory.

        Parameters
        ----------
        suffix: str
            File type to search for. (i.e., `.edr`, `.gro`, `.top`)
        path: str, optional
            Parent path containing files.
        include: str, optional
            String to filter files by. Will only incorporate if more than one file of the desired suffix is found.
        exclude: list[str], optional
            String to exclude from valid files. Will only be searched if more than 1 files found after ``include`` filter.

        Returns
        -------
        list[Path]
            List of path objects containing files of desired suffix.
        """
        # validate filepath and parent directory
        if path:
            path = validate_path(path)
        else:
            raise ValueError("A valid 'filepath' or 'system_path' is required!")

        # get files
        files = sorted(path.glob(f"*.{suffix.strip('.')}"))

        if len(files) == 1:
            return files

        # filter files by ``include`` argument if more than one files are found.
        files_filtered = [f for f in list(files) if include in f.name]
        if not files_filtered:
            return files

        # refine files 1 more time by things not to include; i.e., inital runs
        exclude = exclude or ["init", "eq", "em"]
        files_filtered_again = sorted([f for f in files_filtered if not any(x in f.name for x in exclude)])
        return files_filtered if not files_filtered_again else files_filtered_again

    @cached_property
    def energy(self) -> list[EnergyParser]:
        """list[EnergyParser]: Setup Energy file parsers for all files in ``energy_files``."""
        return [EnergyParser(Path(fpath)) for fpath in self.energy_files]

    @cached_property
    def topology(self) -> TopologyParser:
        """TopologyParser: Setup Topology parser."""
        return TopologyParser(path=self.topology_files[0])

    @property
    def topology_properties(self) -> list[str]:
        """list[str]: Get list of accessible topology properties."""
        return [name for name, _ in inspect.getmembers(self.topology) if not name.startswith("_")]

    def get(self, name: str, units: str | None = None, avg: bool = True, time_series: bool = False) -> Any:
        """
        Master function for getting any property from ``energy`` or ``topology`` files.

        Parameters
        ----------
        name: str
            Name for the property to extract.
        units: str, optional
            Units to convert energy properties to. If not specified, default units from `pyedr` will be used.
        avg: bool, optional
            Returns averaged property if True (default: True). Otherwise returns array of values.
        time_series: bool, optional
            Returns both times and values if True (default: False).

        Returns
        -------
        float | np.ndarray | list[np.ndarray]
            Topology or energy property in desired units.
        """
        # 1. if property is in topology; return it
        if name in self.topology_properties:
            return getattr(self.topology, name)

        # now triple check if electrons are desired but another name is used
        if any(xx in name.lower() for xx in ("elec", "z_", "z-")):
            return self.topology.electron_count

        # 2. now for energy properties
        # first check if property are units; in which case just return unit dictionary
        if "unit" in name.lower():
            return self.energy[0].units

        box_volume = self.topology.box_volume

        # resolves common property names for all EDR properties
        prop = resolve_attr_key(name, ENERGY_ALIASES).lower()

        x_key = self.energy[0]._x_key
        x_arr: list[float] = []
        values = []
        value: float | np.ndarray

        for _i, parser in enumerate(self.energy):
            if prop == "cp":
                value = parser.heat_capacity_cp(
                    nmol=self.topology.total_molecules, volume=box_volume, start=self.start, units=units
                )
            elif prop == "cv":
                value = parser.heat_capacity_cv(nmol=self.topology.total_molecules, start=self.start, units=units)
            elif prop == "enthalpy":
                value = parser.molar_enthalpy(
                    nmol=self.topology.total_molecules, volume=box_volume, start=self.start, units=units
                )
            elif prop == "isothermal-compressibility":
                value = parser.isothermal_compressibility(start=self.start, units=units)
            elif prop in ("number-density", "molar-volume"):
                # get molar volume and convert to number density if desired
                units = units or parser.units["molar-volume"]
                units = units if prop == "molar-volume" else f"{units.split('/')[1]}/{units.split('/')[0]}"
                Vi = parser.molar_volume(
                    nmol=self.topology.total_molecules, volume=box_volume, start=self.start, units=units
                )

                if prop == "number-density":
                    value = 1 / Vi
                else:
                    value = Vi

            else:
                value = parser.get(prop, start=self.start, units=units)

            # now average values if desired
            if avg or prop in EnergyParser.FLUCT_PROPS:
                values.append(np.mean(value))
            elif isinstance(value, (list | np.ndarray)):
                values.extend(value)
                x_arr.extend(parser.get(x_key, start=self.start))

        # return desired values
        if avg or prop in EnergyParser.FLUCT_PROPS:
            # Add check before computing mean
            if len(values) > 0:
                return float(np.mean(values))
            else:
                return np.nan
        else:
            # place into pd.DataFrame and sort by times; if any duplicates are found--remove them
            df = pd.DataFrame({x_key: x_arr, "values": values})
            df.sort_values(x_key, inplace=True)
            df.drop_duplicates(subset=[x_key], keep="first", inplace=True)
            arr = df.to_numpy()
            # return times and values if desired
            if time_series:
                return arr[:, 0], arr[:, 1]
            else:
                return arr[:, 1]

    def timeseries_plotter(self, start: int = 0) -> TimeseriesPlotter:
        """
        Create a TimeseriesPlotter for visualizing time series data for a given system.

        Parameters
        ----------
        start: int
            Initial time for plotting.

        Returns
        -------
        TimeseriesPlotter
            Plotter instance for computing simulation energy properties.
        """
        return TimeseriesPlotter(self, start=start)
