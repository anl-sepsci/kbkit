"""Unified interface for extracting molecular and system-level properties from GROMACS input files."""

import inspect
from functools import cached_property
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from kbkit.config.unit_registry import load_unit_registry
from kbkit.io import EdrParser, GroParser, TopParser
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
    edr_path: str or Path, optional
        Path for an energy (.edr) file.
    top_path: str or Path, optional
        Path for a topology (.top) file.
    gro_path: str or Path, optional
        Path for a structure (.gro) file.
    start_time : int, optional
        Default start time (in ps) for time-averaged properties.

    Attributes
    ----------
    edr_paths: list[Path]
        List of paths to GROMACS energy files.
    top_paths: list[Path]
        List of paths to GROMACS topology files.
    gro_paths: list[Path]
        List of paths to GROMACS structure files.

    .. note::
        - Defaults to looking at files/paths directly specified.
        - If files are not specified or do not exist, a ``system_path`` is required to locate the files with necessary suffix.
    """

    def __init__(
        self,
        system_path: str | None = None,
        include: str = "",
        edr_path: str | None = None,
        top_path: str | None = None,
        gro_path: str | None = None,
        start_time: float = 0.,
    ) -> None:
        self.start_time = float(start_time)

        # setup registry for unit conversions
        self.ureg = load_unit_registry()  # Load the unit registry for unit conversions
        self.Q_ = self.ureg.Quantity

        # validate system paths
        self.system_path = validate_path(system_path) if system_path else system_path

        # get files; first prioritize specified file; then search directory if files do not exist
        self.edr_paths = self.find_files(
            suffix=".edr", include=include, filepath=edr_path, system_path=self.system_path
        )
        self.gro_paths = self.find_files(
            suffix=".gro", include=include, filepath=gro_path, system_path=self.system_path
        )
        self.top_paths = self.find_files(
            suffix=".top", include=include, filepath=top_path, system_path=self.system_path
        )

    @staticmethod
    def find_files(
        suffix: str,
        filepath: str | Path | None = None,
        system_path: str | Path | None = None,
        include: str = "",
        exclude: list[str] | None = None,
    ) -> list[Path]:
        """
        Get list of files with a given suffix in system directory.

        Parameters
        ----------
        suffix: str
            File type to search for. (i.e., `.edr`, `.gro`, `.top`)
        filepath: str, optional
            Optional filepath to validate. Optional if ``system_path`` is included.
        system_path: str, optional
            Parent path containing files. Optional if ``filepath`` is included.
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
        if filepath:
            filepath = validate_path(filepath, suffix)
            system_path = validate_path(filepath.parent)
        elif system_path:
            system_path = validate_path(system_path)
        else:
            raise ValueError("A valid 'filepath' or 'system_path' is required!")

        # get files
        files = [filepath] if filepath else sorted(system_path.glob(f"*.{suffix.strip('.')}"))

        if not files:
            raise ValueError(f"No files with '{suffix}' found in '{system_path}'")

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
    def energy(self) -> list[EdrParser]:
        """list[EdrParser]: Setup EDR file parsers for all files in ``edr_paths``."""
        if len(self.edr_paths) == 0:
            raise FileNotFoundError("Energy file(s) do not exist!")

        return [EdrParser(Path(fpath)) for fpath in self.edr_paths]

    @cached_property
    def topology(self) -> GroParser | TopParser:
        """GroParser | TopParser: Setup Gro/Top parser."""
        # prioritize GRO files -- they contain electron info as well
        if len(self.gro_paths) > 0:
            for file in self.gro_paths:
                gro = GroParser(file)
                if any(len(mol) > 1 for mol in gro.molecules):
                    return gro
        # if gro file(s) are invalid; use topology file
        if len(self.top_paths) > 0:
            for file in self.top_paths:
                return TopParser(file)
        raise FileNotFoundError(f"No topology or structure file found in '{self.system_path}'")

    @property
    def topology_properties(self) -> list[str]:
        """list[str]: Get list of accessible topology properties."""
        return [name for name, _ in inspect.getmembers(self.topology) if not name.startswith("_")]

    def get(
        self, name: str, units: str | None = None, avg: bool = True, time_series: bool = False
    ) -> Any:
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

        # now compute properties from edr-files

        times: list[float] = []
        values = []
        value: float | np.ndarray

        for _i, edr in enumerate(self.energy):
            if prop == "cp":
                value = edr.cp(
                    nmol=self.topology.total_molecules, volume=box_volume, start_time=self.start_time, units=units
                )
            elif prop == "cv":
                value = edr.cv(nmol=self.topology.total_molecules, start_time=self.start_time, units=units)
            elif prop == "enthalpy":
                value = edr.molar_enthalpy(
                    nmol=self.topology.total_molecules, volume=box_volume, start_time=self.start_time, units=units
                )
            elif prop == "isothermal-compressibility":
                value = edr.isothermal_compressibility(start_time=self.start_time, units=units)
            elif prop in ("number-density", "molar-volume"):
                # get molar volume and convert to number density if desired
                units = units or edr.units["molar-volume"]
                units = units if prop == "molar-volume" else f"{units.split('/')[1]}/{units.split('/')[0]}"
                Vi = edr.molar_volume(
                    nmol=self.topology.total_molecules, volume=box_volume, start_time=self.start_time, units=units
                )

                if prop == "number-density":
                    value = 1 / Vi
                else:
                    value = Vi

            else:
                value = edr.get(prop, start_time=self.start_time, units=units)

            # now average values if desired
            if avg or prop in EdrParser.FLUCT_PROPS:
                values.append(np.mean(value))
            elif isinstance(value, (list | np.ndarray)):
                values.extend(value)
                times.extend(edr.get("time", start_time=self.start_time))

        # return desired values
        if avg or prop in EdrParser.FLUCT_PROPS:
            # Add check before computing mean
            if len(values) > 0:
                return float(np.mean(values))
            else:
                return np.nan
        else:
            # place into pd.DataFrame and sort by times; if any duplicates are found--remove them
            df = pd.DataFrame({"times": times, "values": values})
            df.sort_values("times", inplace=True)
            df.drop_duplicates(subset=["times"], keep="first", inplace=True)
            arr = df.to_numpy()
            # return times and values if desired
            if time_series:
                return arr[:, 0], arr[:, 1]
            else:
                return arr[:, 1]

    def timeseries_plotter(self, start_time: int = 0) -> TimeseriesPlotter:
        """
        Create a TimeseriesPlotter for visualizing time series data for a given system.

        Parameters
        ----------
        start_time: int
            Initial time for plotting.

        Returns
        -------
        TimeseriesPlotter
            Plotter instance for computing simulation energy properties.
        """
        return TimeseriesPlotter(self, start_time=start_time)
