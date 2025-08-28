"""Parser for GROMACS energy (.edr) files using `gmx energy`."""

import re
import subprocess
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from kbkit.data.property_resolver import ENERGY_ALIASES, resolve_attr_key
from kbkit.utils.logging import get_logger
from kbkit.utils.validation import validate_path


class EdrFileParser:
    """
    Interface for extracting energy properties from GROMACS .edr files.

    Wraps `gmx energy` to provide access to available properties, time series data,
    and derived quantities such as heat capacity. Supports multiple input files and
    semantic alias resolution.

    Parameters
    ----------
    edr_path : str or list[str]
        Path(s) to one or more .edr files.
    verbose : bool, optional
        If True, enables detailed logging output.
    """

    def __init__(self, edr_path: str | list[str], verbose: bool = False) -> None:
        if isinstance(edr_path, (str, Path)):
            edr_files = [str(edr_path)]
        else:
            edr_files = [str(f) for f in edr_path]
        self.edr_path = [validate_path(f, suffix=".edr") for f in edr_files]
        self.logger = get_logger(f"{__name__}.{self.__class__.__name__}", verbose=verbose)
        self.logger.info(f"Validated .edr file: {self.edr_path}")

    def available_properties(self) -> list[str]:
        """
        Return a list of available energy properties in the .edr file(s).

        Returns
        -------
        list[str]
            Sorted list of property names extracted from `gmx energy` output.

        Notes
        -----
        - Uses subprocess to invoke `gmx energy` and parse output.
        - Aggregates properties across all provided .edr files.
        """
        all_props = set()
        for edr in self.edr_path:
            try:
                result = subprocess.run(
                    ["gmx", "energy", "-f", str(edr)], check=False, input="\n", text=True, capture_output=True
                )
                output = result.stdout + result.stderr
                props = self._extract_properties(output)
                all_props.update(props)
            except Exception as e:
                self.logger.warning(f"Failed to read properties from {edr}: {e}")
        return sorted(all_props)

    def _extract_properties(self, output: str) -> list[str]:
        """
        Parse property names from raw `gmx energy` output.

        Parameters
        ----------
        output : str
            Combined stdout and stderr from `gmx energy`.

        Returns
        -------
        list[str]
            List of non-numeric tokens interpreted as property names.

        Notes
        -----
        - Uses delimiter lines to identify property blocks.
        - Filters out numeric indices and malformed lines.
        """
        lines = output.splitlines()
        props_lines = []
        recording = False
        for line in lines:
            if re.match(r"^-+\s*$", line.strip()):
                recording = not recording
                continue
            if recording and line.strip():
                props_lines.append(line)

        tokens = []
        for line in props_lines:
            try:
                tokens.extend(line.strip().split())
            except Exception as e:
                self.logger.warning(f"Could not split line: {line!r} ({e})")

        props = [token for token in tokens if not token.isdigit()]
        if not props:
            self.logger.warning(f"No properties found in '{self.edr_path}'. Output may have changed format.")
        return props

    def has_property(self, name: str) -> bool:
        """
        Check if a given property is available in the .edr file(s).

        Parameters
        ----------
        name : str
            Property name to check (case-insensitive).

        Returns
        -------
        bool
            True if the property is available, False otherwise.
        """
        return any(p.lower() == name.lower() for p in self.available_properties())

    def extract_timeseries(self, name: str, start_time: float = 0) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """
        Extract time series data for a given property.

        Parameters
        ----------
        name : str
            Property name to extract (e.g., "potential", "temperature").
        start_time : float, optional
            Time (in ps) after which data should be included.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            Time and value arrays concatenated across all .edr files.

        Notes
        -----
        - Uses semantic alias resolution for property names.
        - Automatically runs `gmx energy` if output file is missing.
        - Filters data based on start_time for reproducibility.
        """
        prop = resolve_attr_key(name, ENERGY_ALIASES)
        all_time = []
        all_values = []

        for edr in self.edr_path:
            output_file = edr.with_name(f"{prop}_{edr.stem}.xvg")
            if not output_file.exists():
                self._run_gmx_energy(prop, output_file, edr)

            try:
                time, values = np.loadtxt(output_file, comments=["@", "#"], unpack=True)
                start_idx = np.searchsorted(time, float(start_time))
                all_time.append(time[start_idx:])
                all_values.append(values[start_idx:])
            except Exception as e:
                self.logger.warning(f"Skipping {edr}: {e}")

        return np.concatenate(all_time), np.concatenate(all_values)

    def average_property(
        self, name: str, start_time: float = 0, return_std: bool = False
    ) -> float | tuple[float, float]:
        """
        Compute the average (and optionally standard deviation) of a property.

        Parameters
        ----------
        name : str
            Property name to extract and average.
        start_time : float, optional
            Time (in ps) after which data should be included.
        return_std : bool, optional
            If True, also return the standard deviation.

        Returns
        -------
        float or tuple[float, float]
            Average value, or (average, std) if `return_std` is True.
        """
        _, values = self.extract_timeseries(name, start_time)
        avg = values.mean()
        std = values.std()
        return (float(avg), float(std)) if return_std else float(avg)

    def heat_capacity(self, nmol: int) -> float:
        """
        Extract heat capacity from GROMACS energy output.

        Parameters
        ----------
        nmol : int
            Total number of molecules in the system.

        Returns
        -------
        float
            Average heat capacity in kJ/mol/K.

        Notes
        -----
        - Uses enthalpy or total energy depending on availability.
        - Applies drift correction and fluctuation analysis via `gmx energy`.
        """
        if self.has_property("enthalpy"):
            props = "Enthalpy\nTemperature\n"
            regex = r"Heat capacity at constant pressure Cp\s+=\s+([\d\.Ee+-]+)"
        else:
            props = "total-energy\nTemperature\n"
            regex = r"Heat capacity at constant volume Cv\s+=\s+([\d\.Ee+-]+)"

        capacities = []
        for edr in self.edr_path:
            try:
                result = subprocess.run(
                    ["gmx", "energy", "-f", str(edr), "-nmol", str(nmol), "-fluct_props", "-driftcorr"],
                    input=props,
                    text=True,
                    capture_output=True,
                    check=True,
                )
                match = re.search(regex, result.stdout)
                if match:
                    capacities.append(float(match.group(1)) / 1000)  # J/mol/K → kJ/mol/K
                else:
                    self.logger.warning(f"Heat capacity not found in output from {edr}")
            except subprocess.CalledProcessError as e:
                self.logger.warning(f"GROMACS energy failed for {edr}: {e.stderr}")

        if not capacities:
            raise ValueError("No heat capacity values could be extracted from any .edr file.")

        return float(np.mean(capacities))

    def _run_gmx_energy(self, prop: str, output_file: Path, edr_path: Path) -> None:
        """
        Run `gmx energy` to extract a property and write to .xvg file.

        Parameters
        ----------
        prop : str
            Property name to extract.
        output_file : Path
            Destination .xvg file path.
        edr_path : Path
            Source .edr file path.

        Notes
        -----
        - Suppresses stdout/stderr for clean execution.
        - Logs extraction for traceability.
        """
        subprocess.run(
            ["gmx", "energy", "-f", str(edr_path), "-o", str(output_file)],
            input=f"{prop}\n",
            text=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )
        self.logger.info(f"Extracted '{prop}' from {edr_path} to {output_file}")
