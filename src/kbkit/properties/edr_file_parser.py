"""Extract GROMACS energy properties."""

import re
import subprocess
from typing import Sequence
import numpy as np
from numpy.typing import NDArray
from pathlib import Path

from kbkit.data import energy_aliases, resolve_attr_key
from kbkit.utils.validation import validate_file
from kbkit.utils.logging import get_logger


class EdrFileParser:
    """
    Compute energy properties via gmx energy from an .edr file.

    Parameters
    ----------
    edr_path: str
        Path to the .edr file.
    verbose: bool, optional
        If True, enables detailed logging output.
    """

    def __init__(self, edr_path: str | list[str], verbose: bool = False) -> None:
        if isinstance(edr_path, (str, Path)):
            edr_files = [str(edr_path)]
        else:
            edr_files = [str(f) for f in edr_path]
        self.edr_path = [validate_file(f, suffix=".edr") for f in edr_files]
        self.logger = get_logger(f"{__name__}.{self.__class__.__name__}", verbose=verbose)
        self.logger.info(f"Validated .edr file: {self.edr_path}")
    
    def available_properties(self) -> list[str]:
        """Returns available energy properties from the .edr file."""
        all_props = set()
        for edr in self.edr_path:
            try:
                result = subprocess.run(
                    ["gmx", "energy", "-f", str(edr)],
                    input="\n", text=True, capture_output=True
                )
                output = result.stdout + result.stderr
                props = self._extract_properties(output)
                all_props.update(props)
            except Exception as e:
                self.logger.warning(f"Failed to read properties from {edr}: {e}")
        return sorted(all_props)
    
    def _extract_properties(self, output: str) -> list[str]:
        """Extract property names from gmx energy output."""
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
        return any(p.lower() == name.lower() for p in self.available_properties())

    def extract_timeseries(self, name: str, start_time: float = 0) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Extracts time series data for a given property."""
        prop = resolve_attr_key(name, energy_aliases)
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
        """Returns average (and optionally std) of a property."""
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
            Total number of molecules in system.

        Returns
        -------
        float
            Average heat capacity in the requested units.
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
        """Runs gmx energy to extract a property to .xvg file."""
        subprocess.run(
            ["gmx", "energy", "-f", str(edr_path), "-o", str(output_file)],
            input=f"{prop}\n",
            text=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )
        self.logger.info(f"Extracted '{prop}' from {edr_path} to {output_file}")


