"""Parses a GROMACS .top file to extract molecule names and their counts from the [ molecules ] section."""

import re
from functools import cached_property
from pathlib import Path
from typing import Any

import numpy as np

from kbkit.utils.validation import validate_path

MAX_MOLECULE_PARTS = 2


class TopParser:
    """
    Parses GROMACS topology file to get molecules present and their counts.

    Parameters
    ----------
    top_path : str
        Path to the topology (.top) file.
    """

    def __init__(self, path: str | Path) -> None:
        self.filepath = validate_path(path, suffix=".top")
        self.skipped_lines: list[Any] = []

    def _is_valid_molecule_name(self, name: str) -> bool:
        # Allow letters, numbers, underscores, and hyphens
        return bool(re.match(r"^[A-Za-z0-9_\-]{2,50}$", name))

    def _is_valid_count(self, count: str) -> bool:
        # check if string is valid number
        return count.isdigit()

    def parse(self) -> None:
        """Read the topology file and returns a dictionary of molecule names and counts.

        Returns
        -------
        dict
            Dictionary containing molecules present and their number.
        """
        lines = self.filepath.read_text().splitlines()
        molecules = {}
        in_molecules_section = False

        # extract molecule name and numbers from file
        for _line_num, original_line in enumerate(lines, start=1):
            # Remove comments (anything after a semicolon) and leading/trailing whitespace
            line = original_line.split(";")[0].strip()
            if not line:
                continue  # Skip empty lines

            # search for 'molecules' line
            if line.lower().startswith("[ molecules ]"):
                in_molecules_section = True
                continue

            if in_molecules_section:
                if line.startswith("["):
                    break  # Stop parsing if we encounter another section

                # Split the line by spaces and tabs, filtering out empty strings
                parts = re.split(r"\s+", line)

                if len(parts) < MAX_MOLECULE_PARTS:
                    self.skipped_lines.append((original_line, "Missing molecule name or count"))
                    continue

                molecule_name, count_str = parts[0], parts[1]

                if not self._is_valid_molecule_name(molecule_name):
                    self.skipped_lines.append((original_line, "Invalid molecule name format"))
                    continue

                if not self._is_valid_count(count_str):
                    self.skipped_lines.append((original_line, "Invalid molecule count"))
                    continue

                molecules[molecule_name] = int(count_str)

        if not molecules:
            raise ValueError("No molecules found in topology file.")

        self._molecule_count = molecules

    def report_skipped(self) -> None:
        """Print a summary of lines that were skipped during parsing, including the line content and the reason for skipping."""
        if self.skipped_lines:
            print("Skipped lines during parsing:")
            for line, reason in self.skipped_lines:
                print(f"  Line: '{line}' => Reason: {reason}")

    @cached_property
    def molecule_count(self) -> dict[str, int]:
        """dict[str, int]: Dictionary of molecules present and their corresponding numbers."""
        if "_molecule_count" not in self.__dict__:
            self.parse()
        return self._molecule_count

    @property
    def molecules(self) -> list[str]:
        """list[str]: Names of molecules present."""
        return list(self.molecule_count.keys())

    @property
    def total_molecules(self) -> int:
        """int: Total number of molecules present."""
        return sum(self.molecule_count.values())

    @property
    def electron_count(self) -> dict[str, int]:
        """dict: Empty dict of electron counts."""
        return {}

    @property
    def box_volume(self) -> float:
        """float: NaN value for box volume."""
        return np.nan
