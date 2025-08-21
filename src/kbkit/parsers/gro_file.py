"""Parses a GROMACS .gro file to extract residue electron counts and box volume."""

from collections import defaultdict
from functools import cached_property

from kbkit.parsers.gro_atom import GroAtomParser
from kbkit.utils.chem import get_atomic_number
from kbkit.utils.logging import get_logger
from kbkit.utils.validation import validate_path

class GroFileParser:
    def __init__(self, gro_path: str, verbose: bool = False) -> None:
        """
        Parses a single GROMACS .gro file to compute valence electron counts and box volume.

        Parameters
        ----------
        gro_path: str
            Path to the .gro file.
        verbose: bool, optional
            If True, enables detailed logging output.
        """
        self.gro_path = validate_path(gro_path, suffix=".gro")
        self.logger = get_logger(f"{__name__}.{self.__class__.__name__}", verbose=verbose)
        self.logger.info(f"Validated .gro file: {self.gro_path}")

    def get_atom_parser(self) -> GroAtomParser:
        self.logger.debug(f"Initializing GroAtomParser for file: {self.gro_path}")
        return GroAtomParser(self.gro_path)
    
    def count_electrons(self) -> dict[str, int]:
        """
        Compute total valence electrons per residue type.

        Returns
        -------
        dict[str, int]
            Mapping of residue names to total valence electrons.
        """
        self.logger.debug("Starting electron count per residue.")
        parser = self.get_atom_parser()
        residue_electrons = defaultdict(int)

        for _, res_name, atom_name in parser:
            element = ''.join(filter(str.isalpha, atom_name)).capitalize()
            try:
                electrons = get_atomic_number(element)
                residue_electrons[res_name] += electrons
            except Exception as e:
                self.logger.warning(f"Invalid element '{element}' from atom '{atom_name}' in residue '{res_name}': {e}")

        self.logger.info("Completed electron count.")
        return dict(residue_electrons)

    @cached_property
    def electron_count(self) -> dict[str, int]:
        """dict[str, int]: Dictionary of residue types and their total valence electrons."""
        return self.count_electrons()

    def calculate_box_volume(self) -> float:
        """
        Calculates the box volume from the last line of a GROMACS .gro file.

        Parameters
        ----------
        gro_path : str or Path
            Path to the .gro file.

        Returns
        -------
        float
            Box volume in nanometers cubed (nm³).
        """

        last_line = self.gro_path.read_text().splitlines()[-1].strip()
        parts = last_line.split()

        if len(parts) < 3:
            self.logger.error(f"Invalid box line: {last_line!r}")
            raise ValueError(f"Invalid box line: {last_line!r}")

        try:
            x, y, z = map(float, parts[:3])
            self.logger.info(f"Successfully parsed .gro file for box dimensions.")
            return x * y * z
        except ValueError as e:
            self.logger.error(f"Failed to parse box dimensions: {last_line!r}")
            raise ValueError(f"Failed to parse box dimensions: {last_line!r}") from e
