"""Parses a GROMACS .gro file to extract residue electron counts and box volume."""

from collections import defaultdict
from functools import cached_property

from kbkit.parsers.gro_atom import GroAtomParser
from kbkit.utils.chem import get_atomic_number
from kbkit.utils.logging import get_logger
from kbkit.utils.validation import validate_path

MIN_BOX_LINE_PARTS = 3


class GroFileParser:
    """
    Parse a single GROMACS .gro file to compute valence electron counts and box volume.

    Parameters
    ----------
    gro_path: str
        Path to the .gro file.
    verbose: bool, optional
        If True, enables detailed logging output.
    """

    def __init__(self, gro_path: str, verbose: bool = False) -> None:
        self.gro_path = validate_path(gro_path, suffix=".gro")
        self.logger = get_logger(f"{__name__}.{self.__class__.__name__}", verbose=verbose)
        self.logger.info(f"Validated .gro file: {self.gro_path}")

    def get_atom_parser(self) -> GroAtomParser:
        """
        Initialize and return a GroAtomParser for the current structure file.

        Returns
        -------
        GroAtomParser
            Parser instance for extracting atom-level data from the .gro file.
        """
        self.logger.debug(f"Initializing GroAtomParser for file: {self.gro_path}")
        return GroAtomParser(self.gro_path)

    def count_electrons(self) -> dict[str, int]:
        """
        Compute total electrons per unique residue type.

        Returns
        -------
        dict[str, int]
            Mapping of residue names to total electron count.
        """
        self.logger.debug("Starting electron count per residue.")
        parser = self.get_atom_parser()
        residue_electrons: dict[str, int] = defaultdict(int)
        seen_residues = {}

        for idx, res_name, atom_name in parser:
            if res_name not in seen_residues:
                seen_residues[res_name] = idx

            if idx != seen_residues[res_name]:
                continue

            element = "".join(filter(str.isalpha, atom_name)).capitalize()
            residue_electrons[res_name] += get_atomic_number(element)

        self.logger.info("Completed electron count.")
        return dict(residue_electrons)

    @cached_property
    def electron_count(self) -> dict[str, int]:
        """dict[str, int]: Dictionary of residue types and their total electrons."""
        return self.count_electrons()

    def calculate_box_volume(self) -> float:
        """
        Compute box volume from the last line of a GROMACS .gro file.

        Parameters
        ----------
        gro_path : str or Path
            Path to the .gro file.

        Returns
        -------
        float
            Box volume in nanometers cubed (nm^3).
        """
        last_line = self.gro_path.read_text().splitlines()[-1].strip()
        parts = last_line.split()

        if len(parts) < MIN_BOX_LINE_PARTS:
            self.logger.error("Box dimensions missing or invalid")
            raise ValueError("Box dimensions missing or invalid")

        try:
            x, y, z = map(float, parts[:3])
            self.logger.info("Successfully parsed .gro file for box dimensions.")
            return x * y * z
        except ValueError as e:
            self.logger.error("Box dimensions missing or invalid")
            raise ValueError("Box dimensions missing or invalid") from e
