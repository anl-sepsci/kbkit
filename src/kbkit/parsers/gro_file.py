"""Parses a GROMACS .gro file to extract residue electron counts and box volume."""

from functools import cached_property

import MDAnalysis
import numpy as np

from kbkit.utils.chem import get_atomic_number
from kbkit.utils.logging import get_logger
from kbkit.utils.validation import validate_path


class _NullGroFileParser:
    # objective is to mimic the interface but returns empty values
    @property
    def gro_path(self):
        return ""

    @property
    def electron_count(self):
        return {}

    def calculate_box_volume(self):
        return np.nan


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
        self._universe = MDAnalysis.Universe(self.gro_path)

    @property
    def residues(self) -> MDAnalysis.core.groups.ResidueGroup:
        """mda.core.groups.ResidueGroup: Residues in the .gro file."""
        return self._universe.residues

    @property
    def atom_counts(self) -> dict[str, dict[str, int]]:
        """dict[str, dict[str, int]]: Dictionary of residue names and their atom type counts."""
        atoms_counts = {}
        for res in self.residues:
            if res.resname in atoms_counts:
                continue
            unique_atoms, counts = np.unique(res.atoms.types, return_counts=True)
            atoms_counts[res.resname] = {atom: int(count) for atom, count in zip(unique_atoms, counts, strict=False)}
        return atoms_counts

    @cached_property
    def electron_count(self) -> dict[str, int]:
        """dict[str, int]: Dictionary of residue types and their total electron count."""
        residue_electrons = {}
        for resname, atom_dict in self.atom_counts.items():
            total_electrons = sum(get_atomic_number(atom) * count for atom, count in atom_dict.items())
            residue_electrons[resname] = total_electrons
        return residue_electrons

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
        box_A = np.asarray(self._universe.dimensions[:3])
        box_nm = box_A / 10.0  # convert from Angstroms to nm
        volume_nm3 = np.prod(box_nm)
        self.logger.info("Successfully parsed .gro file for box dimensions.")
        return float(volume_nm3)
