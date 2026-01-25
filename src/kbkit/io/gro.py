"""Parses a GROMACS .gro file to extract residue electron counts and box volume."""

from functools import cached_property
from pathlib import Path
from typing import ClassVar

import MDAnalysis
import numpy as np
from rdkit.Chem import GetPeriodicTable

from kbkit.utils.validation import validate_path


class GroParser:
    """
    Parse a single GROMACS .gro file to compute valence electron counts and box volume.

    Parameters
    ----------
    gro_path: str
        Path to the .gro file.
    """

    MAX_SYMBOL_LENGTH: ClassVar = 2

    def __init__(self, path: str | Path) -> None:
        self.filepath = validate_path(path, suffix=".gro")
        self._universe = MDAnalysis.Universe(self.filepath)

    @property
    def residues(self) -> MDAnalysis.core.groups.ResidueGroup:
        """MDAnalysis.core.groups.ResidueGroup: Residues in the .gro file."""
        return self._universe.residues

    @property
    def molecule_count(self) -> dict[str, int]:
        """dict[str, int]: Unique molecule residues and corresponding counts."""
        resnames, counts = np.unique(self.residues.resnames, return_counts=True)
        mol_dict = {res: int(count) for res, count in zip(resnames, counts, strict=False)}
        return mol_dict

    @property
    def molecules(self) -> list[str]:
        """list[str]: Names of molecules present."""
        return list(self.molecule_count.keys())

    @property
    def total_molecules(self) -> int:
        """int: Total number of molecules present."""
        return sum(self.molecule_count.values())

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
            total_electrons = sum(self.get_atomic_number(atom) * count for atom, count in atom_dict.items())
            residue_electrons[resname] = total_electrons
        return residue_electrons

    @cached_property
    def box_volume(self) -> float:
        """float: Compute box volume (nm^3) from the last line of a GROMACS .gro file."""
        box_A = np.asarray(self._universe.dimensions[:3])
        box_nm = box_A / 10.0  # convert from Angstroms to nm
        volume_nm3 = np.prod(box_nm)
        return float(volume_nm3)

    @staticmethod
    def is_valid_element(symbol: str) -> bool:
        """
        Check if a string is a valid chemical element symbol.

        Parameters
        ----------
        symbol : str
            Element symbol to validate (e.g., 'C', 'Na', 'Cl').

        Returns
        -------
        bool
            True if valid element, False otherwise.
        """
        if not symbol or not isinstance(symbol, str):
            return False

        symbol = symbol.strip().capitalize()
        if len(symbol) > GroParser.MAX_SYMBOL_LENGTH:
            symbol = symbol[:2]

        ptable = GetPeriodicTable()
        return ptable.GetAtomicNumber(symbol) > 0

    @staticmethod
    def get_atomic_number(symbol: str) -> int:
        """
        Return atomic number of a valid element symbol.

        Parameters
        ----------
        symbol : str
            Valid element symbol (e.g., 'C', 'Na', 'Cl').

        Returns
        -------
        int
            Atomic number of the element.
        """
        symbol = symbol.strip().capitalize()
        # checks that symbol is a valid element
        if GroParser.is_valid_element(symbol):
            ptable = GetPeriodicTable()
            return ptable.GetAtomicNumber(symbol)
        else:
            raise ValueError(f"Symbol '{symbol}' is not a valid element.")

