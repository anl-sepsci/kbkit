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
        valid_path = validate_path(path, suffix=".gro")
        try:
            self._universe = MDAnalysis.Universe(valid_path)
        except Exception:  # only reformat if mda doesn't work.
            self._universe = MDAnalysis.Universe(self._resolve_gro_formatting(valid_path))

    def _resolve_gro_formatting(self, input_file: Path) -> str:
        with open(input_file, "r") as f:
            lines = f.readlines()

        # The .gro format:
        # 1. Title (Line 1)
        # 2. Number of atoms (Line 2)
        # 3. Atom data (Line 3 to N+2)
        # 4. Box vectors (Last line)

        reformatted = [lines[0], lines[1]]  # Keep header and atom count

        # Process only the atom lines (skip header/footer)
        for line in lines[2:-1]:
            parts = line.split()
            MAGIC_SIX = 6
            if len(parts) < MAGIC_SIX:
                continue  # Skip malformed empty lines

            # Handle cases where residue ID might be mixed with residue name
            try:
                res_id = int(parts[0])
                res_name = parts[1]
                atom_name = parts[2]
                atom_id = int(parts[3])
                x, y, z = float(parts[4]), float(parts[5]), float(parts[6])
            except ValueError:
                # If first part contains non-numeric characters, extract numeric part
                import re
                res_id_match = re.search(r'\d+', parts[0])
                if res_id_match:
                    res_id = int(res_id_match.group())
                    # Extract residue name from the same field
                    res_name_match = re.search(r'[A-Za-z]+', parts[0])
                    res_name = res_name_match.group() if res_name_match else "UNK"
                    atom_name = parts[1]
                    atom_id = int(parts[2])
                    x, y, z = float(parts[3]), float(parts[4]), float(parts[5])
                else:
                    raise ValueError(f"Could not parse residue ID from: '{parts[0]}'")

            # Strict GROMACS fixed-column format:
            # %5d%-5s%5s%5d%8.3f%8.3f%8.3f
            new_line = f"{res_id:>5}{res_name:<5}{atom_name:>5}{atom_id:>5}{x:>8.3f}{y:>8.3f}{z:>8.3f}\n"
            reformatted.append(new_line)

        reformatted.append(lines[-1])  # Add box vectors back

        output_file = Path(str(input_file).strip(".gro") + "_reformatted.gro")
        with open(output_file, "w") as f:
            f.writelines(reformatted)

        return str(output_file)

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
