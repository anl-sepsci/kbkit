"""Parses a topology file to extract composition information."""

import re
from enum import Enum, auto
from functools import cached_property
from pathlib import Path

import MDAnalysis as mda
import numpy as np
import periodictable

from kbkit.utils.validation import validate_path


class TopologyFormat(Enum):
    """Formatting for various topology types."""

    LAMMPS = auto()
    GROMACS = auto()
    TOP = auto()


class TopologyParser:
    """
    Parses topology files to get molecules present and their counts.

    Parameters
    ----------
    path : str
        Path to the topology file.
    """

    def __init__(self, path: str | Path) -> None:
        self.filepath = validate_path(path, suffix=Path(path).suffix)

    @property
    def _topology_format(self) -> TopologyFormat:
        """Get topology format."""
        # first check filetypes
        if self.filepath.suffix.lower() in (".top"):
            return TopologyFormat.TOP
        elif self.filepath.suffix.lower() in (".gro", ".tpr", ".xtc"):
            return TopologyFormat.GROMACS
        elif self.filepath.suffix.lower() in (".lmp", ".lammpsdump"):
            return TopologyFormat.LAMMPS
        # if filetypes don't work, then default to reading universe objects
        try:
            resnames = self._universe.residues.resnames
            # LAMMPS resnames are often empty strings, '?', or all identical
            if (len(resnames) == 0) or (np.all(r in ("", "?", "UNK", "UNKNOWN") for r in resnames)):
                return TopologyFormat.LAMMPS
            return TopologyFormat.GROMACS
        except (mda.exceptions.NoDataError, AttributeError):
            return TopologyFormat.LAMMPS

    @cached_property
    def _universe(self) -> mda.Universe:
        """Create MDAnalysis Universe object based on TopologyFormat."""
        # for TOP; mda.Universe cannot be created, return None
        if self._topology_format == TopologyFormat.TOP:
            return None

        # try to get mda.Universe object if GRO/LMP type
        input_fmt_map = {TopologyFormat.GROMACS: "GRO", TopologyFormat.LAMMPS: "DATA"}
        input_fmt = input_fmt_map[self._topology_format]
        try:
            return mda.Universe(
                self.filepath, topology_format=input_fmt, format=input_fmt, atom_style="id type q x y z"
            )
        except ValueError:
            return None

    @staticmethod
    def _is_valid_molecule_name(name: str) -> bool:
        # Allow letters, numbers, underscores, and hyphens
        return bool(re.match(r"^[A-Za-z0-9_\-]{2,50}$", name))

    @staticmethod
    def _is_valid_count(count: str) -> bool:
        # check if string is valid number
        return count.isdigit()

    @staticmethod
    def extract_molecules_from_top(top_file: str | Path) -> dict[str, int]:
        """Read the GROMACS topology file (.top) and returns a dictionary of molecule names and counts.

        Returns
        -------
        dict[str, int]
            Dictionary containing molecules present and their number.
        """
        fpath = validate_path(top_file, suffix=".top")
        lines = fpath.read_text().splitlines()
        molecules = {}
        in_molecules_section = False

        MAX_MOLECULE_PARTS = 2

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
                    continue

                molecule_name, count_str = parts[0], parts[1]

                if not TopologyParser._is_valid_molecule_name(molecule_name):
                    continue

                if not TopologyParser._is_valid_count(count_str):
                    continue

                molecules[molecule_name] = int(count_str)

        if not molecules:
            raise ValueError("No molecules found in topology file.")

        return molecules

    @cached_property
    def molecule_count(self) -> dict[str, int]:
        """dict[str, int]: Dictionary of molecules present and their corresponding numbers."""
        # parse .top file if present
        if self._topology_format == TopologyFormat.TOP:
            return self.extract_molecules_from_top(top_file=self.filepath)

        # for all others, use MDAnalysis
        try:
            resnames, counts = np.unique(self._universe.residues.resnames, return_counts=True)
        except Exception:
            resnames, counts = np.unique(self._universe.residues.types, return_counts=True)
        return {res: int(count) for res, count in zip(resnames, counts, strict=False)}

    @property
    def molecules(self) -> list[str]:
        """list[str]: Names of molecules present."""
        return list(self.molecule_count.keys())

    @property
    def total_molecules(self) -> int:
        """int: Total number of molecules present."""
        return sum(self.molecule_count.values())

    @cached_property
    def _electron_lookup(self) -> tuple[np.ndarray, np.ndarray]:
        """Build sorted mass → electron count (Z) arrays from periodictable, called once at module import. Skips neutron (number == 0)."""
        pairs = []
        for el in periodictable.elements:
            if el.number == 0:
                continue
            try:
                pairs.append((float(el.mass), int(el.number)))
            except Exception:
                continue

        pairs.sort(key=lambda x: x[0])
        element_masses = np.array([p[0] for p in pairs], dtype=np.float64)
        element_electrons = np.array([p[1] for p in pairs], dtype=np.int32)

        return element_masses, element_electrons

    @property
    def _element_masses(self) -> np.ndarray:
        """Returns element masses from _electron_lookup."""
        return self._electron_lookup[0]

    @property
    def _element_electrons(self) -> np.ndarray:
        """Returns element electrons from _electron_lookup."""
        return self._electron_lookup[1]

    def _masses_to_electrons_vectorized(
        self,
        masses: np.ndarray,
        tolerance: float = 0.5,
    ) -> np.ndarray:
        """
        Directly map an array of atomic masses to electron counts.

        Operates only on UNIQUE masses (O(U log N) where U = unique atom types),
        then broadcasts back to the full array — ideal for MD systems where
        U << total atoms (e.g., 4 unique types across 50,000 atoms).

        Parameters
        ----------
        masses    : Array of atomic masses in amu, shape (n_atoms,)
        tolerance : Maximum allowed mass difference

        Returns
        -------
        np.ndarray of int32 electron counts, shape (n_atoms,)
        -1 indicates no element found within tolerance
        """
        masses = np.asarray(masses, dtype=np.float64)

        # ── Only resolve unique masses ────────────────────────────────────────
        unique_masses, inverse = np.unique(masses, return_inverse=True)

        # bisect over all unique masses at once
        indices = np.searchsorted(self._element_masses, unique_masses)
        n_known = len(self._element_masses)

        unique_electrons: np.ndarray = np.full(len(unique_masses), -1, dtype=np.int32)

        # type: ignore[call-overload]  # numpy scalar is iterable at runtime, mypy stubs are overly strict
        for i, (mass, idx) in enumerate(zip(unique_masses, indices, strict=False)): # type: ignore[call-overload]
            best_delta = tolerance
            best_electrons = -1

            for j in (idx - 1, idx):
                if 0 <= j < n_known:
                    delta = abs(mass - self._element_masses[j])
                    if delta < best_delta:
                        best_delta = delta
                        best_electrons = int(self._element_electrons[j])

            unique_electrons[i] = best_electrons

        # Broadcast unique results back to full atom array
        return unique_electrons[inverse]

    def _get_electron_counts(
        self,
        tolerance: float = 0.5,
        ionic: bool = True,
    ) -> np.ndarray:
        """
        Get per-atom electron counts from a LAMMPS MDAnalysis universe.

        Parameters
        ----------
        u         : mda.Universe
        tolerance : Mass matching tolerance in amu
        ionic     : If True, adjust Z by formal charge (requires charge data)

        Returns
        -------
        np.ndarray of shape (n_atoms,) with electron counts
        """
        # ── Get masses ────────────────────────────────────────────────────────
        try:
            masses = self._universe.atoms.masses
            if np.all(masses == 0.0):
                raise ValueError("All masses are zero")
        except ValueError as e:
            raise ValueError(f"Cannot resolve masses from universe: {e}\nEnsure your .data file has a Masses section.") from e

        # ── Mass → electrons (vectorized, unique-first) ───────────────────────
        electrons = self._masses_to_electrons_vectorized(masses, tolerance)

        unresolved = np.sum(electrons == -1)
        if unresolved > 0:
            bad_masses = np.unique(masses[electrons == -1])
            raise ValueError(
                f"Could not resolve electron counts for masses: {bad_masses}\n"
                f"Increase tolerance or check your Masses section."
            )

        # ── Adjust for ionic charges ──────────────────────────────────────────
        if ionic:
            try:
                charges = self._universe.atoms.charges
                if not np.all(charges == 0.0):
                    electrons = electrons.astype(np.float64) - charges
            except Exception:
                pass  # no charge data — return neutral Z

        return electrons

    @cached_property
    def electron_count(self) -> dict[str, int]:
        """dict[str, int]: Dictionary of residue types and their total electron count."""
        if self._topology_format == TopologyFormat.TOP:
            return {}

        atom_electrons = self._get_electron_counts(tolerance=0.5, ionic=True)
        atom_types = self._universe.atoms.types
        resnums = self._universe.atoms.resnums

        # get unique residues
        unique_resnums, resnum_counts = np.unique(resnums, return_counts=True)

        # If everything is lumped into 1 residue, or if EVERY atom has its own unique residue ID,
        # then it's a completely separate/ionic system.
        is_molecular = len(unique_resnums) > 1 and not np.all(resnum_counts == 1)

        single_res_electrons = {}

        # case 1: completely ionic
        if not is_molecular:
            # Every unique atom type acts as its own individual residue species
            unique_types = np.unique(atom_types)
            for t in unique_types:
                mask = atom_types == t
                single_res_electrons[t] = atom_electrons[mask][0]

        # case 2: complex molecular system
        else:
            # Extract the residue names array from your universe
            try:
                resnames = self._universe.atoms.resnames
            except (mda.exceptions.NoDataError, AttributeError):
                resnames = self._universe.atoms.types

            # Sort everything by residue/molecule ID to make them contiguous in memory
            sort_idx = np.argsort(resnums)
            sorted_resnums = resnums[sort_idx]
            sorted_electrons = atom_electrons[sort_idx]
            sorted_resnames = resnames[sort_idx]

            # Identify boundaries where one molecule ends and the next begins
            split_indices = np.where(sorted_resnums[:-1] != sorted_resnums[1:])[0] + 1

            # Split the arrays into sub-arrays (one sub-array per molecule)
            mol_electrons_split = np.split(sorted_electrons, split_indices)
            mol_resnums_split = np.split(sorted_resnums, split_indices)
            mol_resnames_split = np.split(sorted_resnames, split_indices)

            # Define generic placeholder names to catch unprovided/default topology names
            GENERIC_NAMES = {"MOL", "UNK", "RES", "SYSTEM", "DEFAULT", ""}

            # Process the molecular splits
            for mol_elec, mol_rnums, mol_rnames in zip(mol_electrons_split, mol_resnums_split, mol_resnames_split, strict=False):
                rname = str(mol_rnames[0])
                rnum = int(mol_rnums[0])
                mol_total_electrons = np.sum(mol_elec)

                # Rule: Use resname if it is a meaningful provided name (e.g., 'SOL', 'IL_CAT')
                if rname not in GENERIC_NAMES:
                    # Group and accumulate by the residue type name
                    if rname not in single_res_electrons:
                        single_res_electrons[rname] = mol_total_electrons

                # Fallback: If it's a generic LAMMPS default name, use the unique Molecule ID/Resnum
                else:
                    single_res_electrons[str(rnum)] = mol_total_electrons

        return single_res_electrons

    @cached_property
    def box_volume(self) -> float:
        """float: Compute box volume (nm^3)."""
        if self._topology_format == TopologyFormat.TOP:
            return np.nan

        volume_ang3 = self._universe.trajectory.ts.volume
        return float(volume_ang3) / 1000  # return volume in nm^3
