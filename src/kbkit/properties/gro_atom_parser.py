"""Create an iterable object over valid atom lines in GROMACS .gro file."""

import re
from kbkit.utils import get_logger, validate_file

class GroAtomParser:
    """
    Parses atom lines from a GROMACS .gro file and yields structured records.

    Parameters
    ----------
    gro_path : str or Path
        Path to the .gro file.
    """

    def __init__(self, gro_path: str, verbose: bool  = False) -> None:
        self.gro_path = validate_file(gro_path, suffix=".gro")
        self.verbose = verbose
        self.logger = get_logger(f"{__name__}.{self.__class__.__name__}", verbose=verbose)

    def __iter__(self):
        """Yields tuples of (mol_idx, res_name, atom_name) for each atom line."""
        lines = self.gro_path.read_text().splitlines()

        try:
            n_atoms = int(lines[1].strip())
        except ValueError as e:
            self.logger.error(f"Invalid atom count in line 2: {lines[1]!r}")
            raise ValueError(f"Invalid atom count in line 2: {lines[1]!r}") from e

        atom_lines = lines[2 : 2 + n_atoms]

        for i, line in enumerate(atom_lines, start=3):
            try:
                parts = line.strip().split()
                if len(parts) < 2:
                    self.logger.error("Line too short")
                    raise ValueError("Line too short")

                fused = parts[0]
                atom_name = parts[1]

                match = re.match(r"(\d+)([A-Za-z]+)", fused)
                if not match:
                    self.logger.error(f"Invalid fused field: {fused!r}")
                    raise ValueError(f"Invalid fused field: {fused!r}")

                res_idx = int(match.group(1))
                res_name = match.group(2)

                self.logger.debug(f"[Line {i}] Parsed residue({res_idx}): {res_name} => {atom_name}")
                yield res_idx, res_name, atom_name

            except Exception as e:
                self.logger.warning(f"Failed to parse atom line {i}: {line!r} — {e}")
                continue
