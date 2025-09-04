"""Semantic file resolution for scientific systems."""

import logging
from pathlib import Path
from typing import ClassVar

from kbkit.utils.io import find_files
from kbkit.utils.logging import get_logger


class FileResolver:
    """Resolves scientific file roles to actual paths based on suffix and ensemble."""

    # role-to-suffix mapping
    ROLE_SUFFIXES: ClassVar[dict[str, list[str]]] = {
        "energy": [".edr"],
        "structure": [".gro", ".pdb"],
        "topology": [".top"],
        "trajectory": [".xtc"],
        "log": [".log"],
        "index": [".ndx"],
        "metadata": [".json", ".yaml"],
    }

    def __init__(self, system_path: Path, ensemble: str, logger: logging.Logger | None = None) -> None:
        """
        Initialize a FileResolver for a given system directory and ensemble.

        Parameters
        ----------
        system_path : Path
            Root directory containing simulation files.
        ensemble : str
            Ensemble name used to refine file matching (e.g., "npt", "nvt").
        logger : logging.Logger, optional
            Custom logger instance. If None, a default logger is created.
        """
        self.system_path = system_path
        self.ensemble = ensemble
        self.logger = logger or get_logger(f"{__name__}.{self.__class__.__name__}", verbose=False)

    def get_file(self, role: str) -> str:
        """
        Return the first matching file for a given semantic role.

        Parameters
        ----------
        role : str
            Semantic role to resolve (e.g., "structure", "energy").

        Returns
        -------
        str
            Path to the first matching file.

        Notes
        -----
        - Uses suffix heuristics and ensemble filtering to identify the best match.
        """
        suffixes = self.ROLE_SUFFIXES.get(role)
        if not suffixes:
            raise ValueError(f"Unknown file role: {role}")

        files = find_files(self.system_path, suffixes, self.ensemble)
        if len(files) == 0:
            if role == "structure":  # .gro only used for electron counting; don't throw error
                return ""
            else:
                raise FileNotFoundError(f"No file found for role '{role}'.")
        else:
            self.logger.debug(f"Resolved {role} => {Path(files[0]).name}")
            return files[0]

    def get_all(self, role: str) -> list[str]:
        """
        Return all matching files for a given semantic role.

        Parameters
        ----------
        role : str
            Semantic role to resolve (e.g., "trajectory", "log").

        Returns
        -------
        list[str]
            List of matching file paths.

        Notes
        -----
        - Applies suffix filtering and ensemble refinement.
        - Useful for workflows that require multiple files per role.
        """
        suffixes = self.ROLE_SUFFIXES.get(role)
        if not suffixes:
            print(f"ValueError: Unknown role: '{role}'.")
            return []
        else:
            return find_files(self.system_path, suffixes, self.ensemble)

    def has_file(self, role: str) -> bool:
        """
        Check whether a file exists for the given semantic role.

        Parameters
        ----------
        role : str
            Semantic role to check (e.g., "topology", "index").

        Returns
        -------
        bool
            True if a matching file exists, False otherwise.

        Notes
        -----
        - Internally calls `get_file` and catches `FileNotFoundError`.
        - Designed for conditional logic in config loaders or analysis pipelines.
        """
        try:
            self.get_file(role)
            return True
        except FileNotFoundError:
            return False
