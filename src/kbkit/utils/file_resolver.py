"""Find files in a system corresponding to a certain property."""

from pathlib import Path
import logging 
from kbkit.utils.io import find_files
from kbkit.utils.logging import get_logger

class FileResolver:
    """Resolves scientific file roles to actual paths based on suffix and ensemble."""

    # role-to-suffix mapping
    ROLE_SUFFIXES: dict[str, list[str]] = {
        "energy": [".edr"],
        "structure": [".gro", ".pdb"],
        "topology": [".top"],
        "trajectory": [".xtc"],
        "log": [".log"],
        "index": [".ndx"],
        "metadata": [".json", ".yaml"],
    }

    def __init__(self, system_path: Path, ensemble: str, logger: logging.Logger | None = None) -> None:
        self.system_path = system_path
        self.ensemble = ensemble
        self.logger = logger or get_logger(f"{__name__}.{self.__class__.__name__}", verbose=False)

    def get_file(self, role: str) -> str | list[str]:
        """Return the first matching file for a given semantic role."""
        suffixes = self.ROLE_SUFFIXES.get(role)
        if not suffixes:
            raise ValueError(f"Unknown file role: {role}")
        
        files = find_files(self.system_path, suffixes, self.ensemble)
        if not files:
            raise FileNotFoundError(f"No file found for role '{role}'.")
        
        self.logger.debug(f"Resolved {role} => {Path(files[0]).name}")
        return files[0]
    
    def get_all(self, role: str) -> list[str]:
        """Return all matching files for a given role."""
        suffixes = self.ROLE_SUFFIXES.get(role)
        if not suffixes:
            raise ValueError(f"Unknown file role: {role}")
        
        return find_files(self.system_path, suffixes, self.ensemble)
    
    def has_file(self, role: str) -> bool:
        """Check if a file exists for the given role."""
        try:
            self.get_file(role)
            return True
        except FileNotFoundError:
            return False

    
