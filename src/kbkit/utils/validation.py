"""
Contains generic input validation utilities used across kbkit modules.

These functions are stateless and reusable, designed to enforce type, value, and structural constraints
without introducing domain-specific logic.
"""

from pathlib import Path 
import os

def validate_path(path: str | Path, suffix: str = "") -> Path:
    # check type of path
    if not isinstance(path, (str, Path)):
        raise TypeError(f"Expected a string path, got {type(path).__name__}: {path}")
    
    # get path object; resolves symlinks to normalize path and anchor to root
    path = Path(path).resolve()
        
    is_dir = path.is_dir()
    is_file = path.is_file()

    # for suffix; path type must be a file
    if suffix:
        # must be file
        if not is_file:
            raise FileNotFoundError(f"Path is not a file: {path}")
        # first validate suffix
        if path.suffix != suffix:
            raise ValueError(f"Suffix {suffix} does not match file suffix: {path.suffix}")
        # special checks for certain file types
        if suffix == ".gro":
            if len(path.read_text().splitlines()) < 3:
                raise ValueError(f"File '{path}' is too short to be a valid .gro file.")

    # if not suffix; then path should be dir
    else:
        if not is_dir:
            raise ValueError(f"Path is not a directory: {path}")
        
    # check that path can be accessed and read
    if not os.access(path, os.R_OK):
        raise PermissionError(f"Cannot read files in path: {path}")
    
    return path

