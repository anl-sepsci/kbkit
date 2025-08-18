"""
Contains generic input validation utilities used across kbkit modules.

These functions are stateless and reusable, designed to enforce type, value, and structural constraints
without introducing domain-specific logic.
"""

from pathlib import Path 

def validate_file(path: str | Path, suffix: str) -> Path:
    path = Path(path)
    # first validate suffix
    if path.suffix != suffix:
        raise ValueError(f"Suffix {suffix} does not match file suffix: {path.suffix}")
    # check if file is a file and exists
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")
    # special checks for certain file types
    if suffix == ".gro":
        if len(path.read_text().splitlines()) < 3:
            raise ValueError(f"File '{path}' is too short to be a valid .gro file.")
    return path


