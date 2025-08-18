"""Centralize reusable checks."""

from pathlib import Path 

def validate_file(path: str | Path, suffix: str) -> Path:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")
    if path.suffix != suffix:
        raise ValueError(f"Expected a {suffix} file, got: {path.name}")
    # special checks for certain file types
    if suffix == ".gro":
        if len(path.read_text().splitlines()) < 3:
            raise ValueError(f"File '{path}' is too short to be a valid .gro file.")
    return path


