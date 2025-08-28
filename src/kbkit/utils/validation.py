"""Contains generic input validation utilities used across kbkit modules."""

import os
from pathlib import Path

GRO_MIN_LENGTH = 3


def validate_path(path: str | Path, suffix: str = "") -> Path:
    """
    Validate and normalize a file or directory path, optionally checking for a specific file suffix.

    Parameters
    ----------
    path : str or Path
        Input path to validate. Can be a string or pathlib.Path object.
    suffix : str, optional
        Expected file suffix (e.g., ".gro"). If provided, the path must be a file with this suffix.

    Returns
    -------
    Path
        Resolved and validated pathlib.Path object.

    Notes
    -----
    - Resolves symlinks and anchors the path to the filesystem root.
    - If `suffix == ".gro"`, performs a minimal length check to ensure file validity.
    """
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
            if len(path.read_text().splitlines()) < GRO_MIN_LENGTH:
                raise ValueError(f"File '{path}' is too short to be a valid .gro file.")

    # if not suffix; then path should be dir
    elif not is_dir:
        raise ValueError(f"Path is not a directory: {path}")

    # check that path can be accessed and read
    if not os.access(path, os.R_OK):
        raise PermissionError(f"Cannot read files in path: {path}")

    return path
