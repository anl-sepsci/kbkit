"""
Provides lightweight file I/O utilities for reading and writing data in kbkit.

These helpers abstract common patterns like safe file opening, extension checking, and encoding handling.
They are not tied to any specific file format or domain logic.
"""

from pathlib import Path 
from natsort import natsorted

from kbkit.utils.validation import validate_path

def find_files(
    path: str | Path,
    suffixes: list[str],
    ensemble: str,
    exclude: tuple = ("init", "eqm")
) -> list[str]:
    """
    Stage 1: Find all files matching suffixes and excluding noisy patterns.
    Stage 2: If multiple found, filter by ensemble name.

    Parameters
    ----------
    path : str or Path
        Directory to search.
    suffixes : list[str]
        File extensions to match.
    ensemble : str
        Ensemble name to refine results.
    exclude : tuple[str], optional
        Substrings to exclude from filenames.

    Returns
    -------
    list[str]
        Sorted list of resolved file paths.
    """
    
    path = validate_path(path)  # Ensure it's a valid Path object

    # stage 1: broad match
    candidates = [
        f for f in path.iterdir()
        if f.suffix in suffixes
        and not any(ex in f.name for ex in exclude)
    ]

    # stage 2: ensemble refinement
    ensemble_matches = [f for f in candidates if ensemble in f.name]
    final = ensemble_matches if ensemble_matches else candidates

    return natsorted(str(f) for f in final)


