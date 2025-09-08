"""Provides lightweight file I/O utilities for reading and writing data in kbkit."""

from pathlib import Path

from natsort import natsorted

from kbkit.utils.validation import validate_path


def find_files(path: str | Path, suffixes: list[str], ensemble: str, exclude: tuple = ("init", "eqm")) -> list[str]:
    """
    Discover and filter files in a directory based on suffixes, ensemble name, and exclusion patterns.

    Performs a two-stage search:
    1. Broad match by suffix and exclusion substrings
    2. Refined match by ensemble name (if multiple candidates remain)

    Parameters
    ----------
    path : str or Path
        Directory to search for files.
    suffixes : list[str]
        File extensions to include (e.g., [".gro", ".xtc"]).
    ensemble : str
        Substring used to refine matches (e.g., "npt", "nvt").
    exclude : tuple[str], optional
        Substrings to exclude from filenames (default: ("init", "eqm")).

    Returns
    -------
    list[str]
        Sorted list of matching file paths as strings.

    Notes
    -----
    - Uses `validate_path` to ensure input is a readable directory.
    - Applies natural sorting for reproducibility across platforms.
    - Ensemble filtering is applied only if multiple candidates are found.
    """
    path = validate_path(path)  # Ensure it's a valid Path object

    # stage 1: broad match
    candidates = [f for f in path.iterdir() if f.suffix in suffixes and not any(ex in f.name for ex in exclude)]

    # stage 2: ensemble refinement
    ensemble_matches = [f for f in candidates if ensemble in f.name]
    final = ensemble_matches if ensemble_matches else candidates

    # natural sort for reproducibility
    sorted_files = natsorted(str(f) for f in final)

    # check that files are readable and not hidden
    readable_files = [f for f in sorted_files if Path(f).is_file() and not Path(f).name.startswith(".")]
    return readable_files
