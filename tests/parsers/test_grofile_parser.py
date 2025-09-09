"""
Unit tests for GroFileParser functionality and error handling.

This suite verifies:
- Correct parsing of atomic coordinates and molecule counts
- Accurate box volume calculation from valid box line
- Robust error handling for invalid file suffixes, missing files, and malformed box lines

Uses a minimal .gro file with two water molecules and a valid box line.
"""

from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from kbkit.parsers.gro_file import GroFileParser  # Adjust import path as needed


# Sample minimal .gro content (properly formatted for MDAnalysis)
def gro_atom_line(resid, resname, atomname, atomnum, x, y, z):
    """Helper to format a single atom line in .gro file."""
    return f"{resid:5d}{resname:<5}{atomname:>5}{atomnum:5d}{x:8.3f}{y:8.3f}{z:8.3f}"


# Correctly formatted .gro content for MDAnalysis (each line ends with a single \n, not a literal '\\n')
SAMPLE_GRO_CONTENT = (
    "Test GRO file\n"
    "6\n"
    f"{gro_atom_line(1, 'WAT', 'O', 1, 0.000, 0.000, 0.000)}\n"
    f"{gro_atom_line(1, 'WAT', 'H', 2, 0.100, 0.000, 0.000)}\n"
    f"{gro_atom_line(1, 'WAT', 'H1', 3, 0.000, 0.100, 0.000)}\n"
    f"{gro_atom_line(2, 'WAT', 'O', 4, 1.000, 1.000, 1.000)}\n"
    f"{gro_atom_line(2, 'WAT', 'H', 5, 1.100, 1.000, 1.000)}\n"
    f"{gro_atom_line(2, 'WAT', 'H1', 6, 1.000, 1.100, 1.000)}\n"
    "   1.00000   1.00000   1.00000\n"
)


def create_temp_gro_file(content=SAMPLE_GRO_CONTENT, name="test.gro"):
    """
    Create a temporary .gro file with specified content and name.

    Parameters
    ----------
    content : str
        GRO file content to write.
    name : str
        Filename to use for the temporary file.

    Returns
    -------
    Tuple[Path, TemporaryDirectory]
        Path to the created file and the temporary directory object.
    """
    temp_dir = TemporaryDirectory()
    gro_path = Path(temp_dir.name) / name
    gro_path.write_text(content)
    return gro_path, temp_dir


def test_valid_gro_file_parsing():
    """
    Test that GroFileParser correctly parses atom entries and computes electron counts.

    Asserts that 'WAT' is detected and has a positive electron count.
    """
    gro_path, temp_dir = create_temp_gro_file()
    parser = GroFileParser(str(gro_path))

    assert isinstance(parser.electron_count, dict)
    assert "WAT" in parser.electron_count
    assert parser.electron_count["WAT"] > 0

    temp_dir.cleanup()


def test_box_volume_calculation():
    """
    Test that GroFileParser correctly calculates box volume from valid box line.

    Asserts that the volume matches the expected value within tolerance.
    """
    gro_path, temp_dir = create_temp_gro_file()
    parser = GroFileParser(str(gro_path))

    volume = parser.calculate_box_volume()
    assert pytest.approx(volume, 0.001) == 1.0

    temp_dir.cleanup()


def test_invalid_suffix():
    """
    Test that GroFileParser raises ValueError for files with invalid suffix.

    Uses a '.txt' file to trigger the error.
    """
    gro_path, _ = create_temp_gro_file(name="test.txt")
    with pytest.raises(ValueError, match=f"Suffix .gro does not match file suffix: {gro_path.suffix}"):
        GroFileParser(gro_path)


def test_missing_file():
    """Test that GroFileParser raises FileNotFoundError for nonexistent file."""
    with pytest.raises(FileNotFoundError):
        GroFileParser("nonexistent.gro")


def test_invalid_box_line():
    """
    Test that GroFileParser raises ValueError when box line is malformed.

    Replaces valid box line with a non-numeric string to simulate failure.
    """
    bad_content = SAMPLE_GRO_CONTENT.replace("   1.00000   1.00000   1.00000\n", "invalid box")
    gro_path, temp_dir = create_temp_gro_file(content=bad_content)

    with pytest.raises(ValueError, match="GRO unitcell has neither 3 nor 9 entries"):
        GroFileParser(str(gro_path))
    temp_dir.cleanup()
