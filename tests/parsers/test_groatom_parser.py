"""
Unit tests for GroAtomParser, which parses atom lines from GROMACS .gro files.

This suite covers:
- Validation of atom count line
- Handling of malformed atom lines (too short, invalid fused field)
- Graceful skipping of invalid entries
- Logging behavior for parse failures
"""

import pytest

from kbkit.parsers.gro_atom import GroAtomParser  # Adjust import path as needed


def test_invalid_atom_count_line(tmp_path):
    """Raises ValueError when atom count line is non-numeric."""
    gro_path = tmp_path / "bad.gro"
    gro_path.write_text("Header\nnot_a_number\n1WAT    O     1   0.000   0.000   0.000\n1.0 1.0 1.0")

    with pytest.raises(ValueError, match="Invalid atom count in line 2"):
        list(GroAtomParser(gro_path))


def test_atom_line_too_short(tmp_path):
    """Skips atom lines with too few parts."""
    gro_path = tmp_path / "short.gro"
    gro_path.write_text("Header\n1\nbadline\n1.0 1.0 1.0")

    atoms = list(GroAtomParser(gro_path))
    assert atoms == []


def test_invalid_fused_field(tmp_path):
    """Skips atom lines with invalid fused field format."""
    gro_path = tmp_path / "badfused.gro"
    gro_path.write_text("Header\n1\nXYZ O 1 0.0 0.0 0.0\n1.0 1.0 1.0")

    atoms = list(GroAtomParser(gro_path))
    assert atoms == []


def test_mixed_valid_and_invalid_lines(tmp_path):
    """Parses only valid atom lines when mixed with malformed ones."""
    gro_path = tmp_path / "mixed.gro"
    gro_path.write_text("Header\n3\nbadline\nXYZ O 1 0.0 0.0 0.0\n1WAT O 1 0.0 0.0 0.0\n1.0 1.0 1.0")

    atoms = list(GroAtomParser(gro_path))
    assert atoms == [(1, "WAT", "O")]


def test_logs_warning_for_skipped_lines(tmp_path, caplog):
    """Logs warnings for lines that fail to parse."""
    gro_path = tmp_path / "warn.gro"
    gro_path.write_text("Header\n1\nXYZ O 1 0.0 0.0 0.0\n1.0 1.0 1.0")

    with caplog.at_level("WARNING"):
        atoms = list(GroAtomParser(gro_path, verbose=True))
        assert atoms == []
        assert "Failed to parse atom line" in caplog.text
