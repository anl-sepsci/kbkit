"""
Unit tests for string formatting utilities and FileResolver.

Covers:
- Unit file_resolver and LaTeX formatting
- Edge cases in string-to-LaTeX conversion
- File role file_resolver with suffix filtering
- Error handling for unknown roles and missing files
"""

from pathlib import Path
from unittest.mock import patch

import pytest

from kbkit.utils.file_resolver import FileResolver
from kbkit.utils.format import format_unit_str, resolve_units

# ---------- String Formatting Tests ----------


def test_resolve_units_returns_requested():
    """Returns requested unit if provided."""
    assert resolve_units("mol", "default") == "mol"


def test_resolve_units_falls_back_to_default():
    """Returns default unit if requested is empty."""
    assert resolve_units("", "nm") == "nm"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("mol/nm**3", "$mol\\text{ }\\mathrm{nm^{-3}}$"),
        ("mol/nm3", "$mol\\text{ }\\mathrm{nm^{-3}}$"),
        ("mol/nm^{2}", "$mol\\text{ }\\mathrm{nm^{-2}}$"),
        ("mol_nm", "$mol_{nm}$"),
        ("mol**2_nm", "$mol^{2}_{nm}$"),
    ],
)
def test_format_unit_str_converts_to_latex(text, expected):
    """Ensure format_unit_str converts unit strings to correct LaTeX math format."""
    result = format_unit_str(text)
    assert result == expected
    assert result.startswith("$")
    assert result.endswith("$")


# ---------- FileResolver Tests ----------


@pytest.fixture
def mock_system_dir(tmp_path: Path) -> Path:
    """Create a mock system directory with sample files."""
    (tmp_path / "npt_structure.gro").write_text("structure")
    (tmp_path / "npt_energy.edr").write_text("energy")
    (tmp_path / "npt_trajectory.xtc").write_text("trajectory")
    return tmp_path


@patch("kbkit.utils.file_resolver.find_files", return_value=["/mock/path/npt_structure.gro"])
def test_get_file_returns_first_match(mock_find, mock_system_dir):
    """Returns first matching file for a role."""
    resolver = FileResolver(mock_system_dir, "npt")
    result = resolver.get_file("structure")
    assert result.endswith(".gro")


@patch("kbkit.utils.file_resolver.find_files", return_value=["/mock/path/npt_trajectory.xtc"])
def test_get_all_returns_all_matches(mock_find, mock_system_dir):
    """Returns all matching files for a role."""
    resolver = FileResolver(mock_system_dir, "npt")
    result = resolver.get_all("trajectory")
    assert isinstance(result, list)
    assert result[0].endswith(".xtc")


@patch("kbkit.utils.file_resolver.find_files", return_value=["/mock/path/npt_energy.edr"])
def test_has_file_returns_true(mock_find, mock_system_dir):
    """Returns True if file exists for role."""
    resolver = FileResolver(mock_system_dir, "npt")
    assert resolver.has_file("energy") is True


def test_get_file_raises_value_error_for_unknown_role(mock_system_dir):
    """Raises ValueError for unknown semantic role."""
    resolver = FileResolver(mock_system_dir, "npt")
    with pytest.raises(ValueError, match="Unknown file role"):
        resolver.get_file("unknown_role")
