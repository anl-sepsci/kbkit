"""
Unit tests for SystemProperties interface and property resolution.

These tests validate:
- File registry detection for GROMACS input files (.top, .gro, .edr)
- Retrieval of thermodynamic properties via the `get()` method
- Correct delegation to internal methods and parsers using mocks

Mocked data includes:
- Minimal .gro and .top content for a 2-molecule water system
- Sample values for heat capacity, volume, enthalpy, and potential energy
"""

from unittest.mock import MagicMock, patch

import pytest

from kbkit.core.system_properties import SystemProperties


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

SAMPLE_TOP_CONTENT = """
[ system ]
Test

[ molecules ]
WAT     2
"""

SAMPLE_HEAT_CAPACITY = 42.0
SAMPLE_VOLUME = 1.0
SAMPLE_ENTHALPY = 123.45
SAMPLE_POTENTIAL_AVG = 100.0
SAMPLE_POTENTIAL_STD = 2.5


@pytest.fixture
def system(tmp_path):
    """
    Create a mock system directory with minimal GROMACS files.

    Returns
    -------
    SystemProperties
        Initialized with mock .top, .gro, and .edr files for a 2-molecule water system.
    """
    syspath = tmp_path
    (syspath / "test_npt.top").write_text(SAMPLE_TOP_CONTENT)
    (syspath / "test_npt.gro").write_text(SAMPLE_GRO_CONTENT)
    (syspath / "test_npt.edr").write_text("Mock edr content")
    return SystemProperties(system_path=str(syspath), ensemble="npt", verbose=True)


def test_file_registry(system):
    """
    Validate that the file registry correctly detects required input files.

    Asserts presence of 'top', 'gro', and 'edr' keys in the registry.
    """
    registry = system.file_registry
    assert "top" in registry
    assert "gro" in registry
    assert "edr" in registry


@patch("kbkit.parsers.edr_file.EdrFileParser.heat_capacity", return_value=SAMPLE_HEAT_CAPACITY)
def test_get_heat_capacity(mock_heat_capacity, system):
    """
    Test retrieval of heat capacity via the `get()` method.

    Verifies correct value and that the parser is called with expected molecule count.
    """
    cap = system.get("heat_capacity")
    assert isinstance(cap, float)
    assert cap == SAMPLE_HEAT_CAPACITY
    mock_heat_capacity.assert_called_once_with(nmol=2)


@patch.object(SystemProperties, "_get_average_property", return_value=SAMPLE_VOLUME)
def test_get_volume(mock_avg, system):
    """
    Test retrieval of volume via the `get()` method.

    Verifies correct value and delegation to `_get_average_property`.
    """
    vol = system.get("volume")
    assert isinstance(vol, float)
    assert vol == SAMPLE_VOLUME
    mock_avg.assert_called_once_with(name="volume", start_time=0.0, units="", return_std=False)


@patch.object(SystemProperties, "_enthalpy", return_value=SAMPLE_ENTHALPY)
def test_get_enthalpy(mock_enthalpy, system):
    """
    Test retrieval of enthalpy via the `get()` method.

    Verifies correct value and delegation to `enthalpy()` method.
    """
    H = system.get("enthalpy", start_time=0.0)
    assert isinstance(H, float)
    assert H == SAMPLE_ENTHALPY
    mock_enthalpy.assert_called_once_with(start_time=0.0, units="")


@patch.object(SystemProperties, "_get_average_property", return_value=(SAMPLE_POTENTIAL_AVG, SAMPLE_POTENTIAL_STD))
def test_get_with_std(mock_avg, system):
    """
    Test retrieval of potential energy with standard deviation via `get(std=True)`.

    Verifies correct tuple output and delegation to `_get_average_property`.
    """
    val, std = system.get("potential", std=True)
    assert isinstance(val, float)
    assert isinstance(std, float)
    assert val == SAMPLE_POTENTIAL_AVG
    assert std == SAMPLE_POTENTIAL_STD
    mock_avg.assert_called_once_with(name="potential", start_time=0.0, units="", return_std=True)


@patch.object(SystemProperties, "_get_average_property", return_value=SAMPLE_VOLUME)
def test_get_with_units_and_start_time(mock_avg, system):
    """
    Test `get()` with custom units and start_time.

    Verifies correct delegation and argument forwarding.
    """
    val = system.get("volume", start_time=5.0, units="nm^3")
    assert val == SAMPLE_VOLUME
    mock_avg.assert_called_once_with(name="volume", start_time=5.0, units="nm^3", return_std=False)


@patch.object(SystemProperties, "_get_average_property", return_value=(SAMPLE_POTENTIAL_AVG, SAMPLE_POTENTIAL_STD))
def test_get_with_std_and_units(mock_avg, system):
    """
    Test `get()` with std=True and custom units.

    Verifies correct delegation and unit handling.
    """
    val, std = system.get("potential", std=True, units="kJ/mol")
    assert val == SAMPLE_POTENTIAL_AVG
    assert std == SAMPLE_POTENTIAL_STD
    mock_avg.assert_called_once_with(name="potential", start_time=0.0, units="kJ/mol", return_std=True)


def test_get_unknown_property(system):
    """
    Test that requesting an unknown property raises a ValueError.

    Validates error handling for unsupported property names.
    """
    with pytest.raises(KeyError):
        system.get("nonexistent_property")


def test_volume_fallback_from_gro(system):
    """
    Test fallback volume estimation from .gro file when .edr lacks volume data.

    Verifies correct value and fallback logic.
    """
    system.energy.has_property = MagicMock(return_value=False)
    system.structure.calculate_box_volume = MagicMock(return_value=1.0)
    val = system.get("volume")
    assert val == 1.0


def test_volume_fallback_failure(system):
    """
    Test error raised when fallback volume estimation from .gro file fails.

    Simulates ValueError from `calculate_box_volume`.
    """
    system.energy.has_property = MagicMock(return_value=False)
    system.structure.calculate_box_volume = MagicMock(side_effect=ValueError("bad box"))
    with pytest.raises(ValueError, match="Alternative volume calculation from .gro file failed"):
        system.get("volume")


def test_missing_edr_property(system):
    """
    Test error raised when requested property is missing from .edr file.

    Validates error propagation from `_get_average_property`.
    """
    system.energy.has_property = MagicMock(return_value=False)
    with pytest.raises(ValueError, match="does not contain property"):
        system.get("pressure")
