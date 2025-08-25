"""
Unit tests for SystemProperties interface and property resolution.

These tests validate:
- File registry detection for GROMACS input files (.top, .gro, .edr)
- Retrieval of thermodynamic properties via the `get()` method
- Correct delegation to internal methods and parsers using mocks

Mocked data includes:
- Minimal .gro and .top content for a 2-molecule water system
- Sample values for heat capacity, volume, enthalpy, and potential energy

Tests focus on reproducibility, semantic clarity, and interface behavior.
"""

from unittest.mock import patch

import pytest

from kbkit.core import SystemProperties

# Sample minimal .gro content
SAMPLE_GRO_CONTENT = """Test GRO file
   3
    1WAT    O     1   0.000   0.000   0.000
    1WAT    H     2   0.100   0.000   0.000
    1WAT    H1    3   0.000   0.100   0.000
    2WAT    O     4   1.000   1.000   1.000
    2WAT    H     5   1.100   1.000   1.000
    2WAT    H1    6   1.000   1.100   1.000
   1.00000  1.00000  1.00000
"""

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
    Test that the file registry correctly detects required input files.

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


@patch.object(SystemProperties, "enthalpy", return_value=SAMPLE_ENTHALPY)
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
