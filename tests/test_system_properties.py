import pytest
from unittest.mock import patch
from kbkit.properties import SystemProperties

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


@pytest.fixture
def system(tmp_path):
    # Simulate a directory with mock GROMACS files
    syspath = tmp_path
    (syspath / "test_npt.top").write_text(SAMPLE_TOP_CONTENT)
    (syspath / "test_npt.gro").write_text(SAMPLE_GRO_CONTENT)
    (syspath / "test_npt.edr").write_text("Mock edr content")
    return SystemProperties(syspath=str(syspath), ensemble="npt", verbose=True)

def test_file_registry(system):
    registry = system.file_registry
    assert "top" in registry
    assert "gro" in registry
    assert "edr" in registry

@patch("kbkit.properties.edr_file_parser.EdrFileParser.heat_capacity", return_value=42.0)
def test_get_heat_capacity(mock_heat_capacity, system):
    cap = system.get("heat_capacity")
    assert isinstance(cap, float)
    assert cap == 42.0
    mock_heat_capacity.assert_called_once_with(nmol=2)

@patch.object(SystemProperties, "_get_average_property", return_value=1.0)
def test_get_volume(mock_avg, system):
    vol = system.get("volume")
    assert isinstance(vol, float)
    assert vol == 1.0
    mock_avg.assert_called_once_with(name="volume", start_time=0.0, units="", return_std=False)

@patch.object(SystemProperties, "enthalpy", return_value=123.45)
def test_get_enthalpy(mock_enthalpy, system):
    H = system.get("enthalpy", start_time=0.0)
    assert isinstance(H, float)
    assert H == 123.45
    mock_enthalpy.assert_called_once_with(start_time=0.0, units="")

@patch.object(SystemProperties, "_get_average_property", return_value=(100.0, 2.5))
def test_get_with_std(mock_avg, system):
    val, std = system.get("potential", std=True)
    assert isinstance(val, float)
    assert isinstance(std, float)
    assert val == 100.0
    assert std == 2.5
    mock_avg.assert_called_once_with(name="potential", start_time=0.0, units="", return_std=True)

