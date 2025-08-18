import pytest
from kbkit.properties import SystemProperties

@pytest.fixture
def system(tmp_path):
    # Simulate a directory with mock GROMACS files
    syspath = tmp_path
    (syspath / "test_npt.top").write_text("[ system ]\nTest")
    (syspath / "test_npt.gro").write_text("Generated gro content")
    (syspath / "test_npt.edr").write_text("Mock edr content")
    return SystemProperties(syspath=str(syspath), ensemble="npt", verbose=True)

def test_file_registry(system):
    registry = system.file_registry
    assert "top" in registry
    assert "gro" in registry
    assert "edr" in registry

def test_get_volume(system):
    vol = system.get("volume")
    assert isinstance(vol, float)

def test_get_heat_capacity(system):
    cap = system.get("heat_capacity")
    assert isinstance(cap, float)

def test_get_enthalpy(system):
    H = system.get("enthalpy", start_time=0.0)
    assert isinstance(H, float)

def test_get_with_std(system):
    val, std = system.get("Potential", std=True)
    assert isinstance(val, float)
    assert isinstance(std, float)
