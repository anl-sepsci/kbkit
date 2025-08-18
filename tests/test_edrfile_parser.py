import pytest
from kbkit.properties import EdrFileParser

@pytest.fixture
def edr_file(tmp_path):
    # Replace with actual test .edr file path or mock setup
    return tmp_path / "test_npt.edr"

def test_edrfileparser_load(edr_file):
    parser = EdrFileParser(edr_file, verbose=True)
    assert parser.edr_path == str(edr_file)
    assert isinstance(parser.available_properties, list)

def test_has_property(edr_file):
    parser = EdrFileParser(edr_file)
    assert parser.has_property("Potential") is True
    assert parser.has_property("NonexistentProperty") is False

def test_average_property(edr_file):
    parser = EdrFileParser(edr_file)
    avg = parser.average_property(name="Potential", start_time=0.0)
    assert isinstance(avg, float)
