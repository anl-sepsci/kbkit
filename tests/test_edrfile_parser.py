import pytest
from pathlib import Path
from unittest.mock import patch
import numpy as np

from kbkit.parsers import EdrFileParser

@pytest.fixture
def edr_file(tmp_path):
    # Replace with actual test .edr file path or mock setup
    edr_path = tmp_path / "test_npt.edr"
    edr_path.write_text("""
# Mock EDR file
potential: 100.0
pressure: 1.0
temperature: 300.0                        
""")
    return edr_path

def test_edrfileparser_load(edr_file):
    parser = EdrFileParser(edr_file, verbose=True)
    assert parser.edr_path == [Path(edr_file)]
    assert parser.available_properties() is not None
    assert isinstance(parser.available_properties(), list)

@patch("kbkit.parsers.edr_file.subprocess.run")
def test_average_property(mock_run, edr_file):
    mock_run.return_value = None  # Simulate successful run
    parser = EdrFileParser(edr_file)

    # Mock available_properties method
    with patch.object(parser, "available_properties", return_value=["potential", "pressure", "temperature"]):
        assert parser.has_property("potential") is True
        assert parser.has_property("NonexistentProperty") is False

    # Mock extract_timeseries method used by average_property
    with patch.object(parser, "extract_timeseries", return_value=(None, np.array([100.0, 102.0, 98.0]))):
        avg = parser.average_property(name="potential", start_time=0.0)
        assert isinstance(avg, float)
        assert round(avg, 2) == 100.0
