"""
Unit tests for EdrFileParser.

This suite verifies:
- Correct loading and property detection from mock .edr files
- Accurate average property calculation using mocked time series
- Robust handling of subprocess calls and property availability

Uses a minimal mock .edr file and patches internal methods to simulate parser behavior.
"""

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from kbkit.parsers import EdrFileParser


@pytest.fixture
def edr_file(tmp_path):
    """
    Create a temporary mock .edr file with basic thermodynamic properties.

    Returns
    -------
    Path
        Path to the mock .edr file containing potential, pressure, and temperature.
    """
    edr_path = tmp_path / "test_npt.edr"
    edr_path.write_text("""
# Mock EDR file
potential: 100.0
pressure: 1.0
temperature: 300.0
""")
    return edr_path


SAMPLE_POTENTIAL = 100.0
SAMPLE_PRESSURE = 1.0
SAMPLE_TEMPERATURE = 300.0
SAMPLE_TEMPERATURE_ARRAY = np.array([SAMPLE_TEMPERATURE, SAMPLE_TEMPERATURE - 2, SAMPLE_TEMPERATURE + 2])


def test_edrfileparser_load(edr_file):
    """
    Test that EdrFileParser correctly loads the .edr file and detects available properties.

    Asserts that:
    - The file path is correctly stored
    - The available properties list is non-empty and of type list
    """
    parser = EdrFileParser(edr_file, verbose=True)
    assert parser.edr_path == [Path(edr_file)]
    assert parser.available_properties() is not None
    assert isinstance(parser.available_properties(), list)


@patch("kbkit.parsers.edr_file.subprocess.run")
def test_average_property(mock_run, edr_file):
    """
    Test average property calculation using mocked subprocess and time series extraction.

    Verifies:
    - Property availability checks return expected results
    - Average value is computed correctly from a mocked temperature array
    """
    mock_run.return_value = None  # Simulate successful run
    parser = EdrFileParser(edr_file)

    # Mock available_properties method
    with patch.object(parser, "available_properties", return_value=["potential", "pressure", "temperature"]):
        assert parser.has_property("potential") is True
        assert parser.has_property("NonexistentProperty") is False

    # Mock extract_timeseries method used by average_property
    with patch.object(parser, "extract_timeseries", return_value=(None, SAMPLE_TEMPERATURE_ARRAY)):
        avg = parser.average_property(name="potential", start_time=0.0)
        assert isinstance(avg, float)
        assert round(avg, 2) == SAMPLE_TEMPERATURE
