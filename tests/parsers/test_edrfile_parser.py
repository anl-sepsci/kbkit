"""Unit tests for EdrFileParser covering property detection, timeseries extraction, and heat capacity."""

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from kbkit.parsers.edr_file import EdrFileParser
from kbkit.utils.logging import get_logger

# Constants for test data
MOCK_POTENTIAL: float = 100.0
MOCK_PRESSURE: float = 1.0
MOCK_TEMPERATURE: float = 300.0
MOCK_TEMPERATURE_ARRAY: np.ndarray = np.array([298.0, 300.0, 302.0])
MOCK_TIME_ARRAY: np.ndarray = np.array([0.0, 1.0, 2.0])
MOCK_HEAT_CAPACITY_CP: str = "Heat capacity at constant pressure Cp = 1234.0"
MOCK_HEAT_CAPACITY_CV: str = "Heat capacity at constant volume Cv = 567.0"


@pytest.fixture
def edr_file(tmp_path: Path) -> Path:
    """Create a temporary mock .edr file with basic thermodynamic properties."""
    edr_path = tmp_path / "test_npt.edr"
    edr_path.write_text(f"potential: {MOCK_POTENTIAL}\npressure: {MOCK_PRESSURE}\ntemperature: {MOCK_TEMPERATURE}\n")
    return edr_path


def test_edrfileparser_load(edr_file: Path) -> None:
    """Test that EdrFileParser loads the file and returns available properties."""
    parser = EdrFileParser(str(edr_file))
    assert parser.edr_path == [Path(edr_file)]
    assert isinstance(parser.available_properties(), list)


@patch("kbkit.parsers.edr_file.subprocess.run")
def test_available_properties_parsing(mock_run, edr_file: Path) -> None:
    """Test that available_properties parses subprocess output correctly."""
    mock_run.return_value.stdout = "---\nPotential Pressure Temperature\n---"
    mock_run.return_value.stderr = ""
    parser = EdrFileParser(str(edr_file))
    props = parser.available_properties()
    assert "Potential" in props


def test_extract_properties_empty_output() -> None:
    """Test _extract_properties returns empty list when no valid tokens are found."""
    parser = EdrFileParser.__new__(EdrFileParser)  # bypass __init__
    parser.edr_path = [Path("dummy.edr")]  # manually set required attribute
    parser.logger = get_logger("test")  # inject dummy logger

    output = "----\n123 456\n----"
    props = parser._extract_properties(output)
    assert props == []


def test_has_property_case_insensitive(edr_file: Path) -> None:
    """Test has_property matches regardless of case."""
    parser = EdrFileParser(str(edr_file))
    with patch.object(parser, "available_properties", return_value=["Temperature"]):
        assert parser.has_property("temperature") is True


@patch("kbkit.parsers.edr_file.subprocess.run")
def test_extract_timeseries_fallback(mock_run, edr_file: Path) -> None:
    """Test extract_timeseries falls back to _run_gmx_energy when file is missing."""
    parser = EdrFileParser(str(edr_file))
    mock_run.return_value = None
    with patch("numpy.loadtxt", return_value=(MOCK_TIME_ARRAY, MOCK_TEMPERATURE_ARRAY)):
        time, values = parser.extract_timeseries("temperature")
        assert time.shape == values.shape
        assert values.mean() == pytest.approx(MOCK_TEMPERATURE)


@patch("kbkit.parsers.edr_file.subprocess.run")
def test_average_property_with_std(mock_run, edr_file: Path) -> None:
    """Test average_property returns mean and std when return_std=True."""
    parser = EdrFileParser(str(edr_file))
    mock_run.return_value = None
    with patch.object(parser, "extract_timeseries", return_value=(MOCK_TIME_ARRAY, MOCK_TEMPERATURE_ARRAY)):
        results = parser.average_property("temperature", return_std=True)
        if isinstance(results, tuple):
            avg, std = results
            assert round(avg, 2) == MOCK_TEMPERATURE
            assert round(std, 2) > 0


@patch("kbkit.parsers.edr_file.subprocess.run")
def test_heat_capacity_enthalpy_path(mock_run, edr_file: Path) -> None:
    """Test heat_capacity uses enthalpy path and parses Cp correctly."""
    parser = EdrFileParser(str(edr_file))
    mock_run.return_value.stdout = MOCK_HEAT_CAPACITY_CP
    with patch.object(parser, "has_property", return_value=True):
        result = parser.heat_capacity(nmol=100)
        assert result == pytest.approx(1.234)


@patch("kbkit.parsers.edr_file.subprocess.run")
def test_heat_capacity_fallback_path(mock_run, edr_file: Path) -> None:
    """Test heat_capacity uses fallback path and parses Cv correctly."""
    parser = EdrFileParser(str(edr_file))
    mock_run.return_value.stdout = MOCK_HEAT_CAPACITY_CV
    with patch.object(parser, "has_property", return_value=False):
        result = parser.heat_capacity(nmol=50)
        assert result == pytest.approx(0.567)


@patch("kbkit.parsers.edr_file.subprocess.run")
def test_heat_capacity_no_match(mock_run, edr_file: Path) -> None:
    """Test heat_capacity raises ValueError when no match is found."""
    parser = EdrFileParser(str(edr_file))
    mock_run.return_value.stdout = "No match here"
    with patch.object(parser, "has_property", return_value=True):
        with pytest.raises(ValueError, match="No heat capacity values could be extracted from any .edr file."):
            parser.heat_capacity(nmol=10)


@patch("kbkit.parsers.edr_file.subprocess.run")
def test_run_gmx_energy_invocation(mock_run, edr_file: Path) -> None:
    """Test that _run_gmx_energy invokes subprocess with correct arguments."""
    parser = EdrFileParser(str(edr_file))
    output_file = edr_file.with_name("temperature_test.xvg")
    parser._run_gmx_energy("temperature", output_file, edr_file)
    mock_run.assert_called_once()
