"""
Test suite for RDFParser.

Validates core functionality of RDFParser, including:
- File reading and data extraction
- Convergence detection logic
- Radial distance masking
- Molecule name extraction from filenames
- rmin setter validation
"""

import tempfile

import matplotlib as mpl
import numpy as np
import pytest

mpl.use("Agg")  # prevent GUI during tests

from kbkit.parsers.rdf_file import RDFParser


@pytest.fixture
def mock_rdf_file():
    """
    Create a temporary RDF file with synthetic data for testing.

    Returns
    -------
    str
        Path to the temporary RDF file.
    """
    r = np.linspace(0.1, 5.0, 100)
    g = np.ones_like(r) + np.random.normal(0, 0.001, size=r.shape)
    content = "\n".join(f"{ri:.3f} {gi:.5f}" for ri, gi in zip(r, g, strict=False))

    with tempfile.NamedTemporaryFile("w+", suffix=".xvg", delete=False) as f:
        f.write(content)
        f.flush()
        yield f.name


def test_rdf_parser_read_failure(tmp_path):
    """Test that RDFParser raises an error for malformed RDF file."""
    bad_file = tmp_path / "bad.xvg"
    bad_file.write_text("not a number\n1.0 two\n")
    with pytest.raises(ValueError, match=f"Failed to parse RDF data from '{bad_file}'"):
        RDFParser(str(bad_file))


def test_rdf_parser_reads_data(mock_rdf_file):
    """Test that RDFParser correctly reads RDF data and sets internal arrays."""
    parser = RDFParser(mock_rdf_file)
    assert isinstance(parser.r, np.ndarray)
    assert isinstance(parser.g, np.ndarray)
    assert parser.r.shape == parser.g.shape
    assert parser.rmin < parser.rmax


def test_rdf_parser_convergence(mock_rdf_file):
    """Test that RDFParser detects convergence for nearly flat synthetic RDF data."""
    parser = RDFParser(mock_rdf_file)
    assert parser.convergence_check() is True


def test_rdf_parser_convergence_failure(monkeypatch, mock_rdf_file):
    """Test convergence failure path when RDF is noisy and slope is too high."""
    parser = RDFParser(mock_rdf_file)

    # artificially increase noise
    parser._g += np.linspace(0, 0.1, len(parser._g))

    result = parser.convergence_check(convergence_threshold=1e-6)
    assert result is False
    assert parser.rmin == pytest.approx(parser.rmax - 0.2)


def test_rdf_parser_r_mask_bounds(mock_rdf_file):
    """Test that r_mask correctly filters radial distances between rmin and rmax."""
    parser = RDFParser(mock_rdf_file)
    mask = parser.r_mask
    assert np.all(parser.r[mask] >= parser.rmin)
    assert np.all(parser.r[mask] <= parser.rmax)


def test_rdf_parser_extract_mols():
    """Test molecule name extraction from RDF filename."""
    filename = "rdf_Na_Cl.xvg"
    mols = RDFParser.extract_mols(filename, ["Na", "Cl", "H2O"])
    assert set(mols) == {"Na", "Cl"}


def test_rdf_parser_extract_mols_empty():
    """Test extract_mols returns empty list when no matches are found."""
    filename = "rdf_unknown.xvg"
    mols = RDFParser.extract_mols(filename, ["Na", "Cl"])
    assert mols == []


def test_rdf_parser_rmin_setter(mock_rdf_file):
    """Test rmin setter validation logic."""
    parser = RDFParser(mock_rdf_file)
    valid_rmin = parser.rmax - 0.5
    parser.rmin = valid_rmin
    assert parser.rmin == valid_rmin

    with pytest.raises(ValueError, match=r"Lower bound .* exceeds rmax"):
        parser.rmin = parser.rmax + 1

    with pytest.raises(TypeError, match=r"Value must be float or int"):
        parser.rmin = "invalid"


def test_rdf_parser_plot(mock_rdf_file, tmp_path):
    """Test that RDFParser.plot() runs and saves a file without error."""
    parser = RDFParser(mock_rdf_file)
    parser.plot(save_dir=str(tmp_path))
    output_file = tmp_path / (parser.rdf_file[:-4] + ".png")
    assert output_file.exists()
