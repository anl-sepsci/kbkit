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

import numpy as np
import pytest

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
    g = np.ones_like(r) + np.random.normal(0, 0.001, size=r.shape)  # nearly flat g(r)
    content = "\n".join(f"{ri:.3f} {gi:.5f}" for ri, gi in zip(r, g, strict=False))

    with tempfile.NamedTemporaryFile("w+", suffix=".xvg", delete=False) as f:
        f.write(content)
        f.flush()
        yield f.name


def test_rdf_parser_reads_data(mock_rdf_file):
    """
    Test that RDFParser correctly reads RDF data and sets internal arrays.

    Asserts
    -------
    - r and g are numpy arrays
    - r and g have matching shapes
    - rmin is less than rmax
    """
    parser = RDFParser(mock_rdf_file)
    assert isinstance(parser.r, np.ndarray)
    assert isinstance(parser.g, np.ndarray)
    assert parser.r.shape == parser.g.shape
    assert parser.rmin < parser.rmax


def test_rdf_parser_convergence(mock_rdf_file):
    """
    Test that RDFParser detects convergence for nearly flat synthetic RDF data.

    Asserts
    -------
    - convergence_check returns True
    """
    parser = RDFParser(mock_rdf_file)
    assert parser.convergence_check() is True


def test_rdf_parser_r_mask_bounds(mock_rdf_file):
    """
    Test that r_mask correctly filters radial distances between rmin and rmax.

    Asserts
    -------
    - All masked r values are within [rmin, rmax]
    """
    parser = RDFParser(mock_rdf_file)
    mask = parser.r_mask
    assert np.all(parser.r[mask] >= parser.rmin)
    assert np.all(parser.r[mask] <= parser.rmax)


def test_rdf_parser_extract_mols():
    """
    Test molecule name extraction from RDF filename.

    Asserts
    -------
    - extract_mols returns expected molecule names
    """
    filename = "rdf_Na_Cl.xvg"
    mols = RDFParser.extract_mols(filename, ["Na", "Cl", "H2O"])
    assert set(mols) == {"Na", "Cl"}


def test_rdf_parser_rmin_setter(mock_rdf_file):
    """
    Test rmin setter validation logic.

    Asserts
    -------
    - Valid rmin is accepted
    - Invalid rmin raises ValueError
    - Non-numeric rmin raises TypeError
    """
    parser = RDFParser(mock_rdf_file)
    valid_rmin = parser.rmax - 0.5
    parser.rmin = valid_rmin
    assert parser.rmin == valid_rmin

    with pytest.raises(ValueError, match=r"Lower bound .* exceeds rmax"):
        parser.rmin = parser.rmax + 1

    with pytest.raises(TypeError, match=r"Value must be float or int"):
        parser.rmin = "invalid"
