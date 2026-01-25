"""Unit tests for RdfParser class."""

import warnings

# Suppress NumPy/SciPy compatibility warning (harmless with NumPy 2.x + SciPy 1.16+)
warnings.filterwarnings('ignore', message='numpy.ndarray size changed', category=RuntimeWarning)


from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pytest

from kbkit.io.rdf import RdfParser


@pytest.fixture
def sample_rdf_data():
    """Generate sample RDF data."""
    r = np.linspace(0, 5, 100)
    # Create g(r) that converges to 1.0
    g = 1.0 + np.exp(-r) * np.sin(10 * r) * 0.5
    # Make tail converge to 1.0
    g[-20:] = 1.0 + np.random.normal(0, 0.001, 20)
    return r, g

@pytest.fixture
def sample_rdf_file(tmp_path, sample_rdf_data):
    """Create a sample RDF .xvg file."""
    r, g = sample_rdf_data
    rdf_file = tmp_path / "rdf_test.xvg"

    content = "# RDF data\n"
    content += "@ title \"Radial Distribution Function\"\n"
    content += "@ xaxis label \"r (nm)\"\n"
    content += "@ yaxis label \"g(r)\"\n"

    for r_val, g_val in zip(r, g, strict=False):
        content += f"{r_val:.6f}    {g_val:.6f}\n"

    rdf_file.write_text(content)
    return str(rdf_file)


@pytest.fixture
def divergent_rdf_file(tmp_path):
    """Create an RDF file with non-converged tail."""
    rdf_file = tmp_path / "rdf_divergent.xvg"

    r = np.linspace(0, 5, 100)
    # Create g(r) with divergent tail
    g = 1.0 + np.exp(-r) * np.sin(10 * r) * 0.5
    g[-20:] = 1.0 + 0.1 * np.arange(20) / 20  # Linear drift

    content = "# RDF data\n"
    for r_val, g_val in zip(r, g, strict=False):
        content += f"{r_val:.6f}    {g_val:.6f}\n"

    rdf_file.write_text(content)
    return str(rdf_file)

@pytest.fixture
def mock_mplstyle():
    """Mock the mplstyle loading."""
    with patch('kbkit.io.rdf.load_mplstyle'):
        yield


class TestRdfParserInitialization:
    """Test RdfParser initialization."""

    def test_valid_rdf_file(self, sample_rdf_file, mock_mplstyle):
        """Test initialization with valid RDF file."""
        parser = RdfParser(sample_rdf_file)

        assert parser.filepath == Path(sample_rdf_file)
        assert parser.fname == "rdf_test.xvg"
        assert isinstance(parser.r, np.ndarray)
        assert isinstance(parser.g, np.ndarray)
        assert isinstance(parser.mask, np.ndarray)

    def test_file_validation(self, tmp_path, mock_mplstyle):
        """Test file path validation."""
        # Test with non-existent file
        with pytest.raises((FileNotFoundError, ValueError)):
            RdfParser(str(tmp_path / "nonexistent.xvg"))

    def test_custom_convergence_thresholds(self, sample_rdf_file, mock_mplstyle):
        """Test initialization with custom convergence thresholds."""
        parser = RdfParser(sample_rdf_file, convergence_thresholds=(1e-4, 1e-3))

        assert isinstance(parser.mask, np.ndarray)

    def test_fixed_tail_length(self, sample_rdf_file, mock_mplstyle):
        """Test initialization with fixed tail length."""
        parser = RdfParser(sample_rdf_file, tail_length=1.0)

        assert isinstance(parser.mask, np.ndarray)
        # Tail should be approximately 1.0 nm
        tail_range = parser.r_tail.max() - parser.r_tail.min()
        assert pytest.approx(tail_range, abs=0.2) == 1.0



class TestReadMethod:
    """Test _read method."""

    def test_read_valid_file(self, sample_rdf_file, mock_mplstyle):
        """Test reading valid RDF file."""
        parser = RdfParser(sample_rdf_file)

        assert len(parser.r) > 0
        assert len(parser.g) > 0
        assert len(parser.r) == len(parser.g)

    def test_read_filters_tail_noise(self, sample_rdf_file, mock_mplstyle):
        """Test that _read filters last 3 points."""
        # Count lines in file
        with open(sample_rdf_file) as f:
            data_lines = [line for line in f if not line.startswith(('#', '@'))]

        parser = RdfParser(sample_rdf_file)

        # Should have 3 fewer points than data lines
        assert len(parser.r) == len(data_lines) - 3

    def test_read_empty_file(self, tmp_path, mock_mplstyle):
        """Test reading empty file raises error."""
        empty_file = tmp_path / "empty.xvg"
        empty_file.write_text("")

        with pytest.raises((ValueError, IOError, RuntimeError, FileNotFoundError)):
            RdfParser(str(empty_file))


    def test_read_malformed_file(self, tmp_path, mock_mplstyle):
        """Test reading malformed file raises error."""
        malformed_file = tmp_path / "malformed.xvg"
        malformed_file.write_text("# Header\ninvalid data\nmore invalid\n")

        with pytest.raises((ValueError, RuntimeError, IOError)):
            RdfParser(str(malformed_file))

    def test_read_single_column_file(self, tmp_path, mock_mplstyle):
        """Test reading file with single column raises error."""
        single_col_file = tmp_path / "single.xvg"
        content = "# Header\n1.0\n2.0\n3.0\n"
        single_col_file.write_text(content)

        with pytest.raises((ValueError, RuntimeError, IOError)):
            RdfParser(str(single_col_file))


class TestExtractMolecules:
    """Test extract_molecules static method."""

    def test_extract_single_molecule(self, mock_mplstyle):
        """Test extracting single molecule from filename."""
        filename = "rdf_water_water.xvg"
        mol_list = ["water", "ethanol", "methanol"]

        result = RdfParser.extract_molecules(filename, mol_list)

        assert "water" in result
        assert len(result) == 2  # "water" appears twice

    def test_extract_multiple_molecules(self, mock_mplstyle):
        """Test extracting multiple molecules from filename."""
        filename = "rdf_water_ethanol.xvg"
        mol_list = ["water", "ethanol", "methanol"]

        result = RdfParser.extract_molecules(filename, mol_list)

        assert "water" in result
        assert "ethanol" in result
        assert len(result) == 2

    def test_extract_no_molecules(self, mock_mplstyle):
        """Test when no molecules are found."""
        filename = "rdf_unknown.xvg"
        mol_list = ["water", "ethanol", "methanol"]

        result = RdfParser.extract_molecules(filename, mol_list)

        assert len(result) == 0

    def test_extract_with_special_characters(self, mock_mplstyle):
        """Test extraction with special characters in molecule names."""
        filename = "rdf_H2O_CO2.xvg"
        mol_list = ["H2O", "CO2", "CH4"]

        result = RdfParser.extract_molecules(filename, mol_list)

        assert "H2O" in result
        assert "CO2" in result

    def test_extract_case_sensitive(self, mock_mplstyle):
        """Test that extraction is case-sensitive."""
        filename = "rdf_Water_WATER.xvg"
        mol_list = ["water", "Water", "WATER"]

        result = RdfParser.extract_molecules(filename, mol_list)

        # Should find exact matches only
        assert "Water" in result
        assert "WATER" in result
        assert "water" not in result

    def test_extract_with_path_object(self, mock_mplstyle):
        """Test extraction with Path object."""
        filepath = Path("path/to/rdf_water_ethanol.xvg")
        mol_list = ["water", "ethanol"]

        result = RdfParser.extract_molecules(str(filepath), mol_list)

        assert "water" in result
        assert "ethanol" in result

    def test_extract_non_string_input(self, mock_mplstyle):
        """Test extraction with non-string input."""
        # Should convert to string
        filepath = Path("rdf_water.xvg")
        mol_list = ["water"]

        result = RdfParser.extract_molecules(filepath, mol_list)

        assert "water" in result

    def test_extract_unconvertible_input(self, mock_mplstyle):
        """Test extraction with unconvertible input."""
        # Object that can't be converted to string properly
        class BadObject:
            def __str__(self):
                raise TypeError("Cannot convert")

        with pytest.raises(TypeError, match="Could not convert"):
            RdfParser.extract_molecules(BadObject(), ["water"])

    def test_extract_empty_mol_list(self, mock_mplstyle):
        """Test extraction with empty molecule list."""
        filename = "rdf_water_ethanol.xvg"
        mol_list = []

        # With empty mol_list, the pattern will be empty and match nothing
        # or the regex will fail. Either way, result should be empty or raise error
        with pytest.raises(ValueError, match="Unable to match molecules"):
            RdfParser.extract_molecules(filename, mol_list)

    def test_extract_overlapping_names(self, mock_mplstyle):
        """Test extraction with overlapping molecule names."""
        filename = "rdf_methanol_methane.xvg"
        mol_list = ["methanol", "methane", "meth"]

        result = RdfParser.extract_molecules(filename, mol_list)

        # Should find exact matches
        assert "methanol" in result
        assert "methane" in result


class TestIntegration:
    """Integration tests for RdfParser."""

    def test_complete_workflow_converged(self, sample_rdf_file, mock_mplstyle):
        """Test complete workflow with converged RDF."""
        parser = RdfParser(sample_rdf_file)

        # Check all properties are accessible
        assert parser.fname is not None
        assert len(parser.r) > 0
        assert len(parser.g) > 0
        assert len(parser.r_tail) > 0
        assert len(parser.g_tail) > 0

        # Check convergence
        assert isinstance(parser.is_converged, bool)

        # Check report
        report = parser.convergence_report
        assert isinstance(report, str)
        assert len(report) > 0

    def test_complete_workflow_divergent(self, divergent_rdf_file, mock_mplstyle):
        """Test complete workflow with divergent RDF."""
        parser = RdfParser(divergent_rdf_file)

        # Should still create parser
        assert parser.fname is not None

        # May or may not be converged depending on thresholds
        assert isinstance(parser.is_converged, bool)

    @patch('matplotlib.pyplot.subplots')
    def test_workflow_with_plotting(self, mock_subplots, sample_rdf_file, tmp_path, mock_mplstyle):
        """Test workflow including plotting."""
        mock_fig = Mock()
        mock_ax = Mock()
        mock_inset = Mock()
        mock_ax.inset_axes.return_value = mock_inset
        mock_subplots.return_value = (mock_fig, mock_ax)

        parser = RdfParser(sample_rdf_file)

        # Generate report
        report = parser.convergence_report
        assert len(report) > 0

        # Create plot
        parser.plot(save_dir=str(tmp_path))

        assert mock_fig.savefig.called

    def test_multiple_parsers(self, tmp_path, mock_mplstyle):
        """Test creating multiple parsers."""
        # Create two different RDF files
        r1 = np.linspace(0, 5, 100)
        g1 = 1.0 + np.exp(-r1) * np.sin(10 * r1) * 0.5
        g1[-20:] = 1.0 + np.random.normal(0, 0.001, 20)

        rdf_file1 = tmp_path / "rdf_test1.xvg"
        content1 = "# RDF data\n"
        for r_val, g_val in zip(r1, g1, strict=False):
            content1 += f"{r_val:.6f}    {g_val:.6f}\n"
        rdf_file1.write_text(content1)

        r2 = np.linspace(0, 4, 80)
        g2 = 1.0 + np.exp(-r2) * np.cos(8 * r2) * 0.3
        g2[-15:] = 1.0 + np.random.normal(0, 0.001, 15)

        rdf_file2 = tmp_path / "rdf_test2.xvg"
        content2 = "# RDF data\n"
        for r_val, g_val in zip(r2, g2, strict=False):
            content2 += f"{r_val:.6f}    {g_val:.6f}\n"
        rdf_file2.write_text(content2)

        parser1 = RdfParser(str(rdf_file1))
        parser2 = RdfParser(str(rdf_file2))

        # Should be independent
        assert parser1.fname != parser2.fname
        assert len(parser1.r) != len(parser2.r)

    def test_extract_molecules_integration(self, sample_rdf_file, mock_mplstyle):
        """Test extract_molecules with actual parser."""
        parser = RdfParser(sample_rdf_file)

        mol_list = ["test", "water", "ethanol"]
        result = RdfParser.extract_molecules(parser.fname, mol_list)

        assert isinstance(result, list)
        assert "test" in result
