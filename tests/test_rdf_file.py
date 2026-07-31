"""Unit tests for RdfParser class."""

import warnings

# Suppress NumPy/SciPy compatibility warning (harmless with NumPy 2.x + SciPy 1.16+)
warnings.filterwarnings("ignore", message="numpy.ndarray size changed", category=RuntimeWarning)

import os
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
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
    content += '@ title "Radial Distribution Function"\n'
    content += '@ xaxis label "r (nm)"\n'
    content += '@ yaxis label "g(r)"\n'

    for r_val, g_val in zip(r, g, strict=False):
        content += f"{r_val:.6f}    {g_val:.6f}\n"

    rdf_file.write_text(content)
    return str(rdf_file)


@pytest.fixture
def sample_txt_file(tmp_path):
    """Create a sample RDF .txt file."""
    rdf_file = tmp_path / "rdf_test.txt"
    r = np.linspace(0.1, 3.0, 50)
    g = 1.0 + np.exp(-r) * np.cos(5 * r) * 0.3

    content = "# RDF data in txt format\n"
    for r_val, g_val in zip(r, g, strict=False):
        content += f"{r_val:.6f}    {g_val:.6f}\n"

    rdf_file.write_text(content)
    return str(rdf_file)


@pytest.fixture
def sample_csv_file(tmp_path):
    """Create a sample RDF .csv file."""
    rdf_file = tmp_path / "rdf_test.csv"
    r = np.linspace(0.1, 2.0, 30)
    g = 1.0 + 0.5 * np.exp(-2 * r)

    # Create DataFrame and save as CSV
    df = pd.DataFrame({"r": r, "g": g})
    df.to_csv(rdf_file, index=False)
    return str(rdf_file)


@pytest.fixture
def sample_xlsx_file(tmp_path):
    """Create a sample RDF .xlsx file."""
    rdf_file = tmp_path / "rdf_test.xlsx"
    r = np.linspace(0.1, 2.5, 40)
    g = 1.0 + 0.3 * np.sin(3 * r)

    # Create DataFrame and save as Excel
    df = pd.DataFrame({"r": r, "g": g})
    df.to_excel(rdf_file, index=False)
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
    with patch("kbkit.io.rdf.load_mplstyle"):
        yield


class TestRdfParserInitialization:
    """Test RdfParser initialization."""

    def test_valid_rdf_file(self, sample_rdf_file, mock_mplstyle):
        """Test initialization with valid RDF file."""
        parser = RdfParser(sample_rdf_file)

        assert isinstance(parser.r, np.ndarray)
        assert isinstance(parser.gr, np.ndarray)
        assert hasattr(parser, "filepath")
        assert parser.filepath.name == "rdf_test.xvg"

    def test_initialization_with_path_object(self, sample_rdf_file, mock_mplstyle):
        """Test initialization with Path object."""
        path_obj = Path(sample_rdf_file)
        parser = RdfParser(path_obj)

        assert isinstance(parser.r, np.ndarray)
        assert isinstance(parser.gr, np.ndarray)
        assert parser.filepath == path_obj

    def test_file_validation_nonexistent(self, tmp_path, mock_mplstyle):
        """Test file path validation with non-existent file."""
        with pytest.raises(FileNotFoundError, match="Path is not a file"):
            RdfParser(str(tmp_path / "nonexistent.xvg"))

    def test_file_validation_unreadable(self, tmp_path, mock_mplstyle):
        """Test file validation with unreadable file."""
        # Create a file but make it unreadable (if possible on the system)
        unreadable_file = tmp_path / "unreadable.xvg"
        unreadable_file.write_text("test")

        # Mock open to raise IOError
        with patch("builtins.open", side_effect=IOError("Permission denied")):
            with pytest.raises(IOError, match="Error reading file"):
                RdfParser(str(unreadable_file))


class TestRdfParserReadMethod:
    """Test _read method with different file formats."""

    def test_read_xvg_file(self, sample_rdf_file, mock_mplstyle):
        """Test reading .xvg file format."""
        parser = RdfParser(sample_rdf_file)

        assert len(parser.r) > 0
        assert len(parser.gr) > 0
        assert len(parser.r) == len(parser.gr)
        # Should filter out last point
        assert len(parser.r) == 99  # 100 - 1

    def test_read_txt_file(self, sample_txt_file, mock_mplstyle):
        """Test reading .txt file format."""
        parser = RdfParser(sample_txt_file)

        assert len(parser.r) > 0
        assert len(parser.gr) > 0
        assert len(parser.r) == len(parser.gr)
        # Should filter out last point
        assert len(parser.r) == 49  # 50 - 1

    def test_read_csv_file(self, sample_csv_file, mock_mplstyle):
        """Test reading .csv file format."""
        parser = RdfParser(sample_csv_file)

        assert len(parser.r) > 0
        assert len(parser.gr) > 0
        assert len(parser.r) == len(parser.gr)
        # Should filter out last point
        assert len(parser.r) == 29  # 30 - 1

    def test_read_xlsx_file(self, sample_xlsx_file, mock_mplstyle):
        """Test reading .xlsx file format."""
        # Excel files are binary and will fail validation when trying to read as text
        # The _validate_file method tries to read the file as text, which fails for binary files
        with pytest.raises(ValueError, match="Failed to parse RDF data"):
            RdfParser(sample_xlsx_file)

    def test_read_xlsx_file_with_mock_validation(self, tmp_path, mock_mplstyle):
        """Test reading .xlsx file format with mocked validation."""
        # Create a proper xlsx file
        xlsx_file = tmp_path / "test.xlsx"
        r = np.linspace(0.1, 2.5, 40)
        g = 1.0 + 0.3 * np.sin(3 * r)
        df = pd.DataFrame({"r": r, "g": g})
        df.to_excel(xlsx_file, index=False)

        # Mock the validation to skip the text reading check
        with patch.object(RdfParser, "_validate_file", return_value=xlsx_file):
            parser = RdfParser(str(xlsx_file))

            assert len(parser.r) > 0
            assert len(parser.gr) > 0
            assert len(parser.r) == len(parser.gr)
            # Should filter out last point
            assert len(parser.r) == 39  # 40 - 1

    def test_read_unsupported_format(self, tmp_path, mock_mplstyle):
        """Test reading unsupported file format."""
        unsupported_file = tmp_path / "test.dat"
        unsupported_file.write_text("0.1 1.0\n0.2 1.1\n")

        with pytest.raises(ValueError, match="Filetype not supported"):
            RdfParser(str(unsupported_file))

    def test_read_filters_tail_noise(self, sample_rdf_file, mock_mplstyle):
        """Test that _read filters last point."""
        # Count lines in file
        with open(sample_rdf_file) as f:
            data_lines = [line for line in f if not line.startswith(("#", "@"))]

        parser = RdfParser(sample_rdf_file)

        # Should have 1 fewer point than data lines (removes last point)
        assert len(parser.r) == len(data_lines) - 1

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

    def test_read_with_comments(self, tmp_path, mock_mplstyle):
        """Test reading file with various comment characters."""
        comment_file = tmp_path / "comments.xvg"
        content = """# This is a hash comment
@ This is an at comment
; This is a semicolon comment
0.1 1.0
0.2 1.1
0.3 1.2
"""
        comment_file.write_text(content)

        parser = RdfParser(str(comment_file))
        assert len(parser.r) == 2  # 3 data lines - 1 (tail filtering)
        np.testing.assert_array_almost_equal(parser.r, [0.1, 0.2])
        np.testing.assert_array_almost_equal(parser.gr, [1.0, 1.1])


class TestExtractMolecules:
    """Test extract_molecules static method."""

    def test_extract_single_molecule_repeated(self, mock_mplstyle):
        """Test extracting single molecule that appears twice."""
        filename = "rdf_water_water.xvg"
        mol_list = ["water", "ethanol", "methanol"]

        result = RdfParser.extract_molecules(filename, mol_list)

        assert "water" in result
        assert len(result) == 2  # "water" appears twice
        assert result == ["water", "water"]

    def test_extract_multiple_molecules(self, mock_mplstyle):
        """Test extracting multiple different molecules."""
        filename = "rdf_water_ethanol.xvg"
        mol_list = ["water", "ethanol", "methanol"]

        result = RdfParser.extract_molecules(filename, mol_list)

        assert "water" in result
        assert "ethanol" in result
        assert len(result) == 2

    def test_extract_insufficient_molecules(self, mock_mplstyle):
        """Test when insufficient molecules are found."""
        filename = "rdf_water.xvg"  # Only one molecule
        mol_list = ["water", "ethanol", "methanol"]

        with pytest.raises(ValueError, match="Unable to find both molecules"):
            RdfParser.extract_molecules(filename, mol_list)

    def test_extract_no_molecules(self, mock_mplstyle):
        """Test when no molecules are found."""
        filename = "rdf_unknown.xvg"
        mol_list = ["water", "ethanol", "methanol"]

        with pytest.raises(ValueError, match="Unable to find both molecules"):
            RdfParser.extract_molecules(filename, mol_list)

    def test_extract_with_special_characters(self, mock_mplstyle):
        """Test extraction with special characters in molecule names."""
        filename = "rdf_H2O_CO2.xvg"
        mol_list = ["H2O", "CO2", "CH4"]

        result = RdfParser.extract_molecules(filename, mol_list)

        assert "H2O" in result
        assert "CO2" in result
        assert len(result) == 2

    def test_extract_case_sensitive(self, mock_mplstyle):
        """Test that extraction is case-sensitive."""
        filename = "rdf_Water_WATER.xvg"
        mol_list = ["water", "Water", "WATER"]

        result = RdfParser.extract_molecules(filename, mol_list)

        # Should find exact matches only
        assert "Water" in result
        assert "WATER" in result
        assert "water" not in result
        assert len(result) == 2

    def test_extract_with_path_object(self, mock_mplstyle):
        """Test extraction with Path object."""
        filepath = Path("path/to/rdf_water_ethanol.xvg")
        mol_list = ["water", "ethanol"]

        result = RdfParser.extract_molecules(str(filepath), mol_list)

        assert "water" in result
        assert "ethanol" in result
        assert len(result) == 2

    def test_extract_non_string_input(self, mock_mplstyle):
        """Test extraction with non-string input that can be converted."""
        filepath = Path("rdf_water_water.xvg")
        mol_list = ["water"]

        result = RdfParser.extract_molecules(filepath, mol_list)

        assert "water" in result
        assert len(result) == 2

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

        with pytest.raises(ValueError, match="Unable to match molecules"):
            RdfParser.extract_molecules(filename, mol_list)

    def test_extract_overlapping_names(self, mock_mplstyle):
        """Test extraction with overlapping molecule names."""
        filename = "rdf_methanol_methane.xvg"
        mol_list = ["methanol", "methane", "meth"]

        result = RdfParser.extract_molecules(filename, mol_list)

        # Should find exact matches - methanol and methane, not meth
        assert "methanol" in result
        assert "methane" in result
        assert len(result) == 2

    def test_extract_with_underscores_and_numbers(self, mock_mplstyle):
        """Test extraction with complex molecule names."""
        filename = "rdf_mol_1_mol_2.xvg"
        mol_list = ["mol_1", "mol_2", "mol_3"]

        result = RdfParser.extract_molecules(filename, mol_list)

        assert "mol_1" in result
        assert "mol_2" in result
        assert len(result) == 2

    def test_extract_regex_escaping(self, mock_mplstyle):
        """Test that special regex characters are properly escaped."""
        filename = "rdf_mol+1_mol*2.xvg"
        mol_list = ["mol+1", "mol*2", "mol.3"]

        result = RdfParser.extract_molecules(filename, mol_list)

        assert "mol+1" in result
        assert "mol*2" in result
        assert len(result) == 2


class TestPlotRDF:
    """Test plotRDF method."""

    def test_plot_rdf_basic(self, sample_rdf_file, mock_mplstyle):
        """Test basic plotting functionality."""
        parser = RdfParser(sample_rdf_file)
        mock_ax = Mock()

        parser.plotRDF(mock_ax)

        # Verify plot was called with correct data
        mock_ax.plot.assert_called_once_with(parser.r, parser.gr)

    def test_plot_rdf_with_kwargs(self, sample_rdf_file, mock_mplstyle):
        """Test plotting with keyword arguments."""
        parser = RdfParser(sample_rdf_file)
        mock_ax = Mock()

        parser.plotRDF(mock_ax, label="test", color="red", linewidth=2)

        # Verify plot was called with kwargs
        mock_ax.plot.assert_called_once_with(parser.r, parser.gr, label="test", color="red", linewidth=2)

    def test_plot_rdf_multiple_calls(self, sample_rdf_file, mock_mplstyle):
        """Test multiple plotting calls."""
        parser = RdfParser(sample_rdf_file)
        mock_ax = Mock()

        parser.plotRDF(mock_ax, label="first")
        parser.plotRDF(mock_ax, label="second")

        # Verify plot was called twice
        assert mock_ax.plot.call_count == 2


class TestRdfParserIntegration:
    """Integration tests for RdfParser."""

    def test_complete_workflow_converged(self, sample_rdf_file, mock_mplstyle):
        """Test complete workflow with converged RDF."""
        parser = RdfParser(sample_rdf_file)

        # Check all properties are accessible
        assert len(parser.r) > 0
        assert len(parser.gr) > 0
        assert isinstance(parser.filepath, Path)

        # Test that data makes sense
        assert parser.r[0] < parser.r[-1]  # r should be increasing
        assert np.all(parser.r >= 0)  # r should be non-negative

    def test_complete_workflow_divergent(self, divergent_rdf_file, mock_mplstyle):
        """Test complete workflow with divergent RDF."""
        parser = RdfParser(divergent_rdf_file)

        # Should still create parser successfully
        assert len(parser.r) > 0
        assert len(parser.gr) > 0

    @patch("matplotlib.pyplot.subplots")
    def test_workflow_with_plotting(self, mock_subplots, sample_rdf_file, mock_mplstyle):
        """Test workflow including plotting."""
        mock_fig = Mock()
        mock_ax = Mock()
        mock_subplots.return_value = (mock_fig, mock_ax)

        parser = RdfParser(sample_rdf_file)

        # Create plot using plotRDF method
        parser.plotRDF(mock_ax, label="test")

        # Verify plot was called
        assert mock_ax.plot.called

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
        assert len(parser1.r) != len(parser2.r)
        assert not np.array_equal(parser1.r, parser2.r)

    def test_extract_molecules_integration(self, sample_rdf_file, mock_mplstyle):
        """Test extract_molecules with actual parser."""
        parser = RdfParser(sample_rdf_file)

        # Extract filename from path - need a filename with two molecules
        # The extract_molecules method expects exactly 2 molecules in the filename
        test_filename = "rdf_water_ethanol.xvg"
        mol_list = ["water", "ethanol", "methanol"]
        result = RdfParser.extract_molecules(test_filename, mol_list)

        assert isinstance(result, list)
        assert len(result) == 2
        assert "water" in result
        assert "ethanol" in result

    def test_real_data_format(self, tmp_path, mock_mplstyle):
        """Test with realistic RDF data format."""
        # Create file with realistic water RDF data
        rdf_file = tmp_path / "water_water.xvg"

        # Simulate water-water RDF with first peak around 2.8 Å
        r = np.linspace(0.1, 10.0, 200)
        g = np.ones_like(r)

        # Add first hydration shell peak
        for i, r_val in enumerate(r):
            if 2.5 <= r_val <= 3.2:
                g[i] = 1.0 + 2.0 * np.exp(-(((r_val - 2.8) / 0.2) ** 2))
            elif 4.0 <= r_val <= 5.0:
                g[i] = 1.0 + 0.5 * np.exp(-(((r_val - 4.5) / 0.3) ** 2))
            else:
                g[i] = 1.0 + 0.05 * np.random.normal()

        content = "# Water-Water RDF\n"
        content += "@ title 'Water-Water Radial Distribution Function'\n"
        for r_val, g_val in zip(r, g, strict=False):
            content += f"{r_val:.6f}    {g_val:.6f}\n"

        rdf_file.write_text(content)

        parser = RdfParser(str(rdf_file))

        # Verify realistic properties
        assert len(parser.r) == 199  # 200 - 1 (tail filtering)
        assert parser.r[0] == 0.1
        assert parser.r[-1] < 10.0  # Should be less due to tail filtering
        assert np.max(parser.gr) > 2.0  # Should have significant peak


class TestRdfParserEdgeCases:
    """Test edge cases and error conditions."""

    def test_very_small_file(self, tmp_path, mock_mplstyle):
        """Test with very small data file."""
        small_file = tmp_path / "small.xvg"
        content = "0.1 1.0\n0.2 1.1\n"
        small_file.write_text(content)

        parser = RdfParser(str(small_file))

        # Should have 1 point after tail filtering
        assert len(parser.r) == 1
        assert len(parser.gr) == 1

    def test_file_with_only_comments(self, tmp_path, mock_mplstyle):
        """Test file with only comment lines."""
        comment_file = tmp_path / "comments_only.xvg"
        content = "# Only comments\n@ More comments\n; Even more comments\n"
        comment_file.write_text(content)

        with pytest.raises((ValueError, IndexError)):
            RdfParser(str(comment_file))

    def test_file_with_mixed_delimiters(self, tmp_path, mock_mplstyle):
        """Test file with inconsistent delimiters."""
        mixed_file = tmp_path / "mixed.txt"
        content = "0.1\t1.0\n0.2 1.1\n0.3   1.2\n"
        mixed_file.write_text(content)

        # Should still work due to numpy's flexible parsing
        parser = RdfParser(str(mixed_file))
        assert len(parser.r) == 2  # 3 - 1 (tail filtering)

    def test_file_with_scientific_notation(self, tmp_path, mock_mplstyle):
        """Test file with scientific notation."""
        sci_file = tmp_path / "scientific.xvg"
        content = "1.0e-1 1.0e0\n2.0e-1 1.1e0\n3.0e-1 1.2e0\n"
        sci_file.write_text(content)

        parser = RdfParser(str(sci_file))
        assert len(parser.r) == 2  # 3 - 1 (tail filtering)
        np.testing.assert_array_almost_equal(parser.r, [0.1, 0.2])

    def test_csv_with_header(self, tmp_path, mock_mplstyle):
        """Test CSV file with header row."""
        csv_file = tmp_path / "with_header.csv"
        r = np.array([0.1, 0.2, 0.3])
        g = np.array([1.0, 1.1, 1.2])

        df = pd.DataFrame({"distance": r, "rdf": g})
        df.to_csv(csv_file, index=False)

        parser = RdfParser(str(csv_file))
        # Should read the data correctly despite header
        assert len(parser.r) == 2  # 3 - 1 (tail filtering)


class TestRdfParserValidation:
    """Test validation methods."""

    def test_validate_file_with_pathlib(self, sample_rdf_file, mock_mplstyle):
        """Test file validation with pathlib.Path input."""
        path_obj = Path(sample_rdf_file)
        validated_path = RdfParser._validate_file(path_obj)

        assert isinstance(validated_path, Path)
        assert validated_path == path_obj

    def test_validate_file_with_string(self, sample_rdf_file, mock_mplstyle):
        """Test file validation with string input."""
        validated_path = RdfParser._validate_file(sample_rdf_file)

        assert isinstance(validated_path, Path)
        assert str(validated_path) == sample_rdf_file

    def test_validate_file_preserves_suffix(self, sample_rdf_file, mock_mplstyle):
        """Test that file validation preserves the original suffix."""
        validated_path = RdfParser._validate_file(sample_rdf_file)

        assert validated_path.suffix == ".xvg"

    @patch("kbkit.io.rdf.validate_path")
    def test_validate_file_calls_validation(self, mock_validate, sample_rdf_file, mock_mplstyle):
        """Test that _validate_file calls the validation utility."""
        mock_validate.return_value = Path(sample_rdf_file)

        RdfParser._validate_file(sample_rdf_file)

        mock_validate.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__])
