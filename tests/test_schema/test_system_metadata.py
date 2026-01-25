"""Unit tests for SystemMetadata dataclass."""
import warnings

# Suppress NumPy/SciPy compatibility warning (harmless with NumPy 2.x + SciPy 1.16+)
warnings.filterwarnings('ignore', message='numpy.ndarray size changed', category=RuntimeWarning)

from pathlib import Path
from unittest.mock import Mock

import pytest

from kbkit.schema.system_metadata import SystemMetadata


@pytest.fixture
def mock_system_properties():
    """Create a mock SystemProperties object."""
    mock_props = Mock()
    mock_props.name = "test_system"
    return mock_props


@pytest.fixture
def temp_rdf_directory(tmp_path):
    """Create a temporary directory with RDF files."""
    rdf_dir = tmp_path / "rdf"
    rdf_dir.mkdir()
    # Create some .xvg files
    (rdf_dir / "rdf_1.xvg").write_text("# RDF data 1")
    (rdf_dir / "rdf_2.xvg").write_text("# RDF data 2")
    return rdf_dir


@pytest.fixture
def empty_rdf_directory(tmp_path):
    """Create an empty temporary directory."""
    rdf_dir = tmp_path / "empty_rdf"
    rdf_dir.mkdir()
    return rdf_dir


class TestSystemMetadataInitialization:
    """Test SystemMetadata initialization and basic attributes."""

    def test_basic_initialization(self, mock_system_properties, tmp_path):
        """Test creating a SystemMetadata instance with required fields."""
        system = SystemMetadata(
            name="water",
            kind="pure",
            path=tmp_path,
            props=mock_system_properties
        )

        assert system.name == "water"
        assert system.kind == "pure"
        assert system.path == tmp_path
        assert system.props == mock_system_properties
        assert isinstance(system.rdf_path, Path)

    def test_initialization_with_rdf_path(self, mock_system_properties, tmp_path, temp_rdf_directory):
        """Test creating a SystemMetadata instance with RDF path."""
        system = SystemMetadata(
            name="ethanol",
            kind="pure",
            path=tmp_path,
            props=mock_system_properties,
            rdf_path=temp_rdf_directory
        )

        assert system.rdf_path == temp_rdf_directory

    def test_default_rdf_path(self, mock_system_properties, tmp_path):
        """Test that rdf_path defaults to empty Path."""
        system = SystemMetadata(
            name="mixture",
            kind="mixture",
            path=tmp_path,
            props=mock_system_properties
        )

        assert system.rdf_path == Path()

    def test_path_conversion(self, mock_system_properties):
        """Test that string paths are converted to Path objects."""
        system = SystemMetadata(
            name="test",
            kind="pure",
            path=Path("/tmp/test"),
            props=mock_system_properties,
            rdf_path=Path("/tmp/rdf")
        )

        assert isinstance(system.path, Path)
        assert isinstance(system.rdf_path, Path)


class TestHasRdfMethod:
    """Test the has_rdf() method."""

    def test_has_rdf_with_xvg_files(self, mock_system_properties, tmp_path, temp_rdf_directory):
        """Test has_rdf returns True when .xvg files exist."""
        system = SystemMetadata(
            name="test",
            kind="pure",
            path=tmp_path,
            props=mock_system_properties,
            rdf_path=temp_rdf_directory
        )

        assert system.has_rdf() is True

    def test_has_rdf_empty_directory(self, mock_system_properties, tmp_path, empty_rdf_directory):
        """Test has_rdf returns False when directory is empty."""
        system = SystemMetadata(
            name="test",
            kind="pure",
            path=tmp_path,
            props=mock_system_properties,
            rdf_path=empty_rdf_directory
        )

        assert system.has_rdf() is False

    def test_has_rdf_default_path(self, mock_system_properties, tmp_path):
        """Test has_rdf returns False with default empty Path."""
        system = SystemMetadata(
            name="test",
            kind="pure",
            path=tmp_path,
            props=mock_system_properties
        )

        assert system.has_rdf() is False

    def test_has_rdf_nonexistent_directory(self, mock_system_properties, tmp_path):
        """Test has_rdf returns False when directory doesn't exist."""
        nonexistent_path = tmp_path / "nonexistent"
        system = SystemMetadata(
            name="test",
            kind="pure",
            path=tmp_path,
            props=mock_system_properties,
            rdf_path=nonexistent_path
        )

        assert system.has_rdf() is False

    def test_has_rdf_with_non_xvg_files(self, mock_system_properties, tmp_path):
        """Test has_rdf returns False when only non-.xvg files exist."""
        rdf_dir = tmp_path / "rdf"
        rdf_dir.mkdir()
        (rdf_dir / "data.txt").write_text("not an xvg file")
        (rdf_dir / "readme.md").write_text("documentation")

        system = SystemMetadata(
            name="test",
            kind="pure",
            path=tmp_path,
            props=mock_system_properties,
            rdf_path=rdf_dir
        )

        assert system.has_rdf() is False

    def test_has_rdf_mixed_files(self, mock_system_properties, tmp_path):
        """Test has_rdf returns True when .xvg files exist among other files."""
        rdf_dir = tmp_path / "rdf"
        rdf_dir.mkdir()
        (rdf_dir / "rdf.xvg").write_text("RDF data")
        (rdf_dir / "data.txt").write_text("other data")

        system = SystemMetadata(
            name="test",
            kind="pure",
            path=tmp_path,
            props=mock_system_properties,
            rdf_path=rdf_dir
        )

        assert system.has_rdf() is True


class TestIsPureMethod:
    """Test the is_pure() method."""

    def test_is_pure_lowercase(self, mock_system_properties, tmp_path):
        """Test is_pure returns True for 'pure' kind."""
        system = SystemMetadata(
            name="water",
            kind="pure",
            path=tmp_path,
            props=mock_system_properties
        )

        assert system.is_pure() is True

    def test_is_pure_uppercase(self, mock_system_properties, tmp_path):
        """Test is_pure returns True for 'PURE' kind (case-insensitive)."""
        system = SystemMetadata(
            name="water",
            kind="PURE",
            path=tmp_path,
            props=mock_system_properties
        )

        assert system.is_pure() is True

    def test_is_pure_mixed_case(self, mock_system_properties, tmp_path):
        """Test is_pure returns True for 'PuRe' kind (case-insensitive)."""
        system = SystemMetadata(
            name="water",
            kind="PuRe",
            path=tmp_path,
            props=mock_system_properties
        )

        assert system.is_pure() is True

    def test_is_not_pure_mixture(self, mock_system_properties, tmp_path):
        """Test is_pure returns False for 'mixture' kind."""
        system = SystemMetadata(
            name="water_ethanol",
            kind="mixture",
            path=tmp_path,
            props=mock_system_properties
        )

        assert system.is_pure() is False

    def test_is_not_pure_other(self, mock_system_properties, tmp_path):
        """Test is_pure returns False for other kind values."""
        system = SystemMetadata(
            name="test",
            kind="binary",
            path=tmp_path,
            props=mock_system_properties
        )

        assert system.is_pure() is False

    def test_is_not_pure_empty_string(self, mock_system_properties, tmp_path):
        """Test is_pure returns False for empty string kind."""
        system = SystemMetadata(
            name="test",
            kind="",
            path=tmp_path,
            props=mock_system_properties
        )

        assert system.is_pure() is False


class TestSystemMetadataEdgeCases:
    """Test edge cases and special scenarios."""

    def test_system_with_special_characters_in_name(self, mock_system_properties, tmp_path):
        """Test system with special characters in name."""
        system = SystemMetadata(
            name="H2O-EtOH_mix@298K",
            kind="mixture",
            path=tmp_path,
            props=mock_system_properties
        )

        assert system.name == "H2O-EtOH_mix@298K"

    def test_system_with_unicode_name(self, mock_system_properties, tmp_path):
        """Test system with Unicode characters in name."""
        system = SystemMetadata(
            name="système_α",
            kind="pure",
            path=tmp_path,
            props=mock_system_properties
        )

        assert system.name == "système_α"

    def test_dataclass_equality(self, mock_system_properties, tmp_path):
        """Test that two SystemMetadata instances with same values are equal."""
        system1 = SystemMetadata(
            name="water",
            kind="pure",
            path=tmp_path,
            props=mock_system_properties
        )
        system2 = SystemMetadata(
            name="water",
            kind="pure",
            path=tmp_path,
            props=mock_system_properties
        )

        assert system1 == system2

    def test_dataclass_inequality(self, mock_system_properties, tmp_path):
        """Test that SystemMetadata instances with different values are not equal."""
        system1 = SystemMetadata(
            name="water",
            kind="pure",
            path=tmp_path,
            props=mock_system_properties
        )
        system2 = SystemMetadata(
            name="ethanol",
            kind="pure",
            path=tmp_path,
            props=mock_system_properties
        )

        assert system1 != system2

    def test_repr_contains_key_info(self, mock_system_properties, tmp_path):
        """Test that repr contains key information."""
        system = SystemMetadata(
            name="water",
            kind="pure",
            path=tmp_path,
            props=mock_system_properties
        )

        repr_str = repr(system)
        assert "water" in repr_str
        assert "pure" in repr_str


class TestSystemMetadataIntegration:
    """Integration tests for SystemMetadata."""

    def test_complete_workflow_pure_system(self, mock_system_properties, tmp_path, temp_rdf_directory):
        """Test complete workflow for a pure system with RDF data."""
        system = SystemMetadata(
            name="argon",
            kind="pure",
            path=tmp_path,
            props=mock_system_properties,
            rdf_path=temp_rdf_directory
        )

        assert system.is_pure() is True
        assert system.has_rdf() is True
        assert system.name == "argon"
        assert system.path.exists()

    def test_complete_workflow_mixture_system(self, mock_system_properties, tmp_path):
        """Test complete workflow for a mixture system without RDF data."""
        system = SystemMetadata(
            name="water_methanol",
            kind="mixture",
            path=tmp_path,
            props=mock_system_properties
        )

        assert system.is_pure() is False
        assert system.has_rdf() is False
        assert system.name == "water_methanol"

    def test_path_operations(self, mock_system_properties, tmp_path):
        """Test that Path operations work correctly."""
        system_dir = tmp_path / "simulation"
        system_dir.mkdir()

        system = SystemMetadata(
            name="test",
            kind="pure",
            path=system_dir,
            props=mock_system_properties
        )

        assert system.path.exists()
        assert system.path.is_dir()
        assert system.path.name == "simulation"
