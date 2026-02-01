"""
Unit tests for the SystemCollection module.

This test suite provides comprehensive coverage of the SystemCollection class,
including system discovery, metadata creation, property access, and plotting.
"""
import warnings

# Suppress NumPy/SciPy compatibility warning (harmless with NumPy 2.x + SciPy 1.16+)
warnings.filterwarnings('ignore', message='numpy.ndarray size changed', category=RuntimeWarning)

from pathlib import Path
from unittest.mock import Mock, PropertyMock, patch

import numpy as np
import pytest

from kbkit.schema.property_result import PropertyResult
from kbkit.schema.system_metadata import SystemMetadata
from kbkit.systems.collection import SystemCollection
from kbkit.systems.properties import SystemProperties
from kbkit.visualization.timeseries import TimeseriesPlotter


@pytest.fixture
def mock_system_metadata():
    """Create a mock SystemMetadata object."""
    mock_meta = Mock(spec=SystemMetadata)
    mock_meta.name = "test_system"
    mock_meta.kind = "mixture"
    mock_meta.path = Path("/path/to/system")
    mock_meta.rdf_path = Path("/path/to/system/rdf")

    # Mock properties
    mock_props = Mock(spec=SystemProperties)
    mock_topology = Mock()
    mock_topology.molecule_count = {"MOL1": 50, "MOL2": 50}
    mock_topology.total_molecules = 100
    mock_props.topology = mock_topology
    mock_props.get.return_value = 1.0

    mock_meta.props = mock_props
    mock_meta.is_pure.return_value = False

    return mock_meta


@pytest.fixture
def mock_pure_metadata():
    """Create a mock pure SystemMetadata object."""
    mock_meta = Mock(spec=SystemMetadata)
    mock_meta.name = "pure_MOL1"
    mock_meta.kind = "pure"
    mock_meta.path = Path("/path/to/pure/MOL1")
    mock_meta.rdf_path = Path()

    mock_props = Mock(spec=SystemProperties)
    mock_topology = Mock()
    mock_topology.molecule_count = {"MOL1": 100}
    mock_topology.total_molecules = 100
    mock_topology.molecules = ["MOL1"]
    mock_props.topology = mock_topology
    mock_props.get.return_value = 1.0

    mock_meta.props = mock_props
    mock_meta.is_pure.return_value = True

    return mock_meta


@pytest.fixture
def sample_systems(mock_system_metadata, mock_pure_metadata):
    """Create a list of sample systems."""
    return [mock_pure_metadata, mock_system_metadata]


@pytest.fixture
def sample_molecules():
    """Sample molecule list."""
    return ["MOL1", "MOL2"]


class TestSystemCollectionInitialization:
    """Test SystemCollection initialization."""

    def test_init_with_systems(self, sample_systems, sample_molecules):
        """Test initialization with systems and molecules."""
        sc = SystemCollection(sample_systems, sample_molecules)

        assert np.all(sc._systems == sample_systems)
        # The constructor stores a numeric array in _molecules internally;
        # the public .molecules property is what exposes the original string list.
        assert sc.molecules == sample_molecules
        assert len(sc._lookup) == 2
        assert "test_system" in sc._lookup
        assert "pure_MOL1" in sc._lookup
        assert isinstance(sc._cache, dict)

    def test_init_creates_lookup_dict(self, sample_systems, sample_molecules):
        """Test that initialization creates proper lookup dictionary."""
        sc = SystemCollection(sample_systems, sample_molecules)

        assert sc._lookup["test_system"] == sample_systems[1]
        assert sc._lookup["pure_MOL1"] == sample_systems[0]

    def test_init_with_empty_systems(self):
        """Test initialization with empty systems list."""
        sc = SystemCollection([], [])

        assert sc._systems == []
        # Same reason as test_init_with_systems: use the public property.
        assert sc.molecules == []
        assert sc._lookup == {}


class TestSystemCollectionPeekMolecules:
    """Test the _peek_molecules static method."""

    def test_peek_molecules_from_top_file(self, tmp_path):
        """Test extracting molecules from .top file."""
        top_content = """
; topology file
[ molecules ]
; Compound        #mols
MOL1              50
MOL2              50
"""
        top_file = tmp_path / "system.top"
        top_file.write_text(top_content)

        mols = SystemCollection._peek_molecules(tmp_path)

        assert "MOL1" in mols
        assert "MOL2" in mols
        assert len(mols) == 2

    def test_peek_molecules_from_top_file_with_blank_lines(self, tmp_path):
        """Test extracting molecules from .top file with blank lines."""
        top_content = """
[ molecules ]
MOL1    50

MOL2    50

"""
        top_file = tmp_path / "system.top"
        top_file.write_text(top_content)

        mols = SystemCollection._peek_molecules(tmp_path)

        assert "MOL1" in mols
        assert "MOL2" in mols

    def test_peek_molecules_from_gro_file(self, tmp_path):
        """Test extracting molecules from .gro file when .top is absent."""
        gro_content = """Test system
100
    1MOL1    C1    1   1.000   2.000   3.000
    1MOL1    C2    2   1.100   2.100   3.100
    2MOL2    C1    3   1.200   2.200   3.200
    2MOL2    C2    4   1.300   2.300   3.300
   5.0   5.0   5.0
"""
        gro_file = tmp_path / "system.gro"
        gro_file.write_text(gro_content)

        mols = SystemCollection._peek_molecules(tmp_path)

        assert "MOL1" in mols or "MOL2" in mols
        assert len(mols) >= 1

    def test_peek_molecules_from_gro_skips_short_lines(self, tmp_path):
        """Test that peek_molecules skips short lines in .gro file."""
        gro_content = """Test
100
short
    1MOL1    C1    1   1.000   2.000   3.000
   5.0   5.0   5.0
"""
        gro_file = tmp_path / "system.gro"
        gro_file.write_text(gro_content)

        mols = SystemCollection._peek_molecules(tmp_path)

        assert "MOL1" in mols

    def test_peek_molecules_from_gro_skips_numeric_residues(self, tmp_path):
        """Test that peek_molecules skips numeric residue names."""
        gro_content = """Test
100
    11234    C1    1   1.000   2.000   3.000
    2MOL1    C1    2   1.000   2.000   3.000
   5.0   5.0   5.0
"""
        gro_file = tmp_path / "system.gro"
        gro_file.write_text(gro_content)

        mols = SystemCollection._peek_molecules(tmp_path)

        assert "MOL1" in mols
        assert "1234" not in mols

    def test_peek_molecules_empty_directory(self, tmp_path):
        """Test peek_molecules with empty directory."""
        mols = SystemCollection._peek_molecules(tmp_path)

        assert mols == set()

    def test_peek_molecules_ignores_comments(self, tmp_path):
        """Test that peek_molecules ignores commented lines in .top."""
        top_content = """
[ molecules ]
; This is a comment
MOL1    50
; MOL3   10
MOL2    50
"""
        top_file = tmp_path / "system.top"
        top_file.write_text(top_content)

        mols = SystemCollection._peek_molecules(tmp_path)

        assert "MOL1" in mols
        assert "MOL2" in mols
        assert "MOL3" not in mols


class TestSystemCollectionFindPureSystems:
    """Test the _find_pure_systems static method."""

    @patch('kbkit.systems.collection.SystemCollection._extract_temp')
    def test_find_pure_systems_exact_match(self, mock_extract_temp, tmp_path):
        """Test finding pure systems with exact temperature match."""
        # Create mock directories
        pure_mol1 = tmp_path / "pure_MOL1_298K"
        pure_mol2 = tmp_path / "pure_MOL2_298K"
        pure_mol1.mkdir()
        pure_mol2.mkdir()

        mock_extract_temp.return_value = 298.0

        result = SystemCollection._find_pure_systems(
            tmp_path, ["MOL1", "MOL2"], 298.0
        )

        assert "MOL1" in result
        assert "MOL2" in result
        assert result["MOL1"] == pure_mol1
        assert result["MOL2"] == pure_mol2

    @patch('kbkit.systems.collection.SystemCollection._extract_temp')
    def test_find_pure_systems_temperature_threshold(self, mock_extract_temp, tmp_path):
        """Test finding pure systems within temperature threshold."""
        pure_mol1 = tmp_path / "pure_MOL1_299K"
        pure_mol1.mkdir()

        mock_extract_temp.return_value = 299.5

        result = SystemCollection._find_pure_systems(
            tmp_path, ["MOL1"], 298.0
        )

        # Within 2K threshold
        assert "MOL1" in result
        assert result["MOL1"] == pure_mol1

    @patch('kbkit.systems.collection.SystemCollection._extract_temp')
    def test_find_pure_systems_outside_threshold(self, mock_extract_temp, tmp_path):
        """Test that systems outside temperature threshold are excluded."""
        pure_mol1 = tmp_path / "pure_MOL1_310K"
        pure_mol1.mkdir()

        mock_extract_temp.return_value = 310.0

        result = SystemCollection._find_pure_systems(
            tmp_path, ["MOL1"], 298.0
        )

        # Outside 2K threshold
        assert result == {}

    @patch('kbkit.systems.collection.SystemCollection._extract_temp')
    def test_find_pure_systems_no_temperature(self, mock_extract_temp, tmp_path):
        """Test finding pure systems when temperature extraction returns None."""
        pure_mol1 = tmp_path / "pure_MOL1"
        pure_mol1.mkdir()

        mock_extract_temp.return_value = None

        result = SystemCollection._find_pure_systems(
            tmp_path, ["MOL1"], 298.0
        )

        assert result == {}

    @patch('kbkit.systems.collection.SystemCollection._extract_temp')
    def test_find_pure_systems_scoring(self, mock_extract_temp, tmp_path):
        """Test that scoring prefers directories with more molecule matches."""
        # Create competing directories
        pure_mol1_a = tmp_path / "MOL1_298K"
        pure_mol1_b = tmp_path / "MOL1_MOL2_298K"  # Contains both molecules
        pure_mol1_a.mkdir()
        pure_mol1_b.mkdir()

        mock_extract_temp.return_value = 298.0

        result = SystemCollection._find_pure_systems(
            tmp_path, ["MOL1", "MOL2"], 298.0
        )

        # Should prefer the one with higher score
        assert result["MOL1"] == pure_mol1_b

    @patch('kbkit.systems.collection.SystemCollection._extract_temp')
    def test_find_pure_systems_case_insensitive(self, mock_extract_temp, tmp_path):
        """Test that molecule matching is case-insensitive."""
        pure_mol1 = tmp_path / "pure_mol1_298K"
        pure_mol1.mkdir()

        mock_extract_temp.return_value = 298.0

        result = SystemCollection._find_pure_systems(
            tmp_path, ["MOL1"], 298.0
        )

        assert "MOL1" in result


class TestSystemCollectionExtractTemp:
    """Test the _extract_temp static method."""

    def test_extract_temp_from_filename(self):
        """Test extracting temperature from filename."""
        path = Path("/path/to/system_298.15K")
        temp = SystemCollection._extract_temp(path)

        assert temp == 298.15

    def test_extract_temp_from_filename_integer(self):
        """Test extracting integer temperature from filename."""
        path = Path("/path/to/system_300K")
        temp = SystemCollection._extract_temp(path)

        assert temp == 300.0

    def test_extract_temp_from_filename_no_decimal(self):
        """Test extracting temperature without decimal."""
        path = Path("/path/to/system_298K")
        temp = SystemCollection._extract_temp(path)

        assert temp == 298.0

    @patch('kbkit.systems.collection.EdrParser')
    def test_extract_temp_from_edr_file(self, mock_edr_parser, tmp_path):
        """Test extracting temperature from .edr file."""
        edr_file = tmp_path / "system.edr"
        edr_file.touch()

        mock_parser_instance = Mock()
        mock_parser_instance.get_gmx_property.return_value = 298.15
        mock_edr_parser.return_value = mock_parser_instance

        temp = SystemCollection._extract_temp(edr_file)

        assert temp == 298.15
        mock_parser_instance.get_gmx_property.assert_called_once_with("temperature", avg=True)

    @patch('kbkit.systems.collection.SystemProperties.find_files')
    @patch('kbkit.systems.collection.EdrParser')
    def test_extract_temp_from_directory(self, mock_edr_parser, mock_find_files, tmp_path):
        """Test extracting temperature from directory containing .edr."""
        edr_file = tmp_path / "system.edr"
        edr_file.touch()

        mock_find_files.return_value = [edr_file]

        mock_parser_instance = Mock()
        mock_parser_instance.get_gmx_property.return_value = 300.0
        mock_edr_parser.return_value = mock_parser_instance

        # Create directory without temp in name
        test_dir = tmp_path / "system"
        test_dir.mkdir()
        (test_dir / "system.edr").touch()

        mock_find_files.return_value = [test_dir / "system.edr"]

        temp = SystemCollection._extract_temp(test_dir)

        assert temp == 300.0

    def test_extract_temp_raises_on_invalid_input(self):
        """Test that _extract_temp raises ValueError for invalid input."""
        with pytest.raises(ValueError, match="Temperature is not in pathname"):
            SystemCollection._extract_temp(Path("/invalid/path/system"))


class TestSystemCollectionIsValid:
    """Test the _is_valid static method."""

    def test_is_valid_with_edr_and_gro(self, tmp_path):
        """Test validation with .edr and .gro files."""
        (tmp_path / "system.edr").touch()
        (tmp_path / "system.gro").touch()

        assert SystemCollection._is_valid(tmp_path) is True

    def test_is_valid_with_edr_and_top(self, tmp_path):
        """Test validation with .edr and .top files."""
        (tmp_path / "system.edr").touch()
        (tmp_path / "system.top").touch()

        assert SystemCollection._is_valid(tmp_path) is True

    def test_is_valid_missing_edr(self, tmp_path):
        """Test validation fails without .edr file."""
        (tmp_path / "system.gro").touch()

        assert SystemCollection._is_valid(tmp_path) is False

    def test_is_valid_missing_structure(self, tmp_path):
        """Test validation fails without .gro or .top file."""
        (tmp_path / "system.edr").touch()

        assert SystemCollection._is_valid(tmp_path) is False

    def test_is_valid_not_directory(self, tmp_path):
        """Test validation fails for non-directory."""
        file_path = tmp_path / "system.edr"
        file_path.touch()

        assert SystemCollection._is_valid(file_path) is False

    def test_is_valid_deep_search(self, tmp_path):
        """Test validation with deep search enabled."""
        subdir = tmp_path / "subdir"
        subdir.mkdir()
        (subdir / "system.edr").touch()
        (subdir / "system.gro").touch()

        assert SystemCollection._is_valid(tmp_path, deep=True) is True

    def test_is_valid_deep_search_false(self, tmp_path):
        """Test validation with deep search disabled."""
        subdir = tmp_path / "subdir"
        subdir.mkdir()
        (subdir / "system.edr").touch()
        (subdir / "system.gro").touch()

        assert SystemCollection._is_valid(tmp_path, deep=False) is False


class TestSystemCollectionFindReferenceDir:
    """Test the _find_reference_dir static method."""

    @patch('kbkit.systems.collection.SystemCollection._is_valid')
    def test_find_reference_dir_in_parent(self, mock_is_valid, tmp_path):
        """Test finding reference directory in parent."""
        pure_dir = tmp_path / "pure_components"
        pure_dir.mkdir()

        mock_is_valid.return_value = True

        start_path = tmp_path / "mixtures" / "system1"
        start_path.mkdir(parents=True)

        result = SystemCollection._find_reference_dir(start_path)

        assert result == pure_dir

    @patch('kbkit.systems.collection.SystemCollection._is_valid')
    def test_find_reference_dir_multiple_keywords(self, mock_is_valid, tmp_path):
        """Test finding reference directory with different keywords."""
        for keyword in ["pure", "single", "ref", "neat"]:
            ref_dir = tmp_path / f"{keyword}_systems"
            ref_dir.mkdir()

            mock_is_valid.return_value = True

            result = SystemCollection._find_reference_dir(tmp_path)

            assert keyword in result.name.lower()
            break

    @patch('kbkit.systems.collection.SystemCollection._is_valid')
    def test_find_reference_dir_not_found(self, mock_is_valid, tmp_path):
        """Test when no reference directory is found."""
        mock_is_valid.return_value = False

        with pytest.raises(FileNotFoundError, match="No parent directories for pure-components were found"):
            SystemCollection._find_reference_dir(tmp_path)



class TestSystemCollectionResolveRDFPath:
    """Test the _resolve_rdf_path static method."""

    def test_resolve_rdf_path_explicit_name(self, tmp_path):
        """Test resolving RDF path with explicit directory name."""
        rdf_dir = tmp_path / "rdfs"
        rdf_dir.mkdir()

        result = SystemCollection._resolve_rdf_path(tmp_path, "rdfs", is_pure=False)

        assert result == rdf_dir

    def test_resolve_rdf_path_search_subdirs(self, tmp_path):
        """Test resolving RDF path by searching subdirectories."""
        rdf_dir = tmp_path / "rdf_data"
        rdf_dir.mkdir()
        (rdf_dir / "rdf.xvg").touch()

        result = SystemCollection._resolve_rdf_path(tmp_path, "", is_pure=False)

        assert result == rdf_dir

    def test_resolve_rdf_path_with_txt_files(self, tmp_path):
        """Test resolving RDF path with .txt files."""
        rdf_dir = tmp_path / "rdf_files"
        rdf_dir.mkdir()
        (rdf_dir / "rdf.txt").touch()

        result = SystemCollection._resolve_rdf_path(tmp_path, "", is_pure=False)

        assert result == rdf_dir

    def test_resolve_rdf_path_case_insensitive(self, tmp_path):
        """Test resolving RDF path is case-insensitive."""
        rdf_dir = tmp_path / "RDF_Data"
        rdf_dir.mkdir()
        (rdf_dir / "rdf.xvg").touch()

        result = SystemCollection._resolve_rdf_path(tmp_path, "", is_pure=False)

        assert result == rdf_dir

    def test_resolve_rdf_path_pure_system_not_found(self, tmp_path):
        """Test that pure systems return empty Path when RDF not found."""
        result = SystemCollection._resolve_rdf_path(tmp_path, "", is_pure=True)

        assert result == Path()

    def test_resolve_rdf_path_mixture_raises_error(self, tmp_path):
        """Test that mixture systems raise error when RDF not found."""
        with pytest.raises(FileNotFoundError, match="No RDF directory found"):
            SystemCollection._resolve_rdf_path(tmp_path, "", is_pure=False)


class TestSystemCollectionMakeMeta:
    """Test the _make_meta static method."""

    @patch('kbkit.systems.collection.SystemProperties')
    @patch('kbkit.systems.collection.SystemMetadata')
    def test_make_meta_creates_metadata(self, mock_metadata_class, mock_props_class, tmp_path):
        """Test that _make_meta creates SystemMetadata correctly."""
        rdf_path = tmp_path / "rdf"

        mock_props = Mock()
        mock_props_class.return_value = mock_props

        mock_meta = Mock()
        mock_metadata_class.return_value = mock_meta

        result = SystemCollection._make_meta(
            tmp_path, "mixture", rdf_path, start_time=5000, include="npt"
        )

        mock_metadata_class.assert_called_once_with(
            name=tmp_path.name,
            kind="mixture",
            path=tmp_path,
            rdf_path=rdf_path,
            props=mock_props
        )

        # SystemProperties is called with str(path), not path
        mock_props_class.assert_called_once_with(
            str(tmp_path), start_time=5000, include="npt"
        )

        assert result == mock_meta


class TestSystemCollectionSortSystems:
    """Test the _sort_systems static method."""

    def test_sort_systems_by_composition(self):
        """Test sorting systems by mole fraction."""
        # Create mock systems with different compositions
        systems = []
        molecules = ["MOL1", "MOL2"]

        for mol1_count in [100, 50, 0]:
            mock_meta = Mock(spec=SystemMetadata)
            mock_props = Mock(spec=SystemProperties)
            mock_topology = Mock()

            mol2_count = 100 - mol1_count
            mock_topology.molecule_count = {"MOL1": mol1_count, "MOL2": mol2_count}
            mock_topology.total_molecules = 100

            mock_props.topology = mock_topology
            mock_meta.props = mock_props

            systems.append(mock_meta)

        # Shuffle to test sorting
        import random
        random.shuffle(systems)

        sorted_systems = SystemCollection._sort_systems(systems, molecules)

        # Check that systems are sorted by MOL1 mole fraction
        mol1_fractions = []
        for s in sorted_systems:
            counts = s.props.topology.molecule_count
            total = s.props.topology.total_molecules
            mol1_fractions.append(counts.get("MOL1", 0) / total)

        assert mol1_fractions == sorted(mol1_fractions)

    def test_sort_systems_handles_zero_molecules(self):
        """Test sorting handles systems with zero molecules."""
        mock_meta = Mock(spec=SystemMetadata)
        mock_props = Mock(spec=SystemProperties)
        mock_topology = Mock()
        mock_topology.molecule_count = {}
        mock_topology.total_molecules = 0
        mock_props.topology = mock_topology
        mock_meta.props = mock_props

        result = SystemCollection._sort_systems([mock_meta], ["MOL1"])

        assert len(result) == 1


class TestSystemCollectionProperties:
    """Test SystemCollection properties and magic methods."""

    def test_getattr_from_metadata(self, sample_systems, sample_molecules):
        """Test __getattr__ retrieves attributes from metadata."""
        sc = SystemCollection(sample_systems, sample_molecules)

        names = sc.name

        assert len(names) == 2
        assert "pure_MOL1" in names
        assert "test_system" in names

    def test_getattr_from_system_properties(self, sample_systems, sample_molecules):
        """Test __getattr__ retrieves attributes from SystemProperties."""
        # Add a custom attribute to props
        for s in sample_systems:
            s.props.custom_attr = 42.0

        sc = SystemCollection(sample_systems, sample_molecules)

        values = sc.custom_attr

        assert len(values) == 2
        assert all(v == 42.0 for v in values)

    def test_getattr_callable_attribute(self, sample_systems, sample_molecules):
        """Test __getattr__ handles callable attributes."""
        sc = SystemCollection(sample_systems, sample_molecules)

        is_pure_values = sc.is_pure

        assert len(is_pure_values) == 2
        assert bool(is_pure_values[0]) is True
        assert bool(is_pure_values[1]) is False

    def test_getattr_returns_numpy_array_for_numeric(self, sample_systems, sample_molecules):
        """Test __getattr__ returns numpy array for numeric values."""
        for i, s in enumerate(sample_systems):
            s.numeric_value = float(i)

        sc = SystemCollection(sample_systems, sample_molecules)

        values = sc.numeric_value

        assert isinstance(values, np.ndarray)
        assert values.dtype in [np.float64, np.int64]

    def test_getattr_returns_numpy_array_for_boolean(self, sample_systems, sample_molecules):
        """Test __getattr__ returns numpy array for boolean values."""
        for i, s in enumerate(sample_systems):
            s.bool_value = bool(i)

        sc = SystemCollection(sample_systems, sample_molecules)

        values = sc.bool_value

        assert isinstance(values, np.ndarray)

    def test_getattr_uses_props_get_fallback(self, sample_systems, sample_molecules):
        """Test __getattr__ uses props.get as fallback."""
        for s in sample_systems:
            s.props.get.return_value = "fallback_value"

        sc = SystemCollection(sample_systems, sample_molecules)

        values = sc.nonexistent_attr

        for s in sample_systems:
            s.props.get.assert_called_with("nonexistent_attr")

    def test_getattr_empty_systems(self):
        """Test __getattr__ with empty systems list."""
        sc = SystemCollection([], [])

        result = sc.any_attribute

        assert result == []

    def test_getitem_by_name(self, sample_systems, sample_molecules):
        """Test __getitem__ with string key."""
        sc = SystemCollection(sample_systems, sample_molecules)

        system = sc["test_system"]

        assert system.name == "test_system"

    def test_getitem_by_index(self, sample_systems, sample_molecules):
        """Test __getitem__ with integer index."""
        sc = SystemCollection(sample_systems, sample_molecules)

        system = sc[0]

        assert system == sample_systems[0]

    def test_getitem_by_negative_index(self, sample_systems, sample_molecules):
        """Test __getitem__ with negative index."""
        sc = SystemCollection(sample_systems, sample_molecules)

        system = sc[-1]

        assert system == sample_systems[-1]

    def test_len(self, sample_systems, sample_molecules):
        """Test __len__ returns number of systems."""
        sc = SystemCollection(sample_systems, sample_molecules)

        assert len(sc) == 2

    def test_iter(self, sample_systems, sample_molecules):
        """Test __iter__ allows iteration."""
        sc = SystemCollection(sample_systems, sample_molecules)

        systems_list = list(sc)

        assert len(systems_list) == 2
        assert systems_list == sample_systems

    def test_molecules_property(self, sample_systems, sample_molecules):
        """Test molecules property."""
        sc = SystemCollection(sample_systems, sample_molecules)

        assert sc.molecules == sample_molecules

    def test_get_mol_index(self, sample_systems, sample_molecules):
        """Test get_mol_index method."""
        sc = SystemCollection(sample_systems, sample_molecules)

        assert sc.get_mol_index("MOL1") == 0
        assert sc.get_mol_index("MOL2") == 1

    def test_get_mol_index_raises_error(self, sample_systems, sample_molecules):
        """Test get_mol_index raises error for invalid molecule."""
        sc = SystemCollection(sample_systems, sample_molecules)

        with pytest.raises(ValueError, match="Molecule 'MOL3' is not in molecules"):
            sc.get_mol_index("MOL3")

    def test_n_i_property(self, sample_systems, sample_molecules):
        """Test n_i property returns number of components."""
        sc = SystemCollection(sample_systems, sample_molecules)

        assert sc.n_i == 2

    def test_n_sys_property(self, sample_systems, sample_molecules):
        """Test n_sys property returns number of systems."""
        sc = SystemCollection(sample_systems, sample_molecules)

        assert sc.n_sys == 2

    def test_x_property(self, sample_systems, sample_molecules):
        """Test x property returns mole fraction array."""
        sc = SystemCollection(sample_systems, sample_molecules)

        x = sc.x

        assert x.shape == (2, 2)
        assert np.allclose(x[0], [1.0, 0.0])
        assert np.allclose(x[1], [0.5, 0.5])

    def test_x_property_cached(self, sample_systems, sample_molecules):
        """Test x property is cached."""
        sc = SystemCollection(sample_systems, sample_molecules)

        x1 = sc.x
        x2 = sc.x

        assert x1 is x2

    def test_units_property(self, sample_systems, sample_molecules):
        """Test units property aggregates units from all systems."""
        for s in sample_systems:
            s.props.get.return_value = {"Temperature": "K", "Pressure": "bar"}

        sc = SystemCollection(sample_systems, sample_molecules)

        units = sc.units

        assert isinstance(units, dict)

    def test_pures_property(self, sample_systems, sample_molecules):
        """Test pures property returns only pure systems."""
        sc = SystemCollection(sample_systems, sample_molecules)

        pures = sc.pures

        assert len(pures) == 1
        assert pures[0].name == "pure_MOL1"

    def test_mixtures_property(self, sample_systems, sample_molecules):
        """Test mixtures property returns only mixture systems."""
        sc = SystemCollection(sample_systems, sample_molecules)

        mixtures = sc.mixtures

        assert len(mixtures) == 1
        assert mixtures[0].name == "test_system"


class TestSystemCollectionGetMethods:
    """Test SystemCollection get methods."""

    def test_get_units(self, sample_systems, sample_molecules):
        """Test get_units method."""
        sc = SystemCollection(sample_systems, sample_molecules)

        with patch.object(SystemCollection, 'units', new_callable=PropertyMock) as mock_units:
            mock_units.return_value = {"temperature": "K", "pressure": "bar"}
            sc._units = {"temperature": "K", "pressure": "bar"}

            assert sc.get_units("Temperature") == "K"
            assert sc.get_units("pressure") == "bar"

    def test_get_units_returns_empty_string_for_unknown(self, sample_systems, sample_molecules):
        """Test get_units returns empty string for unknown property."""
        sc = SystemCollection(sample_systems, sample_molecules)

        with patch.object(SystemCollection, 'units', new_callable=PropertyMock) as mock_units:
            mock_units.return_value = {}

            assert sc.get_units("UnknownProperty") == ""

    def test_get_property_averaged(self, sample_systems, sample_molecules):
        """Test get method with averaging."""
        for i, s in enumerate(sample_systems):
            s.props.get.return_value = float(i + 1)

        sc = SystemCollection(sample_systems, sample_molecules)

        values = sc.get("Temperature", avg=True)

        assert isinstance(values, np.ndarray)
        assert len(values) == 2

    def test_get_property_time_series(self, sample_systems, sample_molecules):
        """Test get method with time series."""
        for s in sample_systems:
            s.props.get.return_value = np.array([1.0, 2.0, 3.0])

        sc = SystemCollection(sample_systems, sample_molecules)

        values = sc.get("Temperature", avg=False, time_series=True)

        assert isinstance(values, (list, np.ndarray))

    def test_get_property_returns_list_on_value_error(self, sample_systems, sample_molecules):
        """Test get method returns list when numpy array conversion fails."""
        for i, s in enumerate(sample_systems):
            s.props.get.return_value = [1, 2, 3] if i == 0 else [1, 2]

        sc = SystemCollection(sample_systems, sample_molecules)

        values = sc.get("Temperature", avg=False)

        assert isinstance(values, list)

    def test_get_property_with_units(self, sample_systems, sample_molecules):
        """Test get method with unit conversion."""
        for s in sample_systems:
            s.props.get.return_value = 298.15

        sc = SystemCollection(sample_systems, sample_molecules)

        values = sc.get("Temperature", units="K", avg=True)

        for s in sample_systems:
            s.props.get.assert_called_with("Temperature", units="K", avg=True, time_series=False)

    def test_get_from_cache_returns_none_when_empty(self, sample_systems, sample_molecules):
        """Test _get_from_cache returns None when cache is empty."""
        sc = SystemCollection(sample_systems, sample_molecules)

        result = sc._get_from_cache(("test",), "kg/m^3")

        assert result is None

    def test_get_from_cache_returns_cached_result(self, sample_systems, sample_molecules):
        """Test _get_from_cache returns cached result."""
        sc = SystemCollection(sample_systems, sample_molecules)

        mock_result = Mock(spec=PropertyResult)
        mock_result.to.return_value = mock_result

        key = ("test",)
        sc._cache[key] = mock_result

        result = sc._get_from_cache(key, "kg/m^3")

        assert result == mock_result
        mock_result.to.assert_called_once_with("kg/m^3")

    def test_has_all_required_pures_returns_true(self, sample_systems, sample_molecules):
        """Test has_all_required_pures returns True when all pures present."""
        sc = SystemCollection(sample_systems, sample_molecules)

        pure = sc.pures[0]
        pure.props.topology.molecules = ["MOL1"]

        pure2 = Mock(spec=SystemMetadata)
        pure2.props = Mock(spec=SystemProperties)
        pure2.props.topology = Mock()
        pure2.props.topology.molecules = ["MOL2"]
        pure2.is_pure.return_value = True

        sc._systems.append(pure2)

        result = sc.has_all_required_pures()

        assert result is True

    def test_has_all_required_pures_returns_false_when_no_pures(self, sample_molecules):
        """Test has_all_required_pures returns False when no pures."""
        sc = SystemCollection([], sample_molecules)

        result = sc.has_all_required_pures()

        assert result is False

    def test_has_all_required_pures_returns_false_when_missing_molecule(self, sample_systems, sample_molecules):
        """Test has_all_required_pures returns False when molecule missing."""
        sc = SystemCollection(sample_systems, sample_molecules)

        pure = sc.pures[0]
        pure.props.topology.molecules = ["MOL1"]

        result = sc.has_all_required_pures()

        assert result is False


class TestSystemCollectionPropertyMethods:
    """Test property calculation methods."""

    def test_ideal_property_linear_mixing(self, sample_systems, sample_molecules):
        """Test ideal_property with linear mixing rule."""
        # Create a fresh SystemCollection with both pures
        pure1 = Mock(spec=SystemMetadata)
        pure1.name = "pure_MOL1"
        pure1.kind = "pure"
        pure1.is_pure.return_value = True
        pure1_props = Mock(spec=SystemProperties)
        pure1_topology = Mock()
        pure1_topology.molecule_count = {"MOL1": 100}
        pure1_topology.total_molecules = 100
        pure1_topology.molecules = ["MOL1"]
        pure1_props.topology = pure1_topology
        pure1_props.get.return_value = 1000.0
        pure1.props = pure1_props

        pure2 = Mock(spec=SystemMetadata)
        pure2.name = "pure_MOL2"
        pure2.kind = "pure"
        pure2.is_pure.return_value = True
        pure2_props = Mock(spec=SystemProperties)
        pure2_topology = Mock()
        pure2_topology.molecule_count = {"MOL2": 100}
        pure2_topology.total_molecules = 100
        pure2_topology.molecules = ["MOL2"]
        pure2_props.topology = pure2_topology
        pure2_props.get.return_value = 800.0
        pure2.props = pure2_props

        mixture = Mock(spec=SystemMetadata)
        mixture.name = "mixture"
        mixture.kind = "mixture"
        mixture.is_pure.return_value = False
        mixture_props = Mock(spec=SystemProperties)
        mixture_topology = Mock()
        mixture_topology.molecule_count = {"MOL1": 50, "MOL2": 50}
        mixture_topology.total_molecules = 100
        mixture_props.topology = mixture_topology
        mixture_props.get.return_value = 920.0
        mixture.props = mixture_props

        sc = SystemCollection([pure1, mixture, pure2], sample_molecules)

        result = sc.ideal_property(name="Density", mixing_rule="linear", units="kg/m^3", avg=True)

        assert isinstance(result, PropertyResult)
        assert result.property_type == "ideal"
        # Pure MOL1: 1.0 * 1000 + 0.0 * 800 = 1000
        # Mixture: 0.5 * 1000 + 0.5 * 800 = 900
        # Pure MOL2: 0.0 * 1000 + 1.0 * 800 = 800
        np.testing.assert_array_almost_equal(result.value, [1000.0, 900.0, 800.0])

    def test_ideal_property_volume_weighted_mixing(self, sample_systems, sample_molecules):
        """Test ideal_property with volume-weighted mixing rule."""
        # Create a fresh SystemCollection with both pures
        pure1 = Mock(spec=SystemMetadata)
        pure1.name = "pure_MOL1"
        pure1.kind = "pure"
        pure1.is_pure.return_value = True
        pure1_props = Mock(spec=SystemProperties)
        pure1_topology = Mock()
        pure1_topology.molecule_count = {"MOL1": 100}
        pure1_topology.total_molecules = 100
        pure1_topology.molecules = ["MOL1"]
        pure1_props.topology = pure1_topology
        pure1_props.get.return_value = 1000.0
        pure1.props = pure1_props

        pure2 = Mock(spec=SystemMetadata)
        pure2.name = "pure_MOL2"
        pure2.kind = "pure"
        pure2.is_pure.return_value = True
        pure2_props = Mock(spec=SystemProperties)
        pure2_topology = Mock()
        pure2_topology.molecule_count = {"MOL2": 100}
        pure2_topology.total_molecules = 100
        pure2_topology.molecules = ["MOL2"]
        pure2_props.topology = pure2_topology
        pure2_props.get.return_value = 800.0
        pure2.props = pure2_props

        mixture = Mock(spec=SystemMetadata)
        mixture.name = "mixture"
        mixture.kind = "mixture"
        mixture.is_pure.return_value = False
        mixture_props = Mock(spec=SystemProperties)
        mixture_topology = Mock()
        mixture_topology.molecule_count = {"MOL1": 50, "MOL2": 50}
        mixture_topology.total_molecules = 100
        mixture_props.topology = mixture_topology
        mixture_props.get.return_value = 920.0
        mixture.props = mixture_props

        sc = SystemCollection([pure1, mixture, pure2], sample_molecules)

        result = sc.ideal_property(name="Density", mixing_rule="volume_weighted", units="kg/m^3", avg=True)

        # Pure MOL1: 1 / (1.0/1000 + 0.0/800) = 1000
        # Mixture: 1 / (0.5/1000 + 0.5/800) = 888.89
        # Pure MOL2: 1 / (0.0/1000 + 1.0/800) = 800
        assert result.value[0] == pytest.approx(1000.0)
        assert result.value[1] == pytest.approx(888.89, rel=1e-2)
        assert result.value[2] == pytest.approx(800.0)

    def test_ideal_property_raises_on_invalid_mixing_rule(self, sample_molecules):
        """Test ideal_property raises error for invalid mixing rule."""
        # Create a fresh SystemCollection with both pures
        pure1 = Mock(spec=SystemMetadata)
        pure1.name = "pure_MOL1"
        pure1.is_pure.return_value = True
        pure1_props = Mock(spec=SystemProperties)
        pure1_topology = Mock()
        pure1_topology.molecule_count = {"MOL1": 100}
        pure1_topology.total_molecules = 100
        pure1_topology.molecules = ["MOL1"]
        pure1_props.topology = pure1_topology
        pure1_props.get.return_value = 1000.0
        pure1.props = pure1_props

        pure2 = Mock(spec=SystemMetadata)
        pure2.name = "pure_MOL2"
        pure2.is_pure.return_value = True
        pure2_props = Mock(spec=SystemProperties)
        pure2_topology = Mock()
        pure2_topology.molecule_count = {"MOL2": 100}
        pure2_topology.total_molecules = 100
        pure2_topology.molecules = ["MOL2"]
        pure2_props.topology = pure2_topology
        pure2_props.get.return_value = 800.0
        pure2.props = pure2_props

        sc = SystemCollection([pure1, pure2], sample_molecules)

        with pytest.raises(ValueError, match="Unknown mixing rule"):
            sc.ideal_property(name="Density", mixing_rule="invalid", units="kg/m^3", avg=True)

    def test_excess_property(self, sample_molecules):
        """Test excess_property method."""
        # Create a fresh SystemCollection with both pures and mixture
        pure1 = Mock(spec=SystemMetadata)
        pure1.name = "pure_MOL1"
        pure1.is_pure.return_value = True
        pure1_props = Mock(spec=SystemProperties)
        pure1_topology = Mock()
        pure1_topology.molecule_count = {"MOL1": 100}
        pure1_topology.total_molecules = 100
        pure1_topology.molecules = ["MOL1"]
        pure1_props.topology = pure1_topology
        pure1_props.get.return_value = 1000.0
        pure1.props = pure1_props

        mixture = Mock(spec=SystemMetadata)
        mixture.name = "mixture"
        mixture.is_pure.return_value = False
        mixture_props = Mock(spec=SystemProperties)
        mixture_topology = Mock()
        mixture_topology.molecule_count = {"MOL1": 50, "MOL2": 50}
        mixture_topology.total_molecules = 100
        mixture_props.topology = mixture_topology
        mixture_props.get.return_value = 920.0
        mixture.props = mixture_props

        pure2 = Mock(spec=SystemMetadata)
        pure2.name = "pure_MOL2"
        pure2.is_pure.return_value = True
        pure2_props = Mock(spec=SystemProperties)
        pure2_topology = Mock()
        pure2_topology.molecule_count = {"MOL2": 100}
        pure2_topology.total_molecules = 100
        pure2_topology.molecules = ["MOL2"]
        pure2_props.topology = pure2_topology
        pure2_props.get.return_value = 800.0
        pure2.props = pure2_props

        sc = SystemCollection([pure1, mixture, pure2], sample_molecules)

        result = sc.excess_property(name="Density", mixing_rule="linear", units="kg/m^3", avg=True)

        assert isinstance(result, PropertyResult)
        assert result.property_type == "excess"
        # Pure MOL1: 1000 - 1000 = 0
        # Mixture: 920 - 900 = 20
        # Pure MOL2: 800 - 800 = 0
        np.testing.assert_array_almost_equal(result.value, [0.0, 20.0, 0.0])



class TestSystemCollectionPlotters:
    """Test plotter creation methods."""

    @patch('kbkit.systems.collection.TimeseriesPlotter.from_collection')
    def test_timeseries_plotter(self, mock_from_collection, sample_systems, sample_molecules):
        """Test timeseries_plotter method."""
        mock_plotter = Mock(spec=TimeseriesPlotter)
        mock_from_collection.return_value = mock_plotter

        sc = SystemCollection(sample_systems, sample_molecules)

        result = sc.timeseries_plotter("test_system", start_time=5000)

        mock_from_collection.assert_called_once_with(
            sc, system_name="test_system", start_time=5000
        )
        assert result == mock_plotter


class TestSystemCollectionLoad:
    """Test the load class method."""

    @patch('kbkit.systems.collection.SystemCollection._sort_systems')
    @patch('kbkit.systems.collection.SystemCollection._make_meta')
    @patch('kbkit.systems.collection.SystemCollection._resolve_rdf_path')
    @patch('kbkit.systems.collection.SystemCollection._find_pure_systems')
    @patch('kbkit.systems.collection.SystemCollection._find_reference_dir')
    @patch('kbkit.systems.collection.SystemCollection._extract_temp')
    @patch('kbkit.systems.collection.SystemCollection._peek_molecules')
    @patch('kbkit.systems.collection.SystemCollection._is_valid')
    @patch('kbkit.systems.collection.validate_path')
    def test_load_with_explicit_base_systems(
        self, mock_validate_path, mock_is_valid, mock_peek_molecules,
        mock_extract_temp, mock_find_reference_dir, mock_find_pure_systems,
        mock_resolve_rdf_path, mock_make_meta, mock_sort_systems, tmp_path
    ):
        """Test load with explicitly specified base systems."""
        base_path = tmp_path / "mixtures"
        base_path.mkdir()

        sys1 = base_path / "sys1"
        sys1.mkdir()

        mock_validate_path.return_value = base_path
        mock_is_valid.return_value = True
        mock_peek_molecules.return_value = {"MOL1", "MOL2"}
        mock_extract_temp.return_value = 298.0
        mock_find_reference_dir.return_value = None
        mock_resolve_rdf_path.return_value = Path()

        mock_meta = Mock(spec=SystemMetadata)
        mock_meta.name = "sys1"
        mock_topology = Mock()
        mock_topology.molecule_count = {"MOL1": 50, "MOL2": 50}
        mock_topology.total_molecules = 100
        mock_props = Mock(spec=SystemProperties)
        mock_props.topology = mock_topology
        mock_meta.props = mock_props

        mock_make_meta.return_value = mock_meta
        mock_sort_systems.return_value = [mock_meta]

        result = SystemCollection.load(
            base_path=str(base_path),
            base_systems=["sys1"]
        )

        assert isinstance(result, SystemCollection)
        assert len(result._systems) == 1

    @patch('kbkit.systems.collection.SystemCollection._sort_systems')
    @patch('kbkit.systems.collection.SystemCollection._make_meta')
    @patch('kbkit.systems.collection.SystemCollection._resolve_rdf_path')
    @patch('kbkit.systems.collection.SystemCollection._find_pure_systems')
    @patch('kbkit.systems.collection.SystemCollection._find_reference_dir')
    @patch('kbkit.systems.collection.SystemCollection._extract_temp')
    @patch('kbkit.systems.collection.SystemCollection._peek_molecules')
    @patch('kbkit.systems.collection.SystemCollection._is_valid')
    @patch('kbkit.systems.collection.validate_path')
    def test_load_discovers_systems_automatically(
        self, mock_validate_path, mock_is_valid, mock_peek_molecules,
        mock_extract_temp, mock_find_reference_dir, mock_find_pure_systems,
        mock_resolve_rdf_path, mock_make_meta, mock_sort_systems, tmp_path
    ):
        """Test load discovers systems automatically."""
        base_path = tmp_path / "mixtures"
        base_path.mkdir()

        sys1 = base_path / "sys1"
        sys1.mkdir()

        mock_validate_path.return_value = base_path
        mock_is_valid.return_value = True
        mock_peek_molecules.return_value = {"MOL1", "MOL2"}
        mock_extract_temp.return_value = 298.0
        mock_find_reference_dir.return_value = None
        mock_resolve_rdf_path.return_value = Path()

        mock_meta = Mock(spec=SystemMetadata)
        mock_meta.name = "sys1"
        mock_topology = Mock()
        mock_topology.molecule_count = {"MOL1": 50, "MOL2": 50}
        mock_topology.total_molecules = 100
        mock_props = Mock(spec=SystemProperties)
        mock_props.topology = mock_topology
        mock_meta.props = mock_props

        mock_make_meta.return_value = mock_meta
        mock_sort_systems.return_value = [mock_meta]

        result = SystemCollection.load(base_path=str(base_path))

        assert isinstance(result, SystemCollection)

    @patch('kbkit.systems.collection.SystemCollection._sort_systems')
    @patch('kbkit.systems.collection.SystemCollection._make_meta')
    @patch('kbkit.systems.collection.SystemCollection._resolve_rdf_path')
    @patch('kbkit.systems.collection.SystemCollection._find_pure_systems')
    @patch('kbkit.systems.collection.SystemCollection._find_reference_dir')
    @patch('kbkit.systems.collection.SystemCollection._extract_temp')
    @patch('kbkit.systems.collection.SystemCollection._peek_molecules')
    @patch('kbkit.systems.collection.SystemCollection._is_valid')
    @patch('kbkit.systems.collection.validate_path')
    def test_load_with_explicit_pure_systems(
        self, mock_validate_path, mock_is_valid, mock_peek_molecules,
        mock_extract_temp, mock_find_reference_dir, mock_find_pure_systems,
        mock_resolve_rdf_path, mock_make_meta, mock_sort_systems, tmp_path
    ):
        """Test load with explicitly specified pure systems."""
        base_path = tmp_path / "mixtures"
        base_path.mkdir()
        pure_path = tmp_path / "pure"
        pure_path.mkdir()

        sys1 = base_path / "sys1"
        sys1.mkdir()
        pure_mol1 = pure_path / "MOL1"
        pure_mol1.mkdir()

        mock_validate_path.side_effect = lambda x: Path(x)
        mock_is_valid.return_value = True
        mock_peek_molecules.return_value = {"MOL1", "MOL2"}
        mock_extract_temp.return_value = 298.0
        mock_resolve_rdf_path.return_value = Path()

        mock_meta_mixture = Mock(spec=SystemMetadata)
        mock_meta_mixture.name = "sys1"
        mock_meta_pure = Mock(spec=SystemMetadata)
        mock_meta_pure.name = "MOL1"

        for mock_meta, counts in [(mock_meta_mixture, {"MOL1": 50, "MOL2": 50}),
                                   (mock_meta_pure, {"MOL1": 100})]:
            mock_topology = Mock()
            mock_topology.molecule_count = counts
            mock_topology.total_molecules = sum(counts.values())
            mock_props = Mock(spec=SystemProperties)
            mock_props.topology = mock_topology
            mock_meta.props = mock_props

        mock_make_meta.side_effect = [mock_meta_pure, mock_meta_mixture]
        mock_sort_systems.return_value = [mock_meta_pure, mock_meta_mixture]

        result = SystemCollection.load(
            base_path=str(base_path),
            pure_path=str(pure_path),
            pure_systems=["MOL1"]
        )

        assert isinstance(result, SystemCollection)

    @patch('kbkit.systems.collection.SystemCollection._sort_systems')
    @patch('kbkit.systems.collection.SystemCollection._make_meta')
    @patch('kbkit.systems.collection.SystemCollection._resolve_rdf_path')
    @patch('kbkit.systems.collection.SystemCollection._find_pure_systems')
    @patch('kbkit.systems.collection.SystemCollection._find_reference_dir')
    @patch('kbkit.systems.collection.SystemCollection._extract_temp')
    @patch('kbkit.systems.collection.SystemCollection._peek_molecules')
    @patch('kbkit.systems.collection.SystemCollection._is_valid')
    @patch('kbkit.systems.collection.validate_path')
    def test_load_discovers_pure_systems_automatically(
        self, mock_validate_path, mock_is_valid, mock_peek_molecules,
        mock_extract_temp, mock_find_reference_dir, mock_find_pure_systems,
        mock_resolve_rdf_path, mock_make_meta, mock_sort_systems, tmp_path
    ):
        """Test load discovers pure systems automatically."""
        base_path = tmp_path / "mixtures"
        base_path.mkdir()
        pure_path = tmp_path / "pure"
        pure_path.mkdir()

        sys1 = base_path / "sys1"
        sys1.mkdir()
        pure_mol1 = pure_path / "MOL1"
        pure_mol1.mkdir()

        mock_validate_path.side_effect = lambda x: Path(x) if x else base_path
        mock_is_valid.return_value = True
        mock_peek_molecules.return_value = {"MOL1", "MOL2"}
        mock_extract_temp.return_value = 298.0
        mock_find_reference_dir.return_value = pure_path
        mock_find_pure_systems.return_value = {"MOL1": pure_mol1}
        mock_resolve_rdf_path.return_value = Path()

        mock_meta_mixture = Mock(spec=SystemMetadata)
        mock_meta_mixture.name = "sys1"
        mock_meta_pure = Mock(spec=SystemMetadata)
        mock_meta_pure.name = "MOL1"

        for mock_meta, counts in [(mock_meta_mixture, {"MOL1": 50, "MOL2": 50}),
                                   (mock_meta_pure, {"MOL1": 100})]:
            mock_topology = Mock()
            mock_topology.molecule_count = counts
            mock_topology.total_molecules = sum(counts.values())
            mock_props = Mock(spec=SystemProperties)
            mock_props.topology = mock_topology
            mock_meta.props = mock_props

        mock_make_meta.side_effect = [mock_meta_pure, mock_meta_mixture]
        mock_sort_systems.return_value = [mock_meta_pure, mock_meta_mixture]

        result = SystemCollection.load(base_path=str(base_path))

        assert isinstance(result, SystemCollection)
        mock_find_pure_systems.assert_called_once()

    @patch('kbkit.systems.collection.SystemCollection._sort_systems')
    @patch('kbkit.systems.collection.SystemCollection._make_meta')
    @patch('kbkit.systems.collection.SystemCollection._resolve_rdf_path')
    @patch('kbkit.systems.collection.SystemCollection._find_reference_dir')
    @patch('kbkit.systems.collection.SystemCollection._peek_molecules')
    @patch('kbkit.systems.collection.SystemCollection._is_valid')
    @patch('kbkit.systems.collection.validate_path')
    def test_load_uses_current_directory_by_default(
        self, mock_validate_path, mock_is_valid, mock_peek_molecules,
        mock_resolve_rdf_path, mock_make_meta, mock_sort_systems, tmp_path
    ):
        """Test load uses current directory when base_path is None."""
        mock_validate_path.return_value = tmp_path
        mock_is_valid.return_value = False
        mock_peek_molecules.return_value = set()
        mock_sort_systems.return_value = []

        with patch('os.getcwd', return_value=str(tmp_path)):
            result = SystemCollection.load()

            assert isinstance(result, SystemCollection)


class TestSystemCollectionIntegration:
    """Integration tests for SystemCollection."""

    def test_full_workflow_with_mocked_systems(self):
        """Test complete workflow from initialization to property access."""
        systems = []
        for i in range(3):
            mock_meta = Mock(spec=SystemMetadata)
            mock_meta.name = f"system_{i}"
            mock_meta.kind = "mixture" if i > 0 else "pure"

            mock_props = Mock(spec=SystemProperties)
            mock_topology = Mock()
            mock_topology.molecule_count = {"MOL1": 100 - i * 30, "MOL2": i * 30}
            mock_topology.total_molecules = 100
            mock_props.topology = mock_topology
            mock_props.get.return_value = {"Temperature": "K"}

            mock_meta.props = mock_props
            mock_meta.is_pure.return_value = (i == 0)

            systems.append(mock_meta)

        molecules = ["MOL1", "MOL2"]

        sc = SystemCollection(systems, molecules)

        assert len(sc) == 3
        assert sc.n_i == 2
        assert sc.n_sys == 3

        x = sc.x
        assert x.shape == (3, 2)

        assert len(sc.pures) == 1
        assert len(sc.mixtures) == 2

        assert sc[0].name == "system_0"
        assert sc["system_1"].name == "system_1"
