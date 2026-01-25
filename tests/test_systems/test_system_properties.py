"""
Unit tests for the SystemProperties module.

This test suite provides comprehensive coverage of the SystemProperties class,
including file discovery, property access, unit conversion, and plotting.
"""
import warnings

# Suppress NumPy/SciPy compatibility warning (harmless with NumPy 2.x + SciPy 1.16+)
warnings.filterwarnings('ignore', message='numpy.ndarray size changed', category=RuntimeWarning)




from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pytest

from kbkit.io import EdrParser, GroParser, TopParser
from kbkit.systems.properties import SystemProperties
from kbkit.visualization.timeseries import TimeseriesPlotter


@pytest.fixture
def mock_edr_parser():
    """Create a mock EdrParser object."""
    mock_edr = Mock(spec=EdrParser)
    mock_edr.units = {
        "Temperature": "K",
        "Pressure": "bar",
        "Volume": "nm^3",
        "molar-volume": "cm^3/mol"
    }
    mock_edr.get.return_value = np.array([298.15, 298.20, 298.25])
    mock_edr.cp.return_value = np.array([75.0, 75.5, 76.0])
    mock_edr.cv.return_value = np.array([50.0, 50.5, 51.0])
    mock_edr.molar_enthalpy.return_value = np.array([10.0, 10.5, 11.0])
    mock_edr.isothermal_compressibility.return_value = np.array([1e-5, 1.1e-5, 1.2e-5])
    mock_edr.molar_volume.return_value = np.array([18.0, 18.1, 18.2])
    return mock_edr




@pytest.fixture
def mock_gro_parser():
    """Create a mock GroParser object."""
    mock_gro = Mock(spec=GroParser)
    mock_gro.molecules = [["MOL1"] * 50, ["MOL2"] * 50]
    mock_gro.total_molecules = 100
    mock_gro.box_volume = 125.0
    mock_gro.electron_count = {"MOL1": 10, "MOL2": 8}
    mock_gro.molecule_count = {"MOL1": 50, "MOL2": 50}
    return mock_gro




@pytest.fixture
def mock_top_parser():
    """Create a mock TopParser object."""
    mock_top = Mock(spec=TopParser)
    mock_top.total_molecules = 100
    mock_top.box_volume = 125.0
    mock_top.molecule_count = {"MOL1": 50, "MOL2": 50}
    return mock_top




class TestSystemPropertiesInitialization:
    """Test SystemProperties initialization."""

    @patch('kbkit.systems.properties.SystemProperties.find_files')
    def test_init_with_system_path(self, mock_find_files, tmp_path):
        """Test initialization with system_path."""
        edr_file = tmp_path / "system.edr"
        gro_file = tmp_path / "system.gro"
        top_file = tmp_path / "system.top"

        mock_find_files.side_effect = lambda **kwargs: {
            '.edr': [edr_file],
            '.gro': [gro_file],
            '.top': [top_file]
        }.get(kwargs.get('suffix'), [])

        with patch('kbkit.systems.properties.validate_path', return_value=tmp_path):
            props = SystemProperties(system_path=str(tmp_path))

        assert props.system_path == tmp_path
        assert props.start_time == 0.0

    @patch('kbkit.systems.properties.SystemProperties.find_files')
    def test_init_with_explicit_paths(self, mock_find_files, tmp_path):
        """Test initialization with explicit file paths."""
        edr_file = tmp_path / "system.edr"
        gro_file = tmp_path / "system.gro"
        top_file = tmp_path / "system.top"

        mock_find_files.side_effect = lambda **kwargs: {
            '.edr': [edr_file],
            '.gro': [gro_file],
            '.top': [top_file]
        }.get(kwargs.get('suffix'), [])

        with patch('kbkit.systems.properties.validate_path') as mock_validate:
            mock_validate.side_effect = lambda x, *args: Path(x) if x else None

            props = SystemProperties(
                edr_path=str(edr_file),
                gro_path=str(gro_file),
                top_path=str(top_file),
            )

        assert len(props.edr_paths) > 0
        assert len(props.gro_paths) > 0
        assert len(props.top_paths) > 0

    @patch('kbkit.systems.properties.SystemProperties.find_files')
    def test_init_with_custom_start_time(self, mock_find_files, tmp_path):
        """Test initialization with custom start_time."""
        edr_file = tmp_path / "system.edr"
        gro_file = tmp_path / "system.gro"

        mock_find_files.side_effect = lambda **kwargs: {
            '.edr': [edr_file],
            '.gro': [gro_file],
            '.top': []
        }.get(kwargs.get('suffix'), [])

        with patch('kbkit.systems.properties.validate_path', return_value=tmp_path):
            props = SystemProperties(system_path=str(tmp_path), start_time=5000)

        assert props.start_time == 5000.0

    @patch('kbkit.systems.properties.SystemProperties.find_files')
    def test_init_converts_start_time_to_float(self, mock_find_files, tmp_path):
        """Test that start_time is converted to float."""
        edr_file = tmp_path / "system.edr"
        gro_file = tmp_path / "system.gro"

        mock_find_files.side_effect = lambda **kwargs: {
            '.edr': [edr_file],
            '.gro': [gro_file],
            '.top': []
        }.get(kwargs.get('suffix'), [])

        with patch('kbkit.systems.properties.validate_path', return_value=tmp_path):
            props = SystemProperties(system_path=str(tmp_path), start_time=1000)

        assert isinstance(props.start_time, float)
        assert props.start_time == 1000.0

    @patch('kbkit.systems.properties.SystemProperties.find_files')
    def test_init_without_system_path_validation(self, mock_find_files):
        """Test initialization without system_path doesn't validate."""
        mock_find_files.return_value = []

        props = SystemProperties()

        assert props.system_path is None




class TestSystemPropertiesFindFiles:
    """Test the find_files static method."""

    def test_find_files_with_filepath(self, tmp_path):
        """Test find_files with explicit filepath."""
        edr_file = tmp_path / "system.edr"
        edr_file.touch()

        with patch('kbkit.systems.properties.validate_path') as mock_validate:
            mock_validate.side_effect = lambda x, *args: Path(x)

            files = SystemProperties.find_files(suffix=".edr", filepath=str(edr_file))

        assert len(files) == 1
        assert files[0] == edr_file

    def test_find_files_with_system_path(self, tmp_path):
        """Test find_files searching in system_path."""
        (tmp_path / "system.edr").touch()

        with patch('kbkit.systems.properties.validate_path', return_value=tmp_path):
            files = SystemProperties.find_files(suffix=".edr", system_path=str(tmp_path))

        assert len(files) == 1

    def test_find_files_with_include_filter(self, tmp_path):
        """Test find_files with include filter."""
        (tmp_path / "npt_system.edr").touch()
        (tmp_path / "nvt_system.edr").touch()

        with patch('kbkit.systems.properties.validate_path', return_value=tmp_path):
            files = SystemProperties.find_files(
                suffix=".edr",
                system_path=str(tmp_path),
                include="npt"
            )

        assert len(files) == 1
        assert "npt" in files[0].name

    def test_find_files_with_exclude_filter(self, tmp_path):
        """Test find_files with exclude filter."""
        (tmp_path / "system.edr").touch()
        (tmp_path / "init_system.edr").touch()
        (tmp_path / "eq_system.edr").touch()

        with patch('kbkit.systems.properties.validate_path', return_value=tmp_path):
            files = SystemProperties.find_files(
                suffix=".edr",
                system_path=str(tmp_path),
                exclude=["init", "eq"]
            )

        # Should exclude init and eq files
        assert len(files) >= 1
        assert not any("init" in f.name or "eq" in f.name for f in files)

    def test_find_files_raises_without_paths(self):
        """Test find_files raises error without filepath or system_path."""
        with pytest.raises(ValueError, match="A valid 'filepath' or 'system_path' is required"):
            SystemProperties.find_files(suffix=".edr")

    def test_find_files_raises_when_no_files_found(self, tmp_path):
        """Test find_files raises error when no files found."""
        with patch('kbkit.systems.properties.validate_path', return_value=tmp_path):
            with pytest.raises(ValueError, match="No files with '.edr' found"):
                SystemProperties.find_files(suffix=".edr", system_path=str(tmp_path))

    def test_find_files_strips_dot_from_suffix(self, tmp_path):
        """Test that find_files handles suffix with or without dot."""
        (tmp_path / "system.edr").touch()

        with patch('kbkit.systems.properties.validate_path', return_value=tmp_path):
            files1 = SystemProperties.find_files(suffix=".edr", system_path=str(tmp_path))
            files2 = SystemProperties.find_files(suffix="edr", system_path=str(tmp_path))

        assert files1 == files2

    def test_find_files_returns_sorted(self, tmp_path):
        """Test that find_files returns sorted list."""
        (tmp_path / "c_system.edr").touch()
        (tmp_path / "a_system.edr").touch()
        (tmp_path / "b_system.edr").touch()

        with patch('kbkit.systems.properties.validate_path', return_value=tmp_path):
            files = SystemProperties.find_files(suffix=".edr", system_path=str(tmp_path))

        file_names = [f.name for f in files]
        assert file_names == sorted(file_names)




class TestSystemPropertiesEnergyProperty:
    """Test the energy cached property."""

    @patch('kbkit.systems.properties.EdrParser')
    @patch('kbkit.systems.properties.SystemProperties.find_files')
    def test_energy_creates_parsers(self, mock_find_files, mock_edr_class, tmp_path):
        """Test that energy property creates EdrParser instances."""
        edr_file = tmp_path / "system.edr"

        mock_find_files.side_effect = lambda **kwargs: {
            '.edr': [edr_file],
            '.gro': [tmp_path / "system.gro"],
            '.top': []
        }.get(kwargs.get('suffix'), [])

        mock_parser = Mock(spec=EdrParser)
        mock_edr_class.return_value = mock_parser

        with patch('kbkit.systems.properties.validate_path', return_value=tmp_path):
            props = SystemProperties(system_path=str(tmp_path))
            energy_parsers = props.energy

        assert len(energy_parsers) > 0
        mock_edr_class.assert_called()

    @patch('kbkit.systems.properties.SystemProperties.find_files')
    def test_energy_raises_when_no_files(self, mock_find_files, tmp_path):
        """Test that energy raises error when no EDR files found."""
        mock_find_files.return_value = []

        with patch('kbkit.systems.properties.validate_path', return_value=tmp_path):
            props = SystemProperties(system_path=str(tmp_path))
            props.edr_paths = []

        with pytest.raises(FileNotFoundError, match="Energy file\\(s\\) do not exist"):
            _ = props.energy

    @patch('kbkit.systems.properties.EdrParser')
    @patch('kbkit.systems.properties.SystemProperties.find_files')
    def test_energy_is_cached(self, mock_find_files, mock_edr_class, tmp_path):
        """Test that energy property is cached."""
        edr_file = tmp_path / "system.edr"

        mock_find_files.side_effect = lambda **kwargs: {
            '.edr': [edr_file],
            '.gro': [tmp_path / "system.gro"],
            '.top': []
        }.get(kwargs.get('suffix'), [])

        mock_parser = Mock(spec=EdrParser)
        mock_edr_class.return_value = mock_parser

        with patch('kbkit.systems.properties.validate_path', return_value=tmp_path):
            props = SystemProperties(system_path=str(tmp_path))

            energy1 = props.energy
            energy2 = props.energy

        # Should only create parsers once
        assert energy1 is energy2




class TestSystemPropertiesTopologyProperty:
    """Test the topology cached property."""

    @patch('kbkit.systems.properties.GroParser')
    @patch('kbkit.systems.properties.SystemProperties.find_files')
    def test_topology_prefers_gro_file(self, mock_find_files, mock_gro_class, tmp_path):
        """Test that topology prefers GRO file over TOP file."""
        gro_file = tmp_path / "system.gro"
        top_file = tmp_path / "system.top"

        mock_find_files.side_effect = lambda **kwargs: {
            '.edr': [tmp_path / "system.edr"],
            '.gro': [gro_file],
            '.top': [top_file]
        }.get(kwargs.get('suffix'), [])

        mock_gro = Mock(spec=GroParser)
        mock_gro.molecules = [["MOL1"] * 10]
        mock_gro_class.return_value = mock_gro

        with patch('kbkit.systems.properties.validate_path', return_value=tmp_path):
            props = SystemProperties(system_path=str(tmp_path))
            topology = props.topology

        mock_gro_class.assert_called()
        assert topology == mock_gro

    @patch('kbkit.systems.properties.TopParser')
    @patch('kbkit.systems.properties.SystemProperties.find_files')
    def test_topology_uses_top_when_no_gro(self, mock_find_files, mock_top_class, tmp_path):
        """Test that topology uses TOP file when no GRO file."""
        top_file = tmp_path / "system.top"

        mock_find_files.side_effect = lambda **kwargs: {
            '.edr': [tmp_path / "system.edr"],
            '.gro': [],
            '.top': [top_file]
        }.get(kwargs.get('suffix'), [])

        mock_top = Mock(spec=TopParser)
        mock_top_class.return_value = mock_top

        with patch('kbkit.systems.properties.validate_path', return_value=tmp_path):
            props = SystemProperties(system_path=str(tmp_path))
            topology = props.topology

        mock_top_class.assert_called()
        assert topology == mock_top

    @patch('kbkit.systems.properties.GroParser')
    @patch('kbkit.systems.properties.SystemProperties.find_files')
    def test_topology_skips_invalid_gro(self, mock_find_files, mock_gro_class, tmp_path):
        """Test that topology skips GRO files with invalid molecules."""
        gro_file = tmp_path / "system.gro"

        mock_find_files.side_effect = lambda **kwargs: {
            '.edr': [tmp_path / "system.edr"],
            '.gro': [gro_file],
            '.top': []
        }.get(kwargs.get('suffix'), [])

        # First GRO has empty molecules
        mock_gro_invalid = Mock(spec=GroParser)
        mock_gro_invalid.molecules = [[]]

        mock_gro_class.return_value = mock_gro_invalid

        with patch('kbkit.systems.properties.validate_path', return_value=tmp_path):
            props = SystemProperties(system_path=str(tmp_path))

            with pytest.raises(FileNotFoundError):
                _ = props.topology

    @patch('kbkit.systems.properties.SystemProperties.find_files')
    def test_topology_raises_when_no_files(self, mock_find_files, tmp_path):
        """Test that topology raises error when no files found."""
        mock_find_files.side_effect = lambda **kwargs: {
            '.edr': [tmp_path / "system.edr"],
            '.gro': [],
            '.top': []
        }.get(kwargs.get('suffix'), [])

        with patch('kbkit.systems.properties.validate_path', return_value=tmp_path):
            props = SystemProperties(system_path=str(tmp_path))

        with pytest.raises(FileNotFoundError, match="No topology or structure file found"):
            _ = props.topology




class TestSystemPropertiesTopologyProperties:
    """Test the topology_properties property."""

    @patch('kbkit.systems.properties.GroParser')
    @patch('kbkit.systems.properties.SystemProperties.find_files')
    def test_topology_properties_lists_attributes(self, mock_find_files, mock_gro_class, tmp_path):
        """Test that topology_properties lists topology attributes."""
        gro_file = tmp_path / "system.gro"

        mock_find_files.side_effect = lambda **kwargs: {
            '.edr': [tmp_path / "system.edr"],
            '.gro': [gro_file],
            '.top': []
        }.get(kwargs.get('suffix'), [])

        mock_gro = Mock(spec=GroParser)
        mock_gro.molecules = [["MOL1"] * 10]
        mock_gro.total_molecules = 10
        mock_gro.box_volume = 100.0
        mock_gro_class.return_value = mock_gro

        with patch('kbkit.systems.properties.validate_path', return_value=tmp_path):
            props = SystemProperties(system_path=str(tmp_path))
            topology_props = props.topology_properties

        assert isinstance(topology_props, list)
        assert "total_molecules" in topology_props
        assert "box_volume" in topology_props

    @patch('kbkit.systems.properties.GroParser')
    @patch('kbkit.systems.properties.SystemProperties.find_files')
    def test_topology_properties_excludes_private(self, mock_find_files, mock_gro_class, tmp_path):
        """Test that topology_properties excludes private attributes."""
        gro_file = tmp_path / "system.gro"

        mock_find_files.side_effect = lambda **kwargs: {
            '.edr': [tmp_path / "system.edr"],
            '.gro': [gro_file],
            '.top': []
        }.get(kwargs.get('suffix'), [])

        mock_gro = Mock(spec=GroParser)
        mock_gro.molecules = [["MOL1"] * 10]
        mock_gro._private_attr = "private"
        mock_gro_class.return_value = mock_gro

        with patch('kbkit.systems.properties.validate_path', return_value=tmp_path):
            props = SystemProperties(system_path=str(tmp_path))
            topology_props = props.topology_properties

        assert not any(name.startswith("_") for name in topology_props)




class TestSystemPropertiesGet:
    """Test the get method."""

    @patch('kbkit.systems.properties.GroParser')
    @patch('kbkit.systems.properties.SystemProperties.find_files')
    def test_get_topology_property(self, mock_find_files, mock_gro_class, tmp_path):
        """Test getting property from topology."""
        mock_find_files.side_effect = lambda **kwargs: {
            '.edr': [tmp_path / "system.edr"],
            '.gro': [tmp_path / "system.gro"],
            '.top': []
        }.get(kwargs.get('suffix'), [])

        mock_gro = Mock(spec=GroParser)
        mock_gro.molecules = [["MOL1"] * 10]
        mock_gro.total_molecules = 100
        mock_gro_class.return_value = mock_gro

        with patch('kbkit.systems.properties.validate_path', return_value=tmp_path):
            props = SystemProperties(system_path=str(tmp_path))
            result = props.get("total_molecules")

        assert result == 100

    @patch('kbkit.systems.properties.GroParser')
    @patch('kbkit.systems.properties.SystemProperties.find_files')
    def test_get_electron_count_with_alias(self, mock_find_files, mock_gro_class, tmp_path):
        """Test getting electron count with various aliases."""
        mock_find_files.side_effect = lambda **kwargs: {
            '.edr': [tmp_path / "system.edr"],
            '.gro': [tmp_path / "system.gro"],
            '.top': []
        }.get(kwargs.get('suffix'), [])

        mock_gro = Mock(spec=GroParser)
        mock_gro.molecules = [["MOL1"] * 10]
        mock_gro.electron_count = {"MOL1": 10}
        mock_gro_class.return_value = mock_gro

        with patch('kbkit.systems.properties.validate_path', return_value=tmp_path):
            props = SystemProperties(system_path=str(tmp_path))

            for alias in ["electron", "elec", "z_count", "z-count"]:
                result = props.get(alias)
                assert result == {"MOL1": 10}

    @patch('kbkit.systems.properties.EdrParser')
    @patch('kbkit.systems.properties.GroParser')
    @patch('kbkit.systems.properties.SystemProperties.find_files')
    def test_get_units_property(self, mock_find_files, mock_gro_class, mock_edr_class, tmp_path):
        """Test getting units dictionary."""
        mock_find_files.side_effect = lambda **kwargs: {
            '.edr': [tmp_path / "system.edr"],
            '.gro': [tmp_path / "system.gro"],
            '.top': []
        }.get(kwargs.get('suffix'), [])

        mock_gro = Mock(spec=GroParser)
        mock_gro.molecules = [["MOL1"] * 10]
        mock_gro_class.return_value = mock_gro

        mock_edr = Mock(spec=EdrParser)
        mock_edr.units = {"Temperature": "K"}
        mock_edr_class.return_value = mock_edr

        with patch('kbkit.systems.properties.validate_path', return_value=tmp_path):
            props = SystemProperties(system_path=str(tmp_path))
            result = props.get("units")

        assert result == {"Temperature": "K"}

    @patch('kbkit.systems.properties.EdrParser')
    @patch('kbkit.systems.properties.GroParser')
    @patch('kbkit.systems.properties.SystemProperties.find_files')
    def test_get_energy_property_averaged(self, mock_find_files, mock_gro_class, mock_edr_class, tmp_path):
        """Test getting averaged energy property."""
        mock_find_files.side_effect = lambda **kwargs: {
            '.edr': [tmp_path / "system.edr"],
            '.gro': [tmp_path / "system.gro"],
            '.top': []
        }.get(kwargs.get('suffix'), [])

        mock_gro = Mock(spec=GroParser)
        mock_gro.molecules = [["MOL1"] * 10]
        mock_gro.box_volume = 100.0
        mock_gro.total_molecules = 100
        mock_gro_class.return_value = mock_gro

        mock_edr = Mock(spec=EdrParser)
        mock_edr.get.return_value = np.array([298.0, 299.0, 300.0])
        mock_edr_class.return_value = mock_edr

        with patch('kbkit.systems.properties.validate_path', return_value=tmp_path):
            props = SystemProperties(system_path=str(tmp_path))
            result = props.get("Temperature", avg=True)

        assert isinstance(result, float)
        assert result == pytest.approx(299.0)

    @patch('kbkit.systems.properties.EdrParser')
    @patch('kbkit.systems.properties.GroParser')
    @patch('kbkit.systems.properties.SystemProperties.find_files')
    def test_get_energy_property_time_series(self, mock_find_files, mock_gro_class, mock_edr_class, tmp_path):
        """Test getting energy property as time series."""
        mock_find_files.side_effect = lambda **kwargs: {
            '.edr': [tmp_path / "system.edr"],
            '.gro': [tmp_path / "system.gro"],
            '.top': []
        }.get(kwargs.get('suffix'), [])

        mock_gro = Mock(spec=GroParser)
        mock_gro.molecules = [["MOL1"] * 10]
        mock_gro.box_volume = 100.0
        mock_gro.total_molecules = 100
        mock_gro_class.return_value = mock_gro

        mock_edr = Mock(spec=EdrParser)
        mock_edr.get.side_effect = lambda prop, **kwargs: (
            np.array([0, 1, 2]) if prop == "time" else np.array([298.0, 299.0, 300.0])
        )
        mock_edr_class.return_value = mock_edr

        with patch('kbkit.systems.properties.validate_path', return_value=tmp_path):
            props = SystemProperties(system_path=str(tmp_path))
            times, values = props.get("Temperature", avg=False, time_series=True)

        assert isinstance(times, np.ndarray)
        assert isinstance(values, np.ndarray)
        assert len(times) == len(values)

    @patch('kbkit.systems.properties.EdrParser')
    @patch('kbkit.systems.properties.GroParser')
    @patch('kbkit.systems.properties.SystemProperties.find_files')
    def test_get_cp_property(self, mock_find_files, mock_gro_class, mock_edr_class, tmp_path):
        """Test getting heat capacity at constant pressure."""
        mock_find_files.side_effect = lambda **kwargs: {
            '.edr': [tmp_path / "system.edr"],
            '.gro': [tmp_path / "system.gro"],
            '.top': []
        }.get(kwargs.get('suffix'), [])

        mock_gro = Mock(spec=GroParser)
        mock_gro.molecules = [["MOL1"] * 10]
        mock_gro.box_volume = 100.0
        mock_gro.total_molecules = 100
        mock_gro_class.return_value = mock_gro

        mock_edr = Mock(spec=EdrParser)
        mock_edr.cp.return_value = np.array([75.0, 75.5, 76.0])
        mock_edr_class.return_value = mock_edr

        with patch('kbkit.systems.properties.validate_path', return_value=tmp_path):
            props = SystemProperties(system_path=str(tmp_path))
            result = props.get("cp", avg=True)

        assert isinstance(result, float)
        mock_edr.cp.assert_called_once()

    @patch('kbkit.systems.properties.EdrParser')
    @patch('kbkit.systems.properties.GroParser')
    @patch('kbkit.systems.properties.SystemProperties.find_files')
    def test_get_cv_property(self, mock_find_files, mock_gro_class, mock_edr_class, tmp_path):
        """Test getting heat capacity at constant volume."""
        mock_find_files.side_effect = lambda **kwargs: {
            '.edr': [tmp_path / "system.edr"],
            '.gro': [tmp_path / "system.gro"],
            '.top': []
        }.get(kwargs.get('suffix'), [])

        mock_gro = Mock(spec=GroParser)
        mock_gro.molecules = [["MOL1"] * 10]
        mock_gro.box_volume = 100.0
        mock_gro.total_molecules = 100
        mock_gro_class.return_value = mock_gro

        mock_edr = Mock(spec=EdrParser)
        mock_edr.cv.return_value = np.array([50.0, 50.5, 51.0])
        mock_edr_class.return_value = mock_edr

        with patch('kbkit.systems.properties.validate_path', return_value=tmp_path):
            props = SystemProperties(system_path=str(tmp_path))
            result = props.get("cv", avg=True)

        assert isinstance(result, float)
        mock_edr.cv.assert_called_once()

    @patch('kbkit.systems.properties.EdrParser')
    @patch('kbkit.systems.properties.GroParser')
    @patch('kbkit.systems.properties.SystemProperties.find_files')
    def test_get_enthalpy_property(self, mock_find_files, mock_gro_class, mock_edr_class, tmp_path):
        """Test getting molar enthalpy."""
        mock_find_files.side_effect = lambda **kwargs: {
            '.edr': [tmp_path / "system.edr"],
            '.gro': [tmp_path / "system.gro"],
            '.top': []
        }.get(kwargs.get('suffix'), [])

        mock_gro = Mock(spec=GroParser)
        mock_gro.molecules = [["MOL1"] * 10]
        mock_gro.box_volume = 100.0
        mock_gro.total_molecules = 100
        mock_gro_class.return_value = mock_gro

        mock_edr = Mock(spec=EdrParser)
        mock_edr.molar_enthalpy.return_value = np.array([10.0, 10.5, 11.0])
        mock_edr_class.return_value = mock_edr

        with patch('kbkit.systems.properties.validate_path', return_value=tmp_path):
            props = SystemProperties(system_path=str(tmp_path))
            result = props.get("enthalpy", avg=True)

        assert isinstance(result, float)
        mock_edr.molar_enthalpy.assert_called_once()

    @patch('kbkit.systems.properties.EdrParser')
    @patch('kbkit.systems.properties.GroParser')
    @patch('kbkit.systems.properties.SystemProperties.find_files')
    def test_get_isothermal_compressibility(self, mock_find_files, mock_gro_class, mock_edr_class, tmp_path):
        """Test getting isothermal compressibility."""
        mock_find_files.side_effect = lambda **kwargs: {
            '.edr': [tmp_path / "system.edr"],
            '.gro': [tmp_path / "system.gro"],
            '.top': []
        }.get(kwargs.get('suffix'), [])

        mock_gro = Mock(spec=GroParser)
        mock_gro.molecules = [["MOL1"] * 10]
        mock_gro.box_volume = 100.0
        mock_gro.total_molecules = 100
        mock_gro_class.return_value = mock_gro

        mock_edr = Mock(spec=EdrParser)
        mock_edr.isothermal_compressibility.return_value = np.array([1e-5, 1.1e-5, 1.2e-5])
        mock_edr_class.return_value = mock_edr

        with patch('kbkit.systems.properties.validate_path', return_value=tmp_path):
            props = SystemProperties(system_path=str(tmp_path))
            result = props.get("isothermal-compressibility", avg=True)

        assert isinstance(result, float)
        mock_edr.isothermal_compressibility.assert_called_once()

    @patch('kbkit.systems.properties.EdrParser')
    @patch('kbkit.systems.properties.GroParser')
    @patch('kbkit.systems.properties.SystemProperties.find_files')
    def test_get_molar_volume(self, mock_find_files, mock_gro_class, mock_edr_class, tmp_path):
        """Test getting molar volume."""
        mock_find_files.side_effect = lambda **kwargs: {
            '.edr': [tmp_path / "system.edr"],
            '.gro': [tmp_path / "system.gro"],
            '.top': []
        }.get(kwargs.get('suffix'), [])

        mock_gro = Mock(spec=GroParser)
        mock_gro.molecules = [["MOL1"] * 10]
        mock_gro.box_volume = 100.0
        mock_gro.total_molecules = 100
        mock_gro_class.return_value = mock_gro

        mock_edr = Mock(spec=EdrParser)
        mock_edr.units = {"molar-volume": "cm^3/mol"}
        mock_edr.molar_volume.return_value = np.array([18.0, 18.1, 18.2])
        mock_edr_class.return_value = mock_edr

        with patch('kbkit.systems.properties.validate_path', return_value=tmp_path):
            props = SystemProperties(system_path=str(tmp_path))
            result = props.get("molar-volume", avg=True)

        assert isinstance(result, float)
        mock_edr.molar_volume.assert_called_once()

    @patch('kbkit.systems.properties.EdrParser')
    @patch('kbkit.systems.properties.GroParser')
    @patch('kbkit.systems.properties.SystemProperties.find_files')
    def test_get_number_density(self, mock_find_files, mock_gro_class, mock_edr_class, tmp_path):
        """Test getting number density (inverse of molar volume)."""
        mock_find_files.side_effect = lambda **kwargs: {
            '.edr': [tmp_path / "system.edr"],
            '.gro': [tmp_path / "system.gro"],
            '.top': []
        }.get(kwargs.get('suffix'), [])

        mock_gro = Mock(spec=GroParser)
        mock_gro.molecules = [["MOL1"] * 10]
        mock_gro.box_volume = 100.0
        mock_gro.total_molecules = 100
        mock_gro_class.return_value = mock_gro

        mock_edr = Mock(spec=EdrParser)
        mock_edr.units = {"molar-volume": "cm^3/mol"}
        mock_edr.molar_volume.return_value = np.array([18.0, 18.0, 18.0])
        mock_edr_class.return_value = mock_edr

        with patch('kbkit.systems.properties.validate_path', return_value=tmp_path):
            props = SystemProperties(system_path=str(tmp_path))
            result = props.get("number-density", avg=True)

        assert isinstance(result, float)
        assert result == pytest.approx(1.0 / 18.0)

    @patch('kbkit.systems.properties.EdrParser')
    @patch('kbkit.systems.properties.GroParser')
    @patch('kbkit.systems.properties.SystemProperties.find_files')
    def test_get_with_unit_conversion(self, mock_find_files, mock_gro_class, mock_edr_class, tmp_path):
        """Test getting property with unit conversion."""
        mock_find_files.side_effect = lambda **kwargs: {
            '.edr': [tmp_path / "system.edr"],
            '.gro': [tmp_path / "system.gro"],
            '.top': []
        }.get(kwargs.get('suffix'), [])

        mock_gro = Mock(spec=GroParser)
        mock_gro.molecules = [["MOL1"] * 10]
        mock_gro.box_volume = 100.0
        mock_gro.total_molecules = 100
        mock_gro_class.return_value = mock_gro

        mock_edr = Mock(spec=EdrParser)
        mock_edr.get.return_value = np.array([298.0, 299.0, 300.0])
        mock_edr_class.return_value = mock_edr

        with patch('kbkit.systems.properties.validate_path', return_value=tmp_path):
            props = SystemProperties(system_path=str(tmp_path))
            result = props.get("Temperature", units="K", avg=True)

        mock_edr.get.assert_called()
        call_kwargs = mock_edr.get.call_args[1]
        assert call_kwargs['units'] == "K"

    @patch('kbkit.systems.properties.EdrParser')
    @patch('kbkit.systems.properties.GroParser')
    @patch('kbkit.systems.properties.SystemProperties.find_files')
    def test_get_removes_duplicate_times(self, mock_find_files, mock_gro_class, mock_edr_class, tmp_path):
        """Test that get removes duplicate time points."""
        mock_find_files.side_effect = lambda **kwargs: {
            '.edr': [tmp_path / "system.edr"],
            '.gro': [tmp_path / "system.gro"],
            '.top': []
        }.get(kwargs.get('suffix'), [])

        mock_gro = Mock(spec=GroParser)
        mock_gro.molecules = [["MOL1"] * 10]
        mock_gro.box_volume = 100.0
        mock_gro.total_molecules = 100
        mock_gro_class.return_value = mock_gro

        mock_edr = Mock(spec=EdrParser)
        # Return duplicate times
        mock_edr.get.side_effect = lambda prop, **kwargs: (
            np.array([0, 1, 1, 2]) if prop == "time" else np.array([298.0, 299.0, 299.5, 300.0])
        )
        mock_edr_class.return_value = mock_edr

        with patch('kbkit.systems.properties.validate_path', return_value=tmp_path):
            props = SystemProperties(system_path=str(tmp_path))
            values = props.get("Temperature", avg=False, time_series=False)

        # Should have removed duplicate at time=1
        assert len(values) == 3




class TestSystemPropertiesTimeseriesPlotter:
    """Test the timeseries_plotter method."""

    @patch('kbkit.systems.properties.TimeseriesPlotter')
    @patch('kbkit.systems.properties.GroParser')
    @patch('kbkit.systems.properties.SystemProperties.find_files')
    def test_timeseries_plotter_creates_plotter(self, mock_find_files, mock_gro_class,
                                                mock_plotter_class, tmp_path):
        """Test that timeseries_plotter creates TimeseriesPlotter."""
        mock_find_files.side_effect = lambda **kwargs: {
            '.edr': [tmp_path / "system.edr"],
            '.gro': [tmp_path / "system.gro"],
            '.top': []
        }.get(kwargs.get('suffix'), [])

        mock_gro = Mock(spec=GroParser)
        mock_gro.molecules = [["MOL1"] * 10]
        mock_gro_class.return_value = mock_gro

        mock_plotter = Mock(spec=TimeseriesPlotter)
        mock_plotter_class.return_value = mock_plotter

        with patch('kbkit.systems.properties.validate_path', return_value=tmp_path):
            props = SystemProperties(system_path=str(tmp_path))
            plotter = props.timeseries_plotter(start_time=5000)

        mock_plotter_class.assert_called_once_with(props, start_time=5000)
        assert plotter == mock_plotter

    @patch('kbkit.systems.properties.TimeseriesPlotter')
    @patch('kbkit.systems.properties.GroParser')
    @patch('kbkit.systems.properties.SystemProperties.find_files')
    def test_timeseries_plotter_default_start_time(self, mock_find_files, mock_gro_class,
                                                mock_plotter_class, tmp_path):
        """Test timeseries_plotter with default start_time."""
        mock_find_files.side_effect = lambda **kwargs: {
            '.edr': [tmp_path / "system.edr"],
            '.gro': [tmp_path / "system.gro"],
            '.top': []
        }.get(kwargs.get('suffix'), [])

        mock_gro = Mock(spec=GroParser)
        mock_gro.molecules = [["MOL1"] * 10]
        mock_gro_class.return_value = mock_gro

        mock_plotter = Mock(spec=TimeseriesPlotter)
        mock_plotter_class.return_value = mock_plotter

        with patch('kbkit.systems.properties.validate_path', return_value=tmp_path):
            props = SystemProperties(system_path=str(tmp_path))
            props.timeseries_plotter()

        mock_plotter_class.assert_called_once_with(props, start_time=0)




class TestSystemPropertiesIntegration:
    """Integration tests for SystemProperties."""

    @patch('kbkit.systems.properties.EdrParser')
    @patch('kbkit.systems.properties.GroParser')
    @patch('kbkit.systems.properties.SystemProperties.find_files')
    def test_full_workflow(self, mock_find_files, mock_gro_class, mock_edr_class, tmp_path):
        """Test complete workflow from initialization to property access."""
        mock_find_files.side_effect = lambda **kwargs: {
            '.edr': [tmp_path / "system.edr"],
            '.gro': [tmp_path / "system.gro"],
            '.top': []
        }.get(kwargs.get('suffix'), [])

        # Setup mocks
        mock_gro = Mock(spec=GroParser)
        mock_gro.molecules = [["MOL1"] * 50, ["MOL2"] * 50]
        mock_gro.total_molecules = 100
        mock_gro.box_volume = 125.0
        mock_gro.molecule_count = {"MOL1": 50, "MOL2": 50}
        mock_gro_class.return_value = mock_gro

        mock_edr = Mock(spec=EdrParser)
        mock_edr.units = {"Temperature": "K", "Pressure": "bar"}
        mock_edr.get.return_value = np.array([298.0, 299.0, 300.0])
        mock_edr_class.return_value = mock_edr

        with patch('kbkit.systems.properties.validate_path', return_value=tmp_path):
            # Create SystemProperties
            props = SystemProperties(system_path=str(tmp_path), start_time=1000)

            # Access topology properties
            assert props.topology.total_molecules == 100
            assert props.topology.box_volume == 125.0

            # Access energy properties
            temp = props.get("Temperature", avg=True)
            assert isinstance(temp, float)

            # Get time series
            mock_edr.get.side_effect = lambda prop, **kwargs: (
                np.array([0, 1, 2]) if prop == "time" else np.array([298.0, 299.0, 300.0])
            )
            times, values = props.get("Temperature", avg=False, time_series=True)
            assert len(times) == len(values)

    @patch('kbkit.systems.properties.EdrParser')
    @patch('kbkit.systems.properties.GroParser')
    @patch('kbkit.systems.properties.SystemProperties.find_files')
    def test_multiple_edr_files(self, mock_find_files, mock_gro_class, mock_edr_class, tmp_path):
        """Test handling multiple EDR files."""
        edr_file1 = tmp_path / "system1.edr"
        edr_file2 = tmp_path / "system2.edr"

        mock_find_files.side_effect = lambda **kwargs: {
            '.edr': [edr_file1, edr_file2],
            '.gro': [tmp_path / "system.gro"],
            '.top': []
        }.get(kwargs.get('suffix'), [])

        mock_gro = Mock(spec=GroParser)
        mock_gro.molecules = [["MOL1"] * 10]
        mock_gro.box_volume = 100.0
        mock_gro.total_molecules = 100
        mock_gro_class.return_value = mock_gro

        mock_edr1 = Mock(spec=EdrParser)
        mock_edr1.get.return_value = np.array([298.0, 299.0])

        mock_edr2 = Mock(spec=EdrParser)
        mock_edr2.get.return_value = np.array([300.0, 301.0])

        mock_edr_class.side_effect = [mock_edr1, mock_edr2]

        with patch('kbkit.systems.properties.validate_path', return_value=tmp_path):
            props = SystemProperties(system_path=str(tmp_path))
            result = props.get("Temperature", avg=True)

        # Should average across both files
        assert isinstance(result, float)


