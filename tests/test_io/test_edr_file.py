"""Unit tests for EdrParser class."""
import warnings

# Suppress NumPy/SciPy compatibility warning (harmless with NumPy 2.x + SciPy 1.16+)
warnings.filterwarnings('ignore', message='numpy.ndarray size changed', category=RuntimeWarning)

import subprocess
from pathlib import Path
from unittest.mock import Mock, PropertyMock, mock_open, patch

import numpy as np
import pytest

from kbkit.io.edr import EdrParser


@pytest.fixture
def mock_edr_file(tmp_path):
    """Create a mock .edr file path."""
    edr_file = tmp_path / "test.edr"
    edr_file.touch()
    return str(edr_file)


@pytest.fixture
def mock_ureg():
    """Create a mock unit registry."""
    with patch('kbkit.io.edr.load_unit_registry') as mock_load:
        mock_registry = Mock()

        # Setup Quantity mock to handle unit conversions
        def quantity_side_effect(value, unit):
            q = Mock()
            if isinstance(value, np.ndarray):
                q.magnitude = value
            elif isinstance(value, (list, tuple)):
                q.magnitude = np.array(value)
            else:
                q.magnitude = value

            # Mock the .to() method to return self with same magnitude
            def to_mock(target_unit):
                result = Mock()
                result.magnitude = q.magnitude
                return result

            q.to = Mock(side_effect=to_mock)
            return q

        mock_registry.Quantity = Mock(side_effect=quantity_side_effect)

        # Mock constants R and N_A
        def registry_call(name):
            mock_const = Mock()
            if name == "R":
                mock_const.to.return_value = Mock(magnitude=8.314e-3)
            elif name == "N_A":
                mock_const.to.return_value = Mock(magnitude=6.022e23)
            else:
                mock_const.to.return_value = Mock(magnitude=1.0)
            return mock_const

        mock_registry.side_effect = registry_call
        mock_load.return_value = mock_registry

        yield mock_registry


@pytest.fixture
def sample_xvg_content():
    """Sample XVG file content."""
    return """# This file was created by gmx energy
# Created by:
# GROMACS:      gmx energy, version 2021.1
@    title "GROMACS Energies"
@    xaxis  label "Time (ps)"
@    yaxis  label "Energy (kJ/mol)"
@TYPE xy
@ s0 legend "Potential"
@ s1 legend "Kinetic-En"
@ s2 legend "Total-Energy"
0.0000    -1000.0    500.0    -500.0
1.0000    -1010.0    510.0    -500.0
2.0000    -1020.0    520.0    -500.0
3.0000    -1030.0    530.0    -500.0
4.0000    -1040.0    540.0    -500.0
"""


@pytest.fixture
def sample_gmx_menu_output():
    """Sample GMX energy menu output."""
    return """
Select the terms you want from the following list by
selecting either (part of) the name or the number or a combination.
End your selection with an empty line or a zero.
-------------------------------------------------------------------
  1  Bond             2  Angle            3  Proper-Dih.      4  Ryckaert-Bell.
  5  LJ-14            6  Coulomb-14       7  LJ-(SR)          8  Disper.-corr.
  9  Coulomb-(SR)    10  Coul.-recip.    11  Potential       12  Kinetic-En.
 13  Total-Energy    14  Temperature     15  Pressure        16  Constr.-rmsd
 17  Box-X           18  Box-Y           19  Box-Z           20  Volume
 21  Density         22  pV              23  Enthalpy        24  Vir-XX
"""


@pytest.fixture
def mock_data():
    """Create mock data dictionary."""
    return {
        "time": np.array([0, 1, 2, 3, 4]),
        "potential": np.array([-1000, -1010, -1020, -1030, -1040]),
        "temperature": np.array([298, 298.1, 298.2, 298.3, 298.4]),
        "pressure": np.array([100, 101, 102, 103, 104]),
        "volume": np.array([10, 10.1, 10.2, 10.3, 10.4]),
        "kinetic-en": np.array([500, 510, 520, 530, 540]),
        "total-energy": np.array([-500, -500, -500, -500, -500])
    }


class TestEdrParserInitialization:
    """Test EdrParser initialization."""

    def test_valid_edr_file(self, mock_edr_file, mock_ureg):
        """Test initialization with valid .edr file."""
        parser = EdrParser(mock_edr_file)

        assert parser.edr_path == Path(mock_edr_file)
        assert parser.ureg is not None
        assert parser.Q_ is not None

    def test_invalid_file_extension(self, tmp_path, mock_ureg):
        """Test that non-.edr files raise an error."""
        txt_file = tmp_path / "test.txt"
        txt_file.touch()

        with pytest.raises(ValueError, match=".*\\.edr"):
            EdrParser(str(txt_file))

    def test_nonexistent_file(self, mock_ureg):
        """Test that nonexistent files raise an error."""
        with pytest.raises((FileNotFoundError, ValueError)):
            EdrParser("/nonexistent/path/file.edr")

    def test_unit_registry_loaded(self, mock_edr_file, mock_ureg):
        """Test that unit registry is properly loaded."""
        parser = EdrParser(mock_edr_file)

        assert parser.ureg == mock_ureg
        assert callable(parser.Q_)


class TestClassVariables:
    """Test class-level constants."""

    def test_gromacs_units_defined(self):
        """Test that GROMACS_UNITS is properly defined."""
        assert isinstance(EdrParser.GROMACS_UNITS, dict)
        assert "temperature" in EdrParser.GROMACS_UNITS
        assert EdrParser.GROMACS_UNITS["temperature"] == "K"
        assert EdrParser.GROMACS_UNITS["pressure"] == "bar"

    def test_default_units_defined(self):
        """Test that DEFAULT_UNITS is properly defined."""
        assert isinstance(EdrParser.DEFAULT_UNITS, dict)
        assert "temperature" in EdrParser.DEFAULT_UNITS
        assert "cp" in EdrParser.DEFAULT_UNITS
        assert EdrParser.DEFAULT_UNITS["pressure"] == "kPa"

    def test_fluct_props_defined(self):
        """Test that FLUCT_PROPS is properly defined."""
        assert isinstance(EdrParser.FLUCT_PROPS, tuple)
        assert "cp" in EdrParser.FLUCT_PROPS
        assert "cv" in EdrParser.FLUCT_PROPS
        assert "isothermal-compressibility" in EdrParser.FLUCT_PROPS


class TestGetPropertyNames:
    """Test _get_property_names method."""

    @patch('subprocess.run')
    def test_parse_property_names(self, mock_run, mock_edr_file, mock_ureg, sample_gmx_menu_output):
        """Test parsing property names from GMX output."""
        mock_run.return_value = Mock(
            stderr=sample_gmx_menu_output,
            returncode=0
        )

        parser = EdrParser(mock_edr_file)
        names = parser._get_property_names()

        assert isinstance(names, list)
        assert "bond" in names
        assert "potential" in names
        assert "temperature" in names
        assert "pressure" in names

    @patch('subprocess.run')
    def test_property_names_lowercase(self, mock_run, mock_edr_file, mock_ureg, sample_gmx_menu_output):
        """Test that property names are converted to lowercase."""
        mock_run.return_value = Mock(
            stderr=sample_gmx_menu_output,
            returncode=0
        )

        parser = EdrParser(mock_edr_file)
        names = parser._get_property_names()

        for name in names:
            assert name == name.lower()

    @patch('subprocess.run')
    def test_property_names_no_periods(self, mock_run, mock_edr_file, mock_ureg, sample_gmx_menu_output):
        """Test that periods are removed from property names."""
        mock_run.return_value = Mock(
            stderr=sample_gmx_menu_output,
            returncode=0
        )

        parser = EdrParser(mock_edr_file)
        names = parser._get_property_names()

        for name in names:
            assert "." not in name

    @patch('subprocess.run')
    def test_empty_gmx_output(self, mock_run, mock_edr_file, mock_ureg):
        """Test handling of empty GMX output."""
        mock_run.return_value = Mock(
            stderr="",
            returncode=0
        )

        parser = EdrParser(mock_edr_file)
        names = parser._get_property_names()

        assert isinstance(names, list)
        assert len(names) == 0


class TestGetGmxProperty:
    """Test get_gmx_property method."""

    @patch('subprocess.run')
    @patch('numpy.loadtxt')
    @patch('pathlib.Path.unlink')
    def test_get_property_returns_tuple(self, mock_unlink, mock_loadtxt, mock_run, mock_edr_file, mock_ureg):
        """Test that get_gmx_property returns tuple of time and values."""
        mock_run.return_value = Mock(returncode=0)
        time = np.array([0, 1, 2, 3, 4])
        values = np.array([100, 101, 102, 103, 104])
        mock_loadtxt.return_value = (time, values)

        parser = EdrParser(mock_edr_file)
        result = parser.get_gmx_property("temperature")

        assert isinstance(result, tuple)
        assert len(result) == 2
        np.testing.assert_array_equal(result[0], time)
        np.testing.assert_array_equal(result[1], values)

    @patch('subprocess.run')
    @patch('numpy.loadtxt')
    @patch('pathlib.Path.unlink')
    def test_get_property_with_avg(self, mock_unlink, mock_loadtxt, mock_run, mock_edr_file, mock_ureg):
        """Test get_gmx_property with avg=True returns float."""
        mock_run.return_value = Mock(returncode=0)
        values = np.array([100, 101, 102, 103, 104])
        mock_loadtxt.return_value = (np.array([0, 1, 2, 3, 4]), values)

        parser = EdrParser(mock_edr_file)
        result = parser.get_gmx_property("temperature", avg=True)

        assert isinstance(result, (float, np.floating))
        assert result == values.mean()

    @patch('subprocess.run')
    @patch('numpy.loadtxt')
    @patch('pathlib.Path.unlink')
    def test_get_property_with_kwargs(self, mock_unlink, mock_loadtxt, mock_run, mock_edr_file, mock_ureg):
        """Test get_gmx_property with additional kwargs."""
        mock_run.return_value = Mock(returncode=0)
        mock_loadtxt.return_value = (np.array([0, 1]), np.array([100, 101]))

        parser = EdrParser(mock_edr_file)
        parser.get_gmx_property("temperature", b=1000, e=5000)

        # Check that kwargs were passed to subprocess
        call_args = mock_run.call_args[0][0]
        assert "-b" in call_args or "b" in str(call_args)

    @patch('subprocess.run')
    @patch('pathlib.Path.unlink')
    def test_get_property_cleanup(self, mock_unlink, mock_run, mock_edr_file, mock_ureg):
        """Test that temporary files are cleaned up."""
        mock_run.return_value = Mock(returncode=0)

        with patch('numpy.loadtxt', return_value=(np.array([0]), np.array([100]))):
            parser = EdrParser(mock_edr_file)
            parser.get_gmx_property("temperature")

        # Verify cleanup was called
        assert mock_unlink.called


class TestDataProperty:
    """Test data property."""

    @patch('subprocess.run')
    @patch('builtins.open', new_callable=mock_open)
    @patch('numpy.loadtxt')
    @patch('pathlib.Path.unlink')
    def test_data_property_cached(self, mock_unlink, mock_loadtxt, mock_file, mock_run, mock_edr_file, mock_ureg, sample_gmx_menu_output):
        """Test that data property is cached."""
        # Setup mocks
        mock_run.side_effect = [
            Mock(stderr=sample_gmx_menu_output, returncode=0),  # _get_property_names
            Mock(returncode=0)  # data extraction
        ]

        mock_file.return_value.__iter__.return_value = ['@ s0 legend "Potential"\n']

        time = np.array([0, 1, 2])
        potential = np.array([-1000, -1010, -1020])
        mock_loadtxt.return_value = np.array([time, potential])

        parser = EdrParser(mock_edr_file)

        # Access data twice
        data1 = parser.data
        data2 = parser.data

        # Should be the same object (cached)
        assert data1 is data2

    @patch('subprocess.run')
    @patch('builtins.open', new_callable=mock_open)
    @patch('numpy.loadtxt')
    @patch('pathlib.Path.unlink')
    def test_data_contains_time(self, mock_unlink, mock_loadtxt, mock_file, mock_run, mock_edr_file, mock_ureg, sample_gmx_menu_output):
        """Test that data contains time array."""
        mock_run.side_effect = [
            Mock(stderr=sample_gmx_menu_output, returncode=0),
            Mock(returncode=0)
        ]

        mock_file.return_value.__iter__.return_value = ['@ s0 legend "Potential"\n']

        time = np.array([0, 1, 2])
        potential = np.array([-1000, -1010, -1020])
        mock_loadtxt.return_value = np.array([time, potential])

        parser = EdrParser(mock_edr_file)
        data = parser.data

        assert "time" in data
        assert isinstance(data["time"], np.ndarray)


class TestAvailableProperties:
    """Test available_properties method."""

    def test_available_properties_returns_list(self, mock_edr_file, mock_ureg, mock_data):
        """Test that available_properties returns a list."""
        parser = EdrParser(mock_edr_file)

        # Mock the data property
        with patch.object(type(parser), 'data', new_callable=PropertyMock) as mock_data_prop:
            mock_data_prop.return_value = mock_data
            props = parser.available_properties()

            assert isinstance(props, list)
            assert "time" in props
            assert "potential" in props


class TestUnitsProperty:
    """Test units property."""

    def test_units_property(self, mock_edr_file, mock_ureg, mock_data):
        """Test that units property returns correct mapping."""
        parser = EdrParser(mock_edr_file)

        with patch.object(type(parser), 'data', new_callable=PropertyMock) as mock_data_prop:
            mock_data_prop.return_value = mock_data
            units = parser.units

            assert isinstance(units, dict)
            # Should only include properties that are available
            for key in units.keys():
                assert key in parser.available_properties()


class TestGetMethod:
    """Test get method."""

    def test_get_returns_array(self, mock_edr_file, mock_ureg, mock_data):
        """Test that get returns numpy array."""
        parser = EdrParser(mock_edr_file)

        with patch.object(type(parser), 'data', new_callable=PropertyMock) as mock_data_prop:
            mock_data_prop.return_value = mock_data
            result = parser.get("potential")

            assert isinstance(result, np.ndarray)

    def test_get_with_start_time(self, mock_edr_file, mock_ureg, mock_data):
        """Test get method with start_time filter."""
        parser = EdrParser(mock_edr_file)

        with patch.object(type(parser), 'data', new_callable=PropertyMock) as mock_data_prop:
            mock_data_prop.return_value = mock_data
            result = parser.get("potential", start_time=2)

            # Should filter out times <= 2
            assert len(result) < len(mock_data["potential"])
            assert len(result) == 2  # times 3 and 4

    def test_get_invalid_property(self, mock_edr_file, mock_ureg, mock_data):
        """Test get method with invalid property name."""
        parser = EdrParser(mock_edr_file)

        with patch.object(type(parser), 'data', new_callable=PropertyMock) as mock_data_prop:
            mock_data_prop.return_value = mock_data

            with pytest.raises(KeyError, match="not available"):
                parser.get("nonexistent_property")

    @patch('kbkit.io.edr.resolve_attr_key')
    def test_get_with_alias(self, mock_resolve, mock_edr_file, mock_ureg, mock_data):
        """Test get method with property alias."""
        mock_resolve.return_value = "potential"
        parser = EdrParser(mock_edr_file)

        with patch.object(type(parser), 'data', new_callable=PropertyMock) as mock_data_prop:
            mock_data_prop.return_value = mock_data
            result = parser.get("U")  # Alias for potential

            assert isinstance(result, np.ndarray)
            mock_resolve.assert_called_once()


class TestMolarVolume:
    """Test molar_volume method."""

    def test_molar_volume_with_volume_data(self, mock_edr_file, mock_ureg, mock_data):
        """Test molar_volume when volume is in edr file."""
        parser = EdrParser(mock_edr_file)

        with patch.object(type(parser), 'data', new_callable=PropertyMock) as mock_data_prop:
            mock_data_prop.return_value = mock_data
            result = parser.molar_volume(nmol=100)

            assert isinstance(result, np.ndarray)
            # With start_time=0 (default), time > 0 filters out first point
            # So we get 4 points instead of 5
            assert len(result) == 4

    def test_molar_volume_fallback(self, mock_edr_file, mock_ureg, capsys):
        """Test molar_volume fallback when volume not in edr."""
        parser = EdrParser(mock_edr_file)

        # Mock data without volume
        data_no_volume = {
            "time": np.array([0, 1, 2, 3, 4]),
            "potential": np.array([-1000, -1010, -1020, -1030, -1040]),
            "temperature": np.array([298, 298.1, 298.2, 298.3, 298.4])
        }

        with patch.object(type(parser), 'data', new_callable=PropertyMock) as mock_data_prop:
            mock_data_prop.return_value = data_no_volume
            result = parser.molar_volume(nmol=100, volume=10.0)

            # Check warning was printed
            captured = capsys.readouterr()
            assert "Warning" in captured.out or "Falling back" in captured.out

            # Result should be scalar divided by nmol
            assert isinstance(result, (float, np.floating, np.ndarray))


class TestConfigurationalEnthalpy:
    """Test configurational_enthalpy method."""

    def test_configurational_enthalpy_calculation(self, mock_edr_file, mock_ureg, mock_data):
        """Test configurational enthalpy calculation."""
        parser = EdrParser(mock_edr_file)

        with patch.object(type(parser), 'data', new_callable=PropertyMock) as mock_data_prop:
            mock_data_prop.return_value = mock_data
            result = parser.configurational_enthalpy()

            assert isinstance(result, np.ndarray)
            # With start_time=0 (default), time > 0 filters out first point
            assert len(result) == 4

    def test_configurational_enthalpy_fallback(self, mock_edr_file, mock_ureg, capsys):
        """Test configurational enthalpy with volume fallback."""
        parser = EdrParser(mock_edr_file)

        data_no_volume = {
            "time": np.array([0, 1, 2, 3, 4]),
            "potential": np.array([-1000, -1010, -1020, -1030, -1040]),
            "pressure": np.array([100, 101, 102, 103, 104]),
            "temperature": np.array([298, 298.1, 298.2, 298.3, 298.4])
        }

        with patch.object(type(parser), 'data', new_callable=PropertyMock) as mock_data_prop:
            mock_data_prop.return_value = data_no_volume
            result = parser.configurational_enthalpy(volume=10.0)

            captured = capsys.readouterr()
            assert "Warning" in captured.out or "Falling back" in captured.out

class TestMolarEnthalpy:
    """Test molar_enthalpy method."""

    def test_molar_enthalpy(self, mock_edr_file, mock_ureg, mock_data):
        """Test molar enthalpy calculation."""
        parser = EdrParser(mock_edr_file)

        with patch.object(type(parser), 'data', new_callable=PropertyMock) as mock_data_prop:
            mock_data_prop.return_value = mock_data
            result = parser.molar_enthalpy(nmol=100)

            assert isinstance(result, np.ndarray)


class TestHeatCapacity:
    """Test cp and cv methods."""

    def test_cp_calculation(self, mock_edr_file, mock_ureg, mock_data):
        """Test constant pressure heat capacity calculation."""
        parser = EdrParser(mock_edr_file)

        with patch.object(type(parser), 'data', new_callable=PropertyMock) as mock_data_prop:
            mock_data_prop.return_value = mock_data
            result = parser.cp(nmol=100)

            assert isinstance(result, (float, np.floating))
            assert result >= 0  # Heat capacity should be positive

    def test_cv_calculation(self, mock_edr_file, mock_ureg, mock_data):
        """Test constant volume heat capacity calculation."""
        parser = EdrParser(mock_edr_file)

        with patch.object(type(parser), 'data', new_callable=PropertyMock) as mock_data_prop:
            mock_data_prop.return_value = mock_data
            result = parser.cv(nmol=100)

            assert isinstance(result, (float, np.floating))
            assert result >= 0

    def test_cp_with_start_time(self, mock_edr_file, mock_ureg, mock_data):
        """Test cp with start_time parameter."""
        parser = EdrParser(mock_edr_file)

        with patch.object(type(parser), 'data', new_callable=PropertyMock) as mock_data_prop:
            mock_data_prop.return_value = mock_data
            result = parser.cp(nmol=100, start_time=2)

            assert isinstance(result, (float, np.floating))


class TestIsothermalCompressibility:
    """Test isothermal_compressibility method."""

    def test_isothermal_compressibility_calculation(self, mock_edr_file, mock_ureg, mock_data):
        """Test isothermal compressibility calculation."""
        parser = EdrParser(mock_edr_file)

        with patch.object(type(parser), 'data', new_callable=PropertyMock) as mock_data_prop:
            mock_data_prop.return_value = mock_data
            result = parser.isothermal_compressibility()

            assert isinstance(result, (float, np.floating))
            assert result >= 0  # Compressibility should be positive

    def test_isothermal_compressibility_no_volume(self, mock_edr_file, mock_ureg):
        """Test isothermal compressibility raises error without volume."""
        parser = EdrParser(mock_edr_file)

        data_no_volume = {
            "time": np.array([0, 1, 2]),
            "potential": np.array([-1000, -1010, -1020]),
            "temperature": np.array([298, 298.1, 298.2])
        }

        with patch.object(type(parser), 'data', new_callable=PropertyMock) as mock_data_prop:
            mock_data_prop.return_value = data_no_volume

            with pytest.raises(KeyError, match="constant volume"):
                parser.isothermal_compressibility()


class TestEdgeCases:
    """Test edge cases and error conditions."""

    @patch('subprocess.run')
    def test_subprocess_error_handling(self, mock_run, mock_edr_file, mock_ureg):
        """Test handling of subprocess errors."""
        mock_run.side_effect = subprocess.CalledProcessError(1, "gmx")

        parser = EdrParser(mock_edr_file)

        with pytest.raises(subprocess.CalledProcessError):
            parser.get_gmx_property("temperature")

    @patch('subprocess.run')
    @patch('builtins.open', new_callable=mock_open)
    @patch('numpy.loadtxt')
    @patch('pathlib.Path.unlink')
    def test_empty_data(self, mock_unlink, mock_loadtxt, mock_file, mock_run, mock_edr_file, mock_ureg):
        """Test handling of empty data."""
        mock_run.side_effect = [
            Mock(stderr="", returncode=0),
            Mock(returncode=0)
        ]

        mock_file.return_value.__iter__.return_value = []
        mock_loadtxt.return_value = np.array([[], []])

        parser = EdrParser(mock_edr_file)
        data = parser.data

        assert isinstance(data, dict)

    def test_get_with_custom_units(self, mock_edr_file, mock_ureg, mock_data):
        """Test get method with custom units."""
        parser = EdrParser(mock_edr_file)

        with patch.object(type(parser), 'data', new_callable=PropertyMock) as mock_data_prop:
            mock_data_prop.return_value = mock_data
            result = parser.get("temperature", units="K")

            assert isinstance(result, np.ndarray)


class TestIntegration:
    """Integration tests for EdrParser."""

    def test_complete_workflow(self, mock_edr_file, mock_ureg, mock_data):
        """Test complete workflow with multiple property accesses."""
        parser = EdrParser(mock_edr_file)

        with patch.object(type(parser), 'data', new_callable=PropertyMock) as mock_data_prop:
            mock_data_prop.return_value = mock_data

            # Test multiple operations
            props = parser.available_properties()
            assert isinstance(props, list)

            units = parser.units
            assert isinstance(units, dict)

            # These should not raise errors
            _ = parser.get("potential")
            _ = parser.molar_volume(nmol=100)
            _ = parser.cv(nmol=100)
            _ = parser.cp(nmol=100)

    def test_property_consistency(self, mock_edr_file, mock_ureg, mock_data):
        """Test that properties are consistent."""
        parser = EdrParser(mock_edr_file)

        with patch.object(type(parser), 'data', new_callable=PropertyMock) as mock_data_prop:
            mock_data_prop.return_value = mock_data

            # Available properties should match data keys
            props = parser.available_properties()
            for prop in props:
                assert prop in mock_data

            # Units should only include available properties
            units = parser.units
            for key in units.keys():
                assert key in props
