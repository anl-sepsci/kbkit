"""Unit tests for PropertyResult container."""

from unittest.mock import Mock, patch

import numpy as np
import pytest

from kbkit.schema.kbi_metadata import KBIMetadata
from kbkit.schema.property_result import PropertyResult


class TestPropertyResultInitialization:
    """Test PropertyResult initialization."""

    def test_basic_initialization(self):
        """Test basic initialization with required parameters."""
        name = "density"
        value = np.array([1.0, 2.0, 3.0])

        result = PropertyResult(name=name, value=value)

        assert result.name == name
        assert np.array_equal(result.value, value)
        assert result.metadata == {}
        assert result.property_type is None
        assert result.units is None

    def test_initialization_with_all_parameters(self):
        """Test initialization with all parameters."""
        name = "kbi"
        value = np.array([[1.0, 2.0], [3.0, 4.0]])
        metadata = {"system": "water-ethanol"}
        property_type = "thermodynamic"
        units = "cm^3/mol"

        result = PropertyResult(
            name=name,
            value=value,
            metadata=metadata,
            property_type=property_type,
            units=units
        )

        assert result.name == name
        assert np.array_equal(result.value, value)
        assert result.metadata == metadata
        assert result.property_type == property_type
        assert result.units == units

    def test_initialization_with_list_value(self):
        """Test that list values are converted to numpy arrays."""
        name = "temperature"
        value = [298.15, 300.0, 310.0]

        result = PropertyResult(name=name, value=value)

        assert isinstance(result.value, np.ndarray)
        assert np.array_equal(result.value, np.array(value))

    def test_initialization_with_scalar_value(self):
        """Test initialization with scalar value."""
        name = "pressure"
        value = 101325.0

        result = PropertyResult(name=name, value=value)

        assert isinstance(result.value, np.ndarray)
        assert result.value == value

    def test_initialization_metadata_none_becomes_empty_dict(self):
        """Test that None metadata becomes empty dict."""
        result = PropertyResult(name="test", value=np.array([1.0]))

        assert result.metadata == {}
        assert isinstance(result.metadata, dict)

    def test_initialization_with_empty_metadata(self):
        """Test initialization with explicitly empty metadata."""
        result = PropertyResult(
            name="test",
            value=np.array([1.0]),
            metadata={}
        )

        assert result.metadata == {}

    def test_initialization_with_nested_metadata(self):
        """Test initialization with nested metadata structure."""
        metadata = {
            "system_1": {
                "pair_1": {"data": [1, 2, 3]},
                "pair_2": {"data": [4, 5, 6]}
            }
        }

        result = PropertyResult(
            name="test",
            value=np.array([1.0]),
            metadata=metadata
        )

        assert result.metadata == metadata


class TestPropertyResultToMethod:
    """Test PropertyResult.to() unit conversion method."""

    @patch('kbkit.schema.property_result.load_unit_registry')
    def test_to_same_units_returns_self(self, mock_load_ureg):
        """Test that converting to same units returns self."""
        result = PropertyResult(
            name="density",
            value=np.array([1.0, 2.0]),
            units="kg/m^3"
        )

        converted = result.to("kg/m^3")

        assert converted is result
        mock_load_ureg.assert_not_called()

    @patch('kbkit.schema.property_result.load_unit_registry')
    def test_to_without_units_raises_error(self, mock_load_ureg):
        """Test that converting without original units raises ValueError."""
        result = PropertyResult(
            name="density",
            value=np.array([1.0, 2.0])
        )

        with pytest.raises(ValueError, match="Cannot convert PropertyResult.*without units"):
            result.to("kg/m^3")

    @patch('kbkit.schema.property_result.load_unit_registry')
    def test_to_converts_units(self, mock_load_ureg):
        """Test basic unit conversion."""
        # Mock the unit registry
        mock_ureg = Mock()
        mock_quantity = Mock()
        mock_quantity.to.return_value.magnitude = np.array([1000.0, 2000.0])
        mock_ureg.Quantity.return_value = mock_quantity
        mock_load_ureg.return_value = mock_ureg

        result = PropertyResult(
            name="density",
            value=np.array([1.0, 2.0]),
            units="g/cm^3"
        )

        converted = result.to("kg/m^3")

        assert converted.name == "density"
        assert converted.units == "kg/m^3"
        assert np.array_equal(converted.value, np.array([1000.0, 2000.0]))

    @patch('kbkit.schema.property_result.load_unit_registry')
    def test_to_preserves_metadata_for_non_kbi(self, mock_load_ureg):
        """Test that metadata is preserved for non-KBI properties."""
        mock_ureg = Mock()
        mock_quantity = Mock()
        mock_quantity.to.return_value.magnitude = np.array([100.0])
        mock_ureg.Quantity.return_value = mock_quantity
        mock_load_ureg.return_value = mock_ureg

        metadata = {"system": "test", "data": [1, 2, 3]}
        result = PropertyResult(
            name="density",
            value=np.array([1.0]),
            units="g/cm^3",
            metadata=metadata
        )

        converted = result.to("kg/m^3")

        assert converted.metadata == metadata

    @patch('kbkit.schema.property_result.load_unit_registry')
    def test_to_conversion_error_raises_valueerror(self, mock_load_ureg):
        """Test that conversion errors raise ValueError."""
        mock_ureg = Mock()
        mock_ureg.Quantity.side_effect = Exception("Invalid conversion")
        mock_load_ureg.return_value = mock_ureg

        result = PropertyResult(
            name="density",
            value=np.array([1.0]),
            units="kg/m^3"
        )

        with pytest.raises(ValueError, match="Cannot convert.*from.*to"):
            result.to("invalid_unit")

    @patch('kbkit.schema.property_result.load_unit_registry')
    def test_to_preserves_property_type(self, mock_load_ureg):
        """Test that property_type is preserved during conversion."""
        mock_ureg = Mock()
        mock_quantity = Mock()
        mock_quantity.to.return_value.magnitude = np.array([100.0])
        mock_ureg.Quantity.return_value = mock_quantity
        mock_load_ureg.return_value = mock_ureg

        result = PropertyResult(
            name="density",
            value=np.array([1.0]),
            units="g/cm^3",
            property_type="thermodynamic"
        )

        converted = result.to("kg/m^3")

        assert converted.property_type == "thermodynamic"

    @patch('kbkit.schema.property_result.load_unit_registry')
    def test_to_creates_new_instance(self, mock_load_ureg):
        """Test that to() creates a new PropertyResult instance."""
        mock_ureg = Mock()
        mock_quantity = Mock()
        mock_quantity.to.return_value.magnitude = np.array([100.0])
        mock_ureg.Quantity.return_value = mock_quantity
        mock_load_ureg.return_value = mock_ureg

        result = PropertyResult(
            name="density",
            value=np.array([1.0]),
            units="g/cm^3"
        )

        converted = result.to("kg/m^3")

        assert converted is not result
        assert isinstance(converted, PropertyResult)


class TestPropertyResultKBIConversion:
    """Test KBI-specific unit conversion."""

    @patch('kbkit.schema.property_result.load_unit_registry')
    def test_to_converts_kbi_metadata(self, mock_load_ureg):
        """Test that KBI metadata is converted."""
        # Create mock KBIMetadata
        mock_kbi_meta = Mock(spec=KBIMetadata)
        mock_kbi_meta.mols = ("Water", "Ethanol")
        mock_kbi_meta.r = np.array([0.1, 0.2, 0.3])
        mock_kbi_meta.g = np.array([1.0, 1.1, 1.2])
        mock_kbi_meta.rkbi = np.array([0.5, 0.6, 0.7])
        mock_kbi_meta.scaled_rkbi = np.array([0.05, 0.06, 0.07])
        mock_kbi_meta.r_fit = np.array([0.2, 0.3])
        mock_kbi_meta.scaled_rkbi_fit = np.array([0.06, 0.07])
        mock_kbi_meta.scaled_rkbi_est = np.array([0.06, 0.07])
        mock_kbi_meta.kbi_limit = 0.8

        metadata = {
            "system_1": {
                "Water-Ethanol": mock_kbi_meta
            }
        }

        # Mock unit registry
        mock_ureg = Mock()
        mock_quantity = Mock()
        mock_quantity.to.return_value.magnitude = np.array([1.0])

        def quantity_side_effect(value, units):
            mock_q = Mock()
            if isinstance(value, np.ndarray):
                mock_q.to.return_value.magnitude = value * 1000  # Simple conversion
            else:
                mock_q.to.return_value.magnitude = value * 1000
            return mock_q

        mock_ureg.Quantity.side_effect = quantity_side_effect
        mock_load_ureg.return_value = mock_ureg

        result = PropertyResult(
            name="kbi",
            value=np.array([[1.0, 2.0], [3.0, 4.0]]),
            units="nm^3",
            metadata=metadata
        )

        converted = result.to("A^3")

        # Check that metadata was converted
        assert "system_1" in converted.metadata
        assert "Water-Ethanol" in converted.metadata["system_1"]

    @patch('kbkit.schema.property_result.load_unit_registry')
    def test_to_handles_non_kbi_metadata_in_kbi_property(self, mock_load_ureg):
        """Test that non-KBIMetadata objects in KBI property are preserved."""
        metadata = {
            "system_1": {
                "other_data": {"value": 123}
            }
        }

        mock_ureg = Mock()
        mock_quantity = Mock()
        mock_quantity.to.return_value.magnitude = np.array([1.0])
        mock_ureg.Quantity.return_value = mock_quantity
        mock_load_ureg.return_value = mock_ureg

        result = PropertyResult(
            name="kbi",
            value=np.array([1.0]),
            units="nm^3",
            metadata=metadata
        )

        converted = result.to("A^3")

        assert converted.metadata["system_1"]["other_data"] == {"value": 123}

    @patch('kbkit.schema.property_result.load_unit_registry')
    def test_to_handles_non_dict_metadata_in_kbi(self, mock_load_ureg):
        """Test that non-dict metadata in KBI property is preserved."""
        metadata = {
            "system_1": "some_string_value"
        }

        mock_ureg = Mock()
        mock_quantity = Mock()
        mock_quantity.to.return_value.magnitude = np.array([1.0])
        mock_ureg.Quantity.return_value = mock_quantity
        mock_load_ureg.return_value = mock_ureg

        result = PropertyResult(
            name="kbi",
            value=np.array([1.0]),
            units="nm^3",
            metadata=metadata
        )

        converted = result.to("A^3")

        assert converted.metadata["system_1"] == "some_string_value"


class TestPropertyResultConvertSingleKBIMetadata:
    """Test _convert_single_kbi_metadata method."""

    @patch('kbkit.schema.property_result.load_unit_registry')
    def test_convert_single_kbi_metadata(self, mock_load_ureg):
        """Test conversion of a single KBIMetadata object."""
        # Create KBIMetadata
        kbi_meta = KBIMetadata(
            mols=("Water", "Ethanol"),
            r=np.array([0.1, 0.2, 0.3]),
            g=np.array([1.0, 1.1, 1.2]),
            rkbi=np.array([0.5, 0.6, 0.7]),
            scaled_rkbi=np.array([0.05, 0.06, 0.07]),
            r_fit=np.array([0.2, 0.3]),
            scaled_rkbi_fit=np.array([0.06, 0.07]),
            scaled_rkbi_est=np.array([0.06, 0.07]),
            kbi_limit=0.8
        )

        # Mock unit registry
        mock_ureg = Mock()

        def quantity_side_effect(value, units):
            mock_q = Mock()
            if isinstance(value, np.ndarray):
                mock_q.to.return_value.magnitude = value * 1000
            else:
                mock_q.to.return_value.magnitude = value * 1000
            return mock_q

        mock_ureg.Quantity.side_effect = quantity_side_effect
        mock_load_ureg.return_value = mock_ureg

        result = PropertyResult(
            name="kbi",
            value=np.array([1.0]),
            units="nm^3"
        )

        converted_meta = result._convert_single_kbi_metadata(kbi_meta, "A^3", mock_ureg)

        # Check that converted metadata is KBIMetadata
        assert isinstance(converted_meta, KBIMetadata)

        # Check that mols, r, and g are unchanged
        assert converted_meta.mols == kbi_meta.mols
        assert np.array_equal(converted_meta.r, kbi_meta.r)
        assert np.array_equal(converted_meta.g, kbi_meta.g)
        assert np.array_equal(converted_meta.r_fit, kbi_meta.r_fit)


class TestPropertyResultRepr:
    """Test PropertyResult string representations."""

    def test_repr_with_array(self):
        """Test __repr__ with array value."""
        result = PropertyResult(
            name="density",
            value=np.array([1.0, 2.0, 3.0]),
            units="kg/m^3"
        )

        repr_str = repr(result)

        assert "PropertyResult" in repr_str
        assert "name='density'" in repr_str
        assert "shape=(3,)" in repr_str
        assert "units='kg/m^3'" in repr_str

    def test_repr_with_scalar(self):
        """Test __repr__ with scalar value."""
        result = PropertyResult(
            name="temperature",
            value=np.array(298.15),
            units="K"
        )

        repr_str = repr(result)

        assert "PropertyResult" in repr_str
        assert "name='temperature'" in repr_str
        assert "value=" in repr_str
        assert "units='K'" in repr_str

    def test_repr_without_units(self):
        """Test __repr__ without units."""
        result = PropertyResult(
            name="ratio",
            value=np.array([1.0, 2.0])
        )

        repr_str = repr(result)

        assert "PropertyResult" in repr_str
        assert "name='ratio'" in repr_str
        assert "units" not in repr_str

    def test_repr_with_property_type(self):
        """Test __repr__ with property_type."""
        result = PropertyResult(
            name="kbi",
            value=np.array([1.0]),
            property_type="thermodynamic"
        )

        repr_str = repr(result)

        assert "type='thermodynamic'" in repr_str

    def test_str_with_units(self):
        """Test __str__ with units."""
        result = PropertyResult(
            name="density",
            value=np.array([1.0, 2.0]),
            units="kg/m^3"
        )

        str_repr = str(result)

        assert "density:" in str_repr
        assert "kg/m^3" in str_repr

    def test_str_without_units(self):
        """Test __str__ without units."""
        result = PropertyResult(
            name="ratio",
            value=np.array([1.0, 2.0])
        )

        str_repr = str(result)

        assert "ratio:" in str_repr
        assert "(dimensionless)" in str_repr


class TestPropertyResultEdgeCases:
    """Test edge cases for PropertyResult."""

    def test_empty_array_value(self):
        """Test with empty array."""
        result = PropertyResult(
            name="test",
            value=np.array([])
        )

        assert len(result.value) == 0
        assert result.value.shape == (0,)

    def test_multidimensional_array(self):
        """Test with multidimensional array."""
        value = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
        result = PropertyResult(
            name="tensor",
            value=value
        )

        assert result.value.shape == (2, 2, 2)
        assert np.array_equal(result.value, value)

    def test_complex_metadata_structure(self):
        """Test with complex nested metadata."""
        metadata = {
            "level1": {
                "level2": {
                    "level3": {
                        "data": [1, 2, 3],
                        "info": "test"
                    }
                }
            }
        }

        result = PropertyResult(
            name="test",
            value=np.array([1.0]),
            metadata=metadata
        )

        assert result.metadata == metadata

    @patch('kbkit.schema.property_result.load_unit_registry')
    def test_to_with_empty_metadata(self, mock_load_ureg):
        """Test unit conversion with empty metadata."""
        mock_ureg = Mock()
        mock_quantity = Mock()
        mock_quantity.to.return_value.magnitude = np.array([100.0])
        mock_ureg.Quantity.return_value = mock_quantity
        mock_load_ureg.return_value = mock_ureg

        result = PropertyResult(
            name="test",
            value=np.array([1.0]),
            units="m",
            metadata={}
        )

        converted = result.to("cm")

        assert converted.metadata == {}

    def test_very_long_property_name(self):
        """Test with very long property name."""
        long_name = "a" * 1000
        result = PropertyResult(
            name=long_name,
            value=np.array([1.0])
        )

        assert result.name == long_name
        assert len(result.name) == 1000

    def test_special_characters_in_name(self):
        """Test with special characters in name."""
        result = PropertyResult(
            name="property_with-special.chars!@#",
            value=np.array([1.0])
        )

        assert result.name == "property_with-special.chars!@#"

    def test_unicode_in_name(self):
        """Test with unicode characters in name."""
        result = PropertyResult(
            name="密度_température_плотность",
            value=np.array([1.0])
        )

        assert result.name == "密度_température_плотность"

    @patch('kbkit.schema.property_result.load_unit_registry')
    def test_to_with_none_metadata(self, mock_load_ureg):
        """Test that None metadata is handled correctly in conversion."""
        mock_ureg = Mock()
        mock_quantity = Mock()
        mock_quantity.to.return_value.magnitude = np.array([100.0])
        mock_ureg.Quantity.return_value = mock_quantity
        mock_load_ureg.return_value = mock_ureg

        result = PropertyResult(
            name="test",
            value=np.array([1.0]),
            units="m"
        )

        # Explicitly set metadata to None to test the condition
        result.metadata = None

        converted = result.to("cm")

        # Should handle None metadata gracefully
        assert converted.metadata == {}

    def test_negative_values(self):
        """Test with negative values."""
        result = PropertyResult(
            name="delta_g",
            value=np.array([-10.5, -20.3, -5.7]),
            units="kJ/mol"
        )

        assert np.all(result.value < 0)

    def test_very_large_values(self):
        """Test with very large values."""
        result = PropertyResult(
            name="avogadro",
            value=np.array([6.022e23]),
            units="1/mol"
        )

        assert result.value[0] > 1e23

    def test_very_small_values(self):
        """Test with very small values."""
        result = PropertyResult(
            name="planck",
            value=np.array([6.626e-34]),
            units="J*s"
        )

        assert result.value[0] < 1e-30


class TestPropertyResultMetadataHandling:
    """Test metadata handling in PropertyResult."""

    def test_metadata_is_mutable(self):
        """Test that metadata can be modified after creation."""
        result = PropertyResult(
            name="test",
            value=np.array([1.0]),
            metadata={"key": "value"}
        )

        result.metadata["new_key"] = "new_value"

        assert result.metadata["new_key"] == "new_value"
        assert len(result.metadata) == 2

    def test_metadata_with_numpy_arrays(self):
        """Test metadata containing numpy arrays."""
        metadata = {
            "array_data": np.array([1, 2, 3]),
            "matrix": np.array([[1, 2], [3, 4]])
        }

        result = PropertyResult(
            name="test",
            value=np.array([1.0]),
            metadata=metadata
        )

        assert np.array_equal(result.metadata["array_data"], np.array([1, 2, 3]))
        assert np.array_equal(result.metadata["matrix"], np.array([[1, 2], [3, 4]]))

    def test_metadata_with_mixed_types(self):
        """Test metadata with mixed data types."""
        metadata = {
            "string": "test",
            "int": 42,
            "float": 3.14,
            "list": [1, 2, 3],
            "dict": {"nested": "value"},
            "array": np.array([1, 2]),
            "none": None,
            "bool": True
        }

        result = PropertyResult(
            name="test",
            value=np.array([1.0]),
            metadata=metadata
        )

        assert result.metadata == metadata
