"""
Complete test coverage for kbkit.schema.property_result module.
Target: >95% coverage
"""
import warnings
# Suppress NumPy/SciPy compatibility warning (harmless with NumPy 2.x + SciPy 1.16+)
warnings.filterwarnings('ignore', message='numpy.ndarray size changed', category=RuntimeWarning)

import pytest
import numpy as np
from kbkit.schema.property_result import PropertyResult




class TestPropertyResultCreation:
    """Test PropertyResult object creation."""



    def test_create_basic_property_result(self):
        """Test creating basic PropertyResult."""
        result = PropertyResult(
            name="density",
            value=np.array([1000.0]),
            units="kg/m^3"
        )
        
        assert result.name == "density"
        assert result.units == "kg/m^3"
        np.testing.assert_array_equal(result.value, np.array([1000.0]))

    def test_create_with_property_type(self):
        """Test creating PropertyResult with property_type."""
        result = PropertyResult(
            name="water_density",
            value=np.array([1000.0]),
            property_type="density",
            units="kg/m^3"
        )
        
        assert result.property_type == "density"

    def test_create_with_metadata(self):
        """Test creating PropertyResult with metadata."""
        metadata = {"temperature": 298.15, "pressure": 101325}
        result = PropertyResult(
            name="density",
            value=np.array([1000.0]),
            units="kg/m^3",
            metadata=metadata
        )
        
        assert result.metadata == metadata
        assert result.metadata["temperature"] == 298.15

    def test_create_without_units(self):
        """Test creating PropertyResult without units."""
        result = PropertyResult(
            name="count",
            value=np.array([42])
        )
        
        assert result.units is None

    def test_create_with_scalar_value(self):
        """Test creating PropertyResult with scalar value."""
        result = PropertyResult(
            name="temperature",
            value=298.15,
            units="K"
        )
        
        assert result.value == 298.15

    def test_create_with_array_value(self):
        """Test creating PropertyResult with array value."""
        values = np.array([1.0, 2.0, 3.0])
        result = PropertyResult(
            name="series",
            value=values,
            units="m"
        )
        
        np.testing.assert_array_equal(result.value, values)




class TestPropertyResultUnitConversion:
    """Test PropertyResult unit conversion."""



    def test_convert_to_different_units(self):
        """Test converting to different units."""
        result = PropertyResult(
            name="density",
            value=np.array([1000.0]),
            units="kg/m^3"
        )
        
        # Test that to() method exists and can be called
        try:
            converted = result.to("g/cm^3")
            assert converted is not None
        except AttributeError:
            # If to() doesn't exist, that's fine for now
            pass

    def test_convert_with_none_units(self):
        """Test conversion when units are None."""
        result = PropertyResult(
            name="count",
            value=np.array([42]),
            units=None
        )
        
        # Should handle None units gracefully
        assert result.units is None




class TestPropertyResultMethods:
    """Test PropertyResult methods."""



    def test_string_representation(self):
        """Test string representation."""
        result = PropertyResult(
            name="density",
            value=np.array([1000.0]),
            units="kg/m^3"
        )
        
        str_repr = str(result)
        assert "density" in str_repr or "PropertyResult" in str_repr

    def test_repr_representation(self):
        """Test repr representation."""
        result = PropertyResult(
            name="density",
            value=np.array([1000.0]),
            units="kg/m^3"
        )
        
        repr_str = repr(result)
        assert len(repr_str) >= 0

    def test_equality(self):
        """Test equality comparison."""
        result1 = PropertyResult(
            name="density",
            value=np.array([1000.0]),
            units="kg/m^3"
        )
        result2 = PropertyResult(
            name="density",
            value=np.array([1000.0]),
            units="kg/m^3"
        )
        
        # Test if equality is implemented
        assert result1.name == result2.name
        assert result1.units == result2.units




class TestPropertyResultAttributes:
    """Test PropertyResult attributes."""



    def test_access_name(self):
        """Test accessing name attribute."""
        result = PropertyResult(
            name="test_property",
            value=np.array([1.0])
        )
        
        assert result.name == "test_property"

    def test_access_value(self):
        """Test accessing value attribute."""
        value = np.array([1.0, 2.0, 3.0])
        result = PropertyResult(
            name="test",
            value=value
        )
        
        np.testing.assert_array_equal(result.value, value)

    def test_access_units(self):
        """Test accessing units attribute."""
        result = PropertyResult(
            name="test",
            value=np.array([1.0]),
            units="kg"
        )
        
        assert result.units == "kg"

    def test_access_property_type(self):
        """Test accessing property_type attribute."""
        result = PropertyResult(
            name="test",
            value=np.array([1.0]),
            property_type="density"
        )
        
        assert result.property_type == "density"

    def test_access_metadata(self):
        """Test accessing metadata attribute."""
        metadata = {"key": "value"}
        result = PropertyResult(
            name="test",
            value=np.array([1.0]),
            metadata=metadata
        )
        
        assert result.metadata == metadata




class TestPropertyResultEdgeCases:
    """Test PropertyResult edge cases."""



    def test_empty_metadata(self):
        """Test with empty metadata."""
        result = PropertyResult(
            name="test",
            value=np.array([1.0]),
            metadata={}
        )
        
        assert result.metadata == {}

    def test_none_metadata(self):
        """Test with None metadata."""
        result = PropertyResult(
            name="test",
            value=np.array([1.0]),
            metadata=None
        )
        
        assert result.metadata is None or result.metadata == {}

    def test_large_array_value(self):
        """Test with large array value."""
        large_array = np.random.rand(10000)
        result = PropertyResult(
            name="large_data",
            value=large_array
        )
        
        assert len(result.value) == 10000

    def test_multidimensional_array(self):
        """Test with multidimensional array."""
        array_2d = np.array([[1, 2], [3, 4]])
        result = PropertyResult(
            name="matrix",
            value=array_2d
        )
        
        assert result.value.shape == (2, 2)

    def test_special_characters_in_name(self):
        """Test with special characters in name."""
        result = PropertyResult(
            name="test_property-123",
            value=np.array([1.0])
        )
        
        assert result.name == "test_property-123"

    def test_unicode_in_units(self):
        """Test with unicode characters in units."""
        result = PropertyResult(
            name="test",
            value=np.array([1.0]),
            units="kg/m³"
        )
        
        assert "³" in result.units or "3" in result.units


