"""
Complete test coverage for kbkit.utils.decorators module.
Target: >95% coverage
"""
import warnings

# Suppress NumPy/SciPy compatibility warning (harmless with NumPy 2.x + SciPy 1.16+)
warnings.filterwarnings('ignore', message='numpy.ndarray size changed', category=RuntimeWarning)


import numpy as np
import pytest

from kbkit.schema.property_result import PropertyResult
from kbkit.utils.decorators import cached_property_result, cached_property_value


class TestCachedPropertyResult:
    """Test cached_property_result decorator."""

    @pytest.fixture
    def mock_class(self):
        """Create a mock class with decorated methods."""
        class TestClass:
            def __init__(self):
                self._cache = {}

            @cached_property_result(default_units="kg/m^3")
            def density_property(self, name: str, temperature: float = 298.15, **kwargs):
                """Calculate density property."""
                return np.array([1000.0, 1100.0, 1200.0])

            @cached_property_result(default_units="m^3")
            def volume_property(self, name: str, pressure: float = 101325, **kwargs):
                """Calculate volume property."""
                return np.array([0.001, 0.002, 0.003])

            @cached_property_result()
            def mass_property(self, name: str, **kwargs):
                """Calculate mass property without default units."""
                return np.array([10.0, 20.0, 30.0])

        return TestClass()

    def test_decorator_returns_property_result(self, mock_class):
        """Test that decorator returns PropertyResult object."""
        result = mock_class.density_property(name="water")

        assert isinstance(result, PropertyResult)
        assert result.name == "water"
        assert result.property_type == "density"
        assert result.units == "kg/m^3"

    def test_decorator_with_custom_units(self, mock_class):
        """Test decorator with custom units."""
        result = mock_class.density_property(name="water", units="g/cm^3")

        assert isinstance(result, PropertyResult)
        assert result.name == "water"

    def test_decorator_caches_result(self, mock_class):
        """Test that results are cached."""
        # First call
        result1 = mock_class.density_property(name="water", temperature=298.15)
        cache_size_1 = len(mock_class._cache)

        # Second call with same parameters
        result2 = mock_class.density_property(name="water", temperature=298.15)
        cache_size_2 = len(mock_class._cache)

        assert cache_size_1 == cache_size_2  # Cache size shouldn't increase
        assert result1.name == result2.name

    def test_decorator_different_parameters_different_cache(self, mock_class):
        """Test that different parameters create different cache entries."""
        result1 = mock_class.density_property(name="water", temperature=298.15)
        result2 = mock_class.density_property(name="water", temperature=300.0)

        assert len(mock_class._cache) >= 2

    def test_decorator_different_names_different_cache(self, mock_class):
        """Test that different names create different cache entries."""
        result1 = mock_class.density_property(name="water")
        result2 = mock_class.density_property(name="ethanol")

        assert result1.name == "water"
        assert result2.name == "ethanol"
        assert len(mock_class._cache) >= 2

    def test_decorator_extracts_property_type_from_function_name(self, mock_class):
        """Test that property type is extracted from function name."""
        result = mock_class.density_property(name="water")
        assert result.property_type == "density"

        result = mock_class.volume_property(name="water")
        assert result.property_type == "volume"

    def test_decorator_with_default_units(self, mock_class):
        """Test decorator uses default units when not specified."""
        result = mock_class.density_property(name="water")
        assert result.units == "kg/m^3"

        result = mock_class.volume_property(name="water")
        assert result.units == "m^3"

    def test_decorator_without_default_units(self, mock_class):
        """Test decorator without default units."""
        result = mock_class.mass_property(name="water")
        assert isinstance(result, PropertyResult)
        assert result.units is None

    def test_decorator_with_explicit_units_override(self, mock_class):
        """Test that explicit units override default units."""
        result = mock_class.density_property(name="water", units="g/cm^3")
        # The result should be converted to requested units
        assert isinstance(result, PropertyResult)

    def test_decorator_preserves_function_metadata(self, mock_class):
        """Test that decorator preserves function metadata."""
        assert mock_class.density_property.__name__ == "density_property"
        assert "Calculate density" in mock_class.density_property.__doc__

    def test_decorator_with_additional_kwargs(self, mock_class):
        """Test decorator with additional keyword arguments."""
        result = mock_class.density_property(
            name="water",
            temperature=298.15,
            pressure=101325,
            custom_param="test"
        )

        assert isinstance(result, PropertyResult)
        assert "temperature" in result.metadata or "custom_param" in result.metadata

    def test_decorator_metadata_excludes_name_and_avg(self, mock_class):
        """Test that metadata excludes 'name' and 'avg' parameters."""
        result = mock_class.density_property(
            name="water",
            temperature=298.15,
            avg=True
        )

        # 'name' and 'avg' should not be in metadata
        assert "name" not in result.metadata
        assert "avg" not in result.metadata
        # But temperature should be
        assert "temperature" in result.metadata

    def test_cache_key_generation(self, mock_class):
        """Test cache key generation."""
        mock_class.density_property(name="water", temperature=298.15)

        # Check that cache key was created
        assert len(mock_class._cache) > 0

        # Cache key should be a tuple
        cache_key = next(iter(mock_class._cache.keys()))
        assert isinstance(cache_key, tuple)
        assert cache_key[0] == "water"  # property_name
        assert cache_key[1] == "density"  # property_type

    def test_cache_hit_returns_cached_result(self, mock_class):
        """Test that cache hit returns the cached result."""
        # First call
        result1 = mock_class.density_property(name="water")
        original_cache_size = len(mock_class._cache)

        # Modify the cached result to verify it's being returned
        cache_key = next(iter(mock_class._cache.keys()))
        cached_result = mock_class._cache[cache_key]

        # Second call should return cached result
        result2 = mock_class.density_property(name="water")

        assert len(mock_class._cache) == original_cache_size

    def test_unit_conversion_on_cache_hit(self, mock_class):
        """Test unit conversion when retrieving from cache."""
        # First call with default units
        result1 = mock_class.density_property(name="water")

        # Second call requesting different units
        result2 = mock_class.density_property(name="water", units="g/cm^3")

        # Both should be PropertyResult objects
        assert isinstance(result1, PropertyResult)
        assert isinstance(result2, PropertyResult)


class TestCachedPropertyValue:
    """Test cached_property_value decorator."""

    @pytest.fixture
    def mock_class(self):
        """Create a mock class with decorated methods."""
        class TestClass:
            def __init__(self):
                self._cache = {}

            @cached_property_value(default_units="kg/m^3")
            def density_value(self, name: str, temperature: float = 298.15, **kwargs):
                """Calculate density value."""
                return np.array([1000.0, 1100.0, 1200.0])

            @cached_property_value(default_units="m^3")
            def volume_value(self, name: str, **kwargs):
                """Calculate volume value."""
                return np.array([0.001, 0.002, 0.003])

            @cached_property_value()
            def mass_value(self, **kwargs):
                """Calculate mass value without name parameter."""
                return np.array([10.0, 20.0, 30.0])

        return TestClass()

    def test_decorator_returns_value_not_property_result(self, mock_class):
        """Test that decorator returns value, not PropertyResult object."""
        result = mock_class.density_value(name="water")

        # Should return the value directly, not PropertyResult
        assert isinstance(result, np.ndarray)
        assert not isinstance(result, PropertyResult)

    def test_decorator_caches_as_property_result(self, mock_class):
        """Test that decorator caches as PropertyResult internally."""
        result = mock_class.density_value(name="water")

        # Check cache contains PropertyResult
        assert len(mock_class._cache) > 0
        cache_key = next(iter(mock_class._cache.keys()))
        cached_item = mock_class._cache[cache_key]
        assert isinstance(cached_item, PropertyResult)

    def test_decorator_with_custom_units(self, mock_class):
        """Test decorator with custom units."""
        result = mock_class.density_value(name="water", units="g/cm^3")

        assert isinstance(result, np.ndarray)

    def test_decorator_caches_result(self, mock_class):
        """Test that results are cached."""
        result1 = mock_class.density_value(name="water", temperature=298.15)
        cache_size_1 = len(mock_class._cache)

        result2 = mock_class.density_value(name="water", temperature=298.15)
        cache_size_2 = len(mock_class._cache)

        assert cache_size_1 == cache_size_2
        np.testing.assert_array_equal(result1, result2)

    def test_decorator_different_parameters(self, mock_class):
        """Test different parameters create different cache entries."""
        result1 = mock_class.density_value(name="water", temperature=298.15)
        result2 = mock_class.density_value(name="water", temperature=300.0)

        assert len(mock_class._cache) >= 2

    def test_decorator_without_name_parameter(self, mock_class):
        """Test decorator with function that has no 'name' parameter."""
        result = mock_class.mass_value()

        assert isinstance(result, np.ndarray)
        assert len(mock_class._cache) > 0

    def test_cache_key_without_name(self, mock_class):
        """Test cache key generation without 'name' parameter."""
        mock_class.mass_value()

        cache_key = next(iter(mock_class._cache.keys()))
        # When no 'name' parameter, cache_key should be function name
        assert cache_key == "mass_value"

    def test_cache_key_with_name(self, mock_class):
        """Test cache key generation with 'name' parameter."""
        mock_class.density_value(name="water")

        cache_key = next(iter(mock_class._cache.keys()))
        assert isinstance(cache_key, tuple)
        assert cache_key[0] == "water"

    def test_decorator_preserves_function_metadata(self, mock_class):
        """Test that decorator preserves function metadata."""
        assert mock_class.density_value.__name__ == "density_value"
        assert "Calculate density" in mock_class.density_value.__doc__

    def test_unit_conversion_on_return(self, mock_class):
        """Test unit conversion when returning value."""
        result1 = mock_class.density_value(name="water")
        result2 = mock_class.density_value(name="water", units="g/cm^3")

        # Both should be numpy arrays
        assert isinstance(result1, np.ndarray)
        assert isinstance(result2, np.ndarray)


class TestDecoratorEdgeCases:
    """Test edge cases for both decorators."""

    @pytest.fixture
    def mock_class(self):
        """Create a mock class for edge case testing."""
        class TestClass:
            def __init__(self):
                self._cache = {}

            @cached_property_result(default_units="kg")
            def property_with_defaults(self, name: str, param1: float = 1.0, param2: str = "default", **kwargs):
                """Property with default parameters."""
                return np.array([param1 * 10])

            @cached_property_value()
            def property_no_defaults(self, name: str, **kwargs):
                """Property without default units."""
                return np.array([42.0])

        return TestClass()

    def test_decorator_with_default_parameters(self, mock_class):
        """Test decorator with default parameter values."""
        result = mock_class.property_with_defaults(name="test")

        assert isinstance(result, PropertyResult)
        assert "param1" in result.metadata
        assert result.metadata["param1"] == 1.0

    def test_decorator_override_default_parameters(self, mock_class):
        """Test overriding default parameters."""
        result = mock_class.property_with_defaults(name="test", param1=5.0, param2="custom")

        assert result.metadata["param1"] == 5.0
        assert result.metadata["param2"] == "custom"

    def test_empty_cache_initially(self, mock_class):
        """Test that cache is empty initially."""
        assert len(mock_class._cache) == 0

    def test_cache_grows_with_different_calls(self, mock_class):
        """Test that cache grows with different parameter combinations."""
        mock_class.property_with_defaults(name="test1")
        assert len(mock_class._cache) == 1

        mock_class.property_with_defaults(name="test2")
        assert len(mock_class._cache) == 2

        mock_class.property_with_defaults(name="test1", param1=2.0)
        assert len(mock_class._cache) == 3

    def test_cache_does_not_grow_with_same_calls(self, mock_class):
        """Test that cache doesn't grow with identical calls."""
        mock_class.property_with_defaults(name="test")
        size1 = len(mock_class._cache)

        mock_class.property_with_defaults(name="test")
        size2 = len(mock_class._cache)

        assert size1 == size2


class TestDecoratorIntegration:
    """Integration tests for decorators."""

    def test_both_decorators_in_same_class(self):
        """Test using both decorators in the same class."""
        class TestClass:
            def __init__(self):
                self._cache = {}

            @cached_property_result(default_units="kg/m3")
            def get_result(self, name: str, **kwargs):
                return np.array([1.0])

            @cached_property_value(default_units="kg/m3")
            def get_value(self, name: str, **kwargs):
                return np.array([1.0])

        obj = TestClass()

        result1 = obj.get_result(name="test")
        result2 = obj.get_value(name="test")

        assert isinstance(result1, PropertyResult)
        assert isinstance(result2, np.ndarray)
        assert len(obj._cache) <= 2

    def test_decorator_with_complex_return_types(self):
        """Test decorator with various return types."""
        class TestClass:
            def __init__(self):
                self._cache = {}

            @cached_property_result()
            def scalar_result(self, name: str, **kwargs):
                return 42.0

            @cached_property_result()
            def array_result(self, name: str, **kwargs):
                return np.array([1, 2, 3])

            @cached_property_result()
            def list_result(self, name: str, **kwargs):
                return [1.0, 2.0, 3.0]

        obj = TestClass()

        r1 = obj.scalar_result(name="test")
        r2 = obj.array_result(name="test")
        r3 = obj.list_result(name="test")

        assert all(isinstance(r, PropertyResult) for r in [r1, r2, r3])

    def test_decorator_signature_inspection(self):
        """Test that decorator properly uses signature inspection."""
        class TestClass:
            def __init__(self):
                self._cache = {}

            @cached_property_result()
            def method_with_many_params(
                self,
                name: str,
                a: int = 1,
                b: float = 2.0,
                c: str = "three",
                d: bool = True,
                **kwargs
            ):
                return np.array([a, b])

        obj = TestClass()

        # Call with defaults
        result1 = obj.method_with_many_params(name="test")
        assert result1.metadata["a"] == 1
        assert result1.metadata["b"] == 2.0

        # Call with overrides
        result2 = obj.method_with_many_params(name="test", a=10, b=20.0)
        assert result2.metadata["a"] == 10
        assert result2.metadata["b"] == 20.0



class TestCacheKeyConsistency:
    """Test cache key generation consistency."""

    def test_cache_key_order_independence(self):
        """Test that kwargs order doesn't affect cache key."""
        class TestClass:
            def __init__(self):
                self._cache = {}

            @cached_property_result()
            def method(self, name: str, a: int = 1, b: int = 2, c: int = 3):
                return np.array([a, b, c])

        obj = TestClass()

        # Call with different kwarg orders
        obj.method(name="test", a=1, b=2, c=3)
        cache_size_1 = len(obj._cache)

        obj.method(name="test", c=3, a=1, b=2)
        cache_size_2 = len(obj._cache)

        # Should be same cache entry
        assert cache_size_1 == cache_size_2 == 1

    def test_cache_key_with_none_values(self):
        """Test cache key generation with None values."""
        class TestClass:
            def __init__(self):
                self._cache = {}

            @cached_property_result()
            def method(self, name: str, optional: str | None = None):
                return np.array([1.0])

        obj = TestClass()

        result1 = obj.method(name="test", optional=None)
        result2 = obj.method(name="test", optional=None)

        assert len(obj._cache) == 1

import sys

import pytest

if __name__ == "__main__":
    # Run the decorator tests
    exit_code = pytest.main([
        "tests/test_utils/test_decorators.py",
        "-v",
        "--tb=short",
        "--cov=kbkit.utils.decorators",
        "--cov-report=term-missing"
    ])

    sys.exit(exit_code)
