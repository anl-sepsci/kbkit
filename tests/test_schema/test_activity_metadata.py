"""Unit tests for activity coefficient containers."""

import numpy as np
import pytest

from kbkit.schema.activity_metadata import ActivityCoefficientResult, ActivityMetadata


class TestActivityCoefficientResult:
    """Test ActivityCoefficientResult dataclass."""

    def test_initialization_basic(self):
        """Test basic initialization without function."""
        mol = "Water"
        x = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        y = np.array([0.0, 0.1, 0.2, 0.3, 0.4])
        property_type = "derivative"

        result = ActivityCoefficientResult(
            mol=mol,
            x=x,
            y=y,
            property_type=property_type
        )

        assert result.mol == mol
        assert np.array_equal(result.x, x)
        assert np.array_equal(result.y, y)
        assert result.property_type == property_type
        assert result.fn is None

    def test_initialization_with_function(self):
        """Test initialization with polynomial function."""
        mol = "Ethanol"
        x = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        y = np.array([0.0, 0.1, 0.2, 0.3, 0.4])
        property_type = "integrated"
        fn = np.poly1d([1, 2, 3])  # x^2 + 2x + 3

        result = ActivityCoefficientResult(
            mol=mol,
            x=x,
            y=y,
            property_type=property_type,
            fn=fn
        )

        assert result.mol == mol
        assert result.fn == fn
        assert isinstance(result.fn, np.poly1d)

    def test_x_eval_with_function(self):
        """Test x_eval property when function is defined."""
        mol = "Water"
        x = np.array([0.0, 0.5, 1.0])
        y = np.array([0.0, 0.5, 1.0])
        fn = np.poly1d([1, 0])  # y = x

        result = ActivityCoefficientResult(
            mol=mol,
            x=x,
            y=y,
            property_type="derivative",
            fn=fn
        )

        x_eval = result.x_eval
        assert x_eval is not None
        assert isinstance(x_eval, np.ndarray)
        assert len(x_eval) == 101  # 0 to 1 with 0.01 step
        assert x_eval[0] == 0.0
        assert x_eval[-1] == 1.0

    def test_x_eval_without_function(self):
        """Test x_eval property when function is not defined."""
        mol = "Water"
        x = np.array([0.0, 0.5, 1.0])
        y = np.array([0.0, 0.5, 1.0])

        result = ActivityCoefficientResult(
            mol=mol,
            x=x,
            y=y,
            property_type="derivative"
        )

        assert result.x_eval is None

    def test_y_eval_with_function(self):
        """Test y_eval property when function is defined."""
        mol = "Water"
        x = np.array([0.0, 0.5, 1.0])
        y = np.array([0.0, 0.5, 1.0])
        fn = np.poly1d([2, 0])  # y = 2x

        result = ActivityCoefficientResult(
            mol=mol,
            x=x,
            y=y,
            property_type="derivative",
            fn=fn
        )

        y_eval = result.y_eval
        assert y_eval is not None
        assert isinstance(y_eval, np.ndarray)
        # y_eval is fn evaluated over x_eval (101 points), not over the
        # original 3-point x array.
        expected = fn(result.x_eval)
        assert np.allclose(y_eval, expected)

    def test_y_eval_without_function(self):
        """Test y_eval property when function is not defined."""
        mol = "Water"
        x = np.array([0.0, 0.5, 1.0])
        y = np.array([0.0, 0.5, 1.0])

        result = ActivityCoefficientResult(
            mol=mol,
            x=x,
            y=y,
            property_type="derivative"
        )

        assert result.y_eval is None

    def test_has_fn_true(self):
        """Test has_fn property when function is defined."""
        mol = "Water"
        x = np.array([0.0, 0.5, 1.0])
        y = np.array([0.0, 0.5, 1.0])
        fn = np.poly1d([1, 0])

        result = ActivityCoefficientResult(
            mol=mol,
            x=x,
            y=y,
            property_type="derivative",
            fn=fn
        )

        assert result.has_fn is True

    def test_has_fn_false(self):
        """Test has_fn property when function is not defined."""
        mol = "Water"
        x = np.array([0.0, 0.5, 1.0])
        y = np.array([0.0, 0.5, 1.0])

        result = ActivityCoefficientResult(
            mol=mol,
            x=x,
            y=y,
            property_type="derivative"
        )

        assert result.has_fn is False

    def test_property_type_derivative(self):
        """Test with derivative property type."""
        result = ActivityCoefficientResult(
            mol="Water",
            x=np.array([0.0, 1.0]),
            y=np.array([0.0, 1.0]),
            property_type="derivative"
        )

        assert result.property_type == "derivative"

    def test_property_type_integrated(self):
        """Test with integrated property type."""
        result = ActivityCoefficientResult(
            mol="Water",
            x=np.array([0.0, 1.0]),
            y=np.array([0.0, 1.0]),
            property_type="integrated"
        )

        assert result.property_type == "integrated"

    def test_polynomial_evaluation(self):
        """Test that polynomial function evaluates correctly."""
        mol = "Water"
        x = np.array([0.0, 0.5, 1.0])
        y = np.array([3.0, 4.0, 5.0])
        # Create polynomial: y = 2x + 3
        fn = np.poly1d([2, 3])

        result = ActivityCoefficientResult(
            mol=mol,
            x=x,
            y=y,
            property_type="derivative",
            fn=fn
        )

        # Test evaluation at specific points
        assert fn(0.0) == 3.0
        assert fn(0.5) == 4.0
        assert fn(1.0) == 5.0

        # y_eval is fn evaluated over the 101-point x_eval linspace, not
        # the original 3-point x.
        y_eval = result.y_eval
        assert np.allclose(y_eval, fn(result.x_eval))


class TestActivityMetadata:
    """Test ActivityMetadata dataclass."""

    @pytest.fixture
    def sample_results(self):
        """Create sample ActivityCoefficientResult objects."""
        result1 = ActivityCoefficientResult(
            mol="Water",
            x=np.array([0.0, 0.5, 1.0]),
            y=np.array([0.0, 0.5, 1.0]),
            property_type="derivative",
            fn=np.poly1d([1, 0])
        )

        result2 = ActivityCoefficientResult(
            mol="Ethanol",
            x=np.array([0.0, 0.5, 1.0]),
            y=np.array([0.0, 0.25, 1.0]),
            property_type="derivative",
            fn=np.poly1d([1, 0, 0])
        )

        result3 = ActivityCoefficientResult(
            mol="Water",
            x=np.array([0.0, 0.5, 1.0]),
            y=np.array([1.0, 1.5, 2.0]),
            property_type="integrated"
        )

        result4 = ActivityCoefficientResult(
            mol="Ethanol",
            x=np.array([0.0, 0.5, 1.0]),
            y=np.array([2.0, 2.5, 3.0]),
            property_type="integrated"
        )

        return [result1, result2, result3, result4]

    def test_initialization(self, sample_results):
        """Test basic initialization."""
        metadata = ActivityMetadata(results=sample_results)

        assert metadata.results == sample_results
        assert len(metadata.results) == 4

    def test_initialization_empty(self):
        """Test initialization with empty list."""
        metadata = ActivityMetadata(results=[])

        assert metadata.results == []
        assert len(metadata.results) == 0

    def test_by_types_structure(self, sample_results):
        """Test by_types property structure."""
        metadata = ActivityMetadata(results=sample_results)
        by_types = metadata.by_types

        assert isinstance(by_types, dict)
        assert "derivative" in by_types
        assert "integrated" in by_types
        assert isinstance(by_types["derivative"], dict)
        assert isinstance(by_types["integrated"], dict)

    def test_by_types_derivative_content(self, sample_results):
        """Test by_types derivative content."""
        metadata = ActivityMetadata(results=sample_results)
        by_types = metadata.by_types

        derivative_dict = by_types["derivative"]
        assert "Water" in derivative_dict
        assert "Ethanol" in derivative_dict
        assert derivative_dict["Water"].mol == "Water"
        assert derivative_dict["Ethanol"].mol == "Ethanol"

    def test_by_types_integrated_content(self, sample_results):
        """Test by_types integrated content."""
        metadata = ActivityMetadata(results=sample_results)
        by_types = metadata.by_types

        integrated_dict = by_types["integrated"]
        assert "Water" in integrated_dict
        assert "Ethanol" in integrated_dict
        assert integrated_dict["Water"].mol == "Water"
        assert integrated_dict["Ethanol"].mol == "Ethanol"

    def test_get_derivative_lowercase(self, sample_results):
        """Test get method with derivative (lowercase)."""
        metadata = ActivityMetadata(results=sample_results)

        result = metadata.get("Water", "derivative")

        assert isinstance(result, ActivityCoefficientResult)
        assert result.mol == "Water"
        assert result.property_type == "derivative"

    def test_get_derivative_uppercase(self, sample_results):
        """Test get method with Derivative (uppercase)."""
        metadata = ActivityMetadata(results=sample_results)

        result = metadata.get("Water", "Derivative")

        assert isinstance(result, ActivityCoefficientResult)
        assert result.mol == "Water"
        assert result.property_type == "derivative"

    def test_get_integrated_lowercase(self, sample_results):
        """Test get method with integrated (lowercase)."""
        metadata = ActivityMetadata(results=sample_results)

        result = metadata.get("Ethanol", "integrated")

        assert isinstance(result, ActivityCoefficientResult)
        assert result.mol == "Ethanol"
        assert result.property_type == "integrated"

    def test_get_integrated_uppercase(self, sample_results):
        """Test get method with Integrated (uppercase)."""
        metadata = ActivityMetadata(results=sample_results)

        result = metadata.get("Ethanol", "Integrated")

        assert isinstance(result, ActivityCoefficientResult)
        assert result.mol == "Ethanol"
        assert result.property_type == "integrated"

    def test_get_different_molecules(self, sample_results):
        """Test get method with different molecules."""
        metadata = ActivityMetadata(results=sample_results)

        water_deriv = metadata.get("Water", "derivative")
        ethanol_deriv = metadata.get("Ethanol", "derivative")

        assert water_deriv.mol == "Water"
        assert ethanol_deriv.mol == "Ethanol"
        assert water_deriv != ethanol_deriv

    def test_get_nonexistent_molecule(self, sample_results):
        """Test get method with non-existent molecule raises KeyError."""
        metadata = ActivityMetadata(results=sample_results)

        with pytest.raises(KeyError):
            metadata.get("Methanol", "derivative")

    def test_by_types_empty_results(self):
        """Test by_types with empty results list."""
        metadata = ActivityMetadata(results=[])
        by_types = metadata.by_types

        assert by_types == {}

    def test_by_types_single_type(self):
        """Test by_types with only one property type."""
        results = [
            ActivityCoefficientResult(
                mol="Water",
                x=np.array([0.0, 1.0]),
                y=np.array([0.0, 1.0]),
                property_type="derivative"
            ),
            ActivityCoefficientResult(
                mol="Ethanol",
                x=np.array([0.0, 1.0]),
                y=np.array([0.0, 1.0]),
                property_type="derivative"
            )
        ]

        metadata = ActivityMetadata(results=results)
        by_types = metadata.by_types

        assert "derivative" in by_types
        assert "integrated" not in by_types
        assert len(by_types["derivative"]) == 2

    def test_by_types_multiple_calls_same_result(self, sample_results):
        """Test that by_types returns consistent results on multiple calls."""
        metadata = ActivityMetadata(results=sample_results)

        by_types1 = metadata.by_types
        by_types2 = metadata.by_types

        # Should return the same structure
        assert by_types1.keys() == by_types2.keys()
        assert by_types1["derivative"].keys() == by_types2["derivative"].keys()
        assert by_types1["integrated"].keys() == by_types2["integrated"].keys()


class TestActivityCoefficientResultEdgeCases:
    """Test edge cases for ActivityCoefficientResult."""

    def test_empty_arrays(self):
        """Test with empty numpy arrays."""
        result = ActivityCoefficientResult(
            mol="Water",
            x=np.array([]),
            y=np.array([]),
            property_type="derivative"
        )

        assert len(result.x) == 0
        assert len(result.y) == 0

    def test_single_point(self):
        """Test with single data point."""
        result = ActivityCoefficientResult(
            mol="Water",
            x=np.array([0.5]),
            y=np.array([0.25]),
            property_type="derivative"
        )

        assert len(result.x) == 1
        assert len(result.y) == 1
        assert result.x[0] == 0.5
        assert result.y[0] == 0.25

    def test_high_order_polynomial(self):
        """Test with high-order polynomial."""
        # 5th order polynomial
        fn = np.poly1d([1, 2, 3, 4, 5, 6])

        result = ActivityCoefficientResult(
            mol="Water",
            x=np.array([0.0, 0.5, 1.0]),
            y=np.array([6.0, 10.0, 21.0]),
            property_type="derivative",
            fn=fn
        )

        assert result.has_fn is True
        assert result.fn.order == 5

    def test_constant_polynomial(self):
        """Test with constant polynomial (0th order)."""
        fn = np.poly1d([5])  # y = 5

        result = ActivityCoefficientResult(
            mol="Water",
            x=np.array([0.0, 0.5, 1.0]),
            y=np.array([5.0, 5.0, 5.0]),
            property_type="derivative",
            fn=fn
        )

        # y_eval has 101 elements (one per x_eval point); a constant
        # polynomial means every single one equals 5.
        y_eval = result.y_eval
        assert len(y_eval) == 101
        assert np.all(y_eval == 5.0)

    def test_negative_values(self):
        """Test with negative x and y values."""
        result = ActivityCoefficientResult(
            mol="Water",
            x=np.array([-1.0, 0.0, 1.0]),
            y=np.array([-2.0, 0.0, 2.0]),
            property_type="derivative"
        )

        assert result.x[0] == -1.0
        assert result.y[0] == -2.0

    def test_large_arrays(self):
        """Test with large arrays."""
        x = np.linspace(0, 1, 10000)
        y = np.sin(x)

        result = ActivityCoefficientResult(
            mol="Water",
            x=x,
            y=y,
            property_type="derivative"
        )

        assert len(result.x) == 10000
        assert len(result.y) == 10000


class TestActivityMetadataEdgeCases:
    """Test edge cases for ActivityMetadata."""

    def test_duplicate_molecules_same_type(self):
        """Test with duplicate molecules of same type (last one wins)."""
        results = [
            ActivityCoefficientResult(
                mol="Water",
                x=np.array([0.0, 1.0]),
                y=np.array([0.0, 1.0]),
                property_type="derivative"
            ),
            ActivityCoefficientResult(
                mol="Water",
                x=np.array([0.0, 1.0]),
                y=np.array([1.0, 2.0]),  # Different y values
                property_type="derivative"
            )
        ]

        metadata = ActivityMetadata(results=results)
        by_types = metadata.by_types

        # Should only have one Water entry (the last one)
        assert len(by_types["derivative"]) == 1
        water_result = by_types["derivative"]["Water"]
        assert np.array_equal(water_result.y, np.array([1.0, 2.0]))

    def test_many_molecules(self):
        """Test with many different molecules."""
        results = [
            ActivityCoefficientResult(
                mol=f"Molecule_{i}",
                x=np.array([0.0, 1.0]),
                y=np.array([0.0, float(i)]),
                property_type="derivative"
            )
            for i in range(100)
        ]

        metadata = ActivityMetadata(results=results)
        by_types = metadata.by_types

        assert len(by_types["derivative"]) == 100

    def test_get_with_partial_match(self):
        """Test that get requires exact property_type match."""
        results = [
            ActivityCoefficientResult(
                mol="Water",
                x=np.array([0.0, 1.0]),
                y=np.array([0.0, 1.0]),
                property_type="derivative"
            )
        ]

        metadata = ActivityMetadata(results=results)

        # Should work with 'd' prefix
        result = metadata.get("Water", "deriv")
        assert result.property_type == "derivative"

        # Should work with 'D' prefix
        result = metadata.get("Water", "Deriv")
        assert result.property_type == "derivative"
