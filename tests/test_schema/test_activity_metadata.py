"""
Complete test coverage for kbkit.schema.activity_metadata module.
Target: >95% coverage
"""
import warnings

# Suppress NumPy/SciPy compatibility warning (harmless with NumPy 2.x + SciPy 1.16+)
warnings.filterwarnings('ignore', message='numpy.ndarray size changed', category=RuntimeWarning)

import numpy as np
import pytest

from kbkit.schema.activity_metadata import ActivityCoefficientResult, ActivityMetadata


class TestActivityCoefficientResult:
    """Test ActivityCoefficientResult dataclass."""

    def test_create_basic_result(self):
        """Test creating basic ActivityCoefficientResult."""
        x = np.array([0.0, 0.5, 1.0])
        y = np.array([1.0, 0.9, 0.8])

        result = ActivityCoefficientResult(
            mol="water",
            x=x,
            y=y,
            property_type="derivative"
        )

        assert result.mol == "water"
        np.testing.assert_array_equal(result.x, x)
        np.testing.assert_array_equal(result.y, y)
        assert result.property_type == "derivative"
        assert result.fn is None

    def test_create_result_with_function(self):
        """Test creating ActivityCoefficientResult with function."""
        x = np.array([0.0, 0.5, 1.0])
        y = np.array([1.0, 0.9, 0.8])

        def poly_fn(x):
            return 1.0 - 0.2 * x

        result = ActivityCoefficientResult(
            mol="ethanol",
            x=x,
            y=y,
            property_type="integrated",
            fn=poly_fn
        )

        assert result.mol == "ethanol"
        assert result.fn is not None
        assert callable(result.fn)

    def test_property_type_derivative(self):
        """Test with derivative property type."""
        result = ActivityCoefficientResult(
            mol="water",
            x=np.array([0.0, 1.0]),
            y=np.array([1.0, 0.8]),
            property_type="derivative"
        )

        assert result.property_type == "derivative"

    def test_property_type_integrated(self):
        """Test with integrated property type."""
        result = ActivityCoefficientResult(
            mol="water",
            x=np.array([0.0, 1.0]),
            y=np.array([1.0, 0.8]),
            property_type="integrated"
        )

        assert result.property_type == "integrated"

    def test_x_eval_without_function(self):
        """Test x_eval property when no function is defined."""
        result = ActivityCoefficientResult(
            mol="water",
            x=np.array([0.0, 1.0]),
            y=np.array([1.0, 0.8]),
            property_type="derivative"
        )

        assert result.x_eval is None

    def test_x_eval_with_function(self):
        """Test x_eval property when function is defined."""
        def poly_fn(x):
            return 1.0 - 0.2 * x

        result = ActivityCoefficientResult(
            mol="water",
            x=np.array([0.0, 1.0]),
            y=np.array([1.0, 0.8]),
            property_type="derivative",
            fn=poly_fn
        )

        x_eval = result.x_eval
        assert x_eval is not None
        assert isinstance(x_eval, np.ndarray)
        # Should be from 0 to 1 with step 1
        expected = np.arange(0, 1.01, 1)
        np.testing.assert_array_almost_equal(x_eval, expected)

    def test_y_eval_without_function(self):
        """Test y_eval property when no function is defined."""
        result = ActivityCoefficientResult(
            mol="water",
            x=np.array([0.0, 1.0]),
            y=np.array([1.0, 0.8]),
            property_type="derivative"
        )

        assert result.y_eval is None

    def test_y_eval_with_function(self):
        """Test y_eval property when function is defined."""
        def poly_fn(x):
            return 1.0 - 0.2 * x

        x = np.array([0.0, 0.5, 1.0])
        y = np.array([1.0, 0.9, 0.8])

        result = ActivityCoefficientResult(
            mol="water",
            x=x,
            y=y,
            property_type="derivative",
            fn=poly_fn
        )

        y_eval = result.y_eval
        assert y_eval is not None
        assert isinstance(y_eval, np.ndarray)
        # Should be function evaluated at x
        expected = poly_fn(x)
        np.testing.assert_array_almost_equal(y_eval, expected)

    def test_has_fn_false(self):
        """Test has_fn property when no function is defined."""
        result = ActivityCoefficientResult(
            mol="water",
            x=np.array([0.0, 1.0]),
            y=np.array([1.0, 0.8]),
            property_type="derivative"
        )

        assert result.has_fn is False

    def test_has_fn_true(self):
        """Test has_fn property when function is defined."""
        def poly_fn(x):
            return 1.0 - 0.2 * x

        result = ActivityCoefficientResult(
            mol="water",
            x=np.array([0.0, 1.0]),
            y=np.array([1.0, 0.8]),
            property_type="derivative",
            fn=poly_fn
        )

        assert result.has_fn is True

    def test_function_evaluation(self):
        """Test that function can be evaluated."""
        def linear_fn(x):
            return 2.0 * x + 1.0

        result = ActivityCoefficientResult(
            mol="water",
            x=np.array([0.0, 0.5, 1.0]),
            y=np.array([1.0, 2.0, 3.0]),
            property_type="derivative",
            fn=linear_fn
        )

        # Test function evaluation
        test_x = np.array([0.0, 0.5, 1.0])
        expected = np.array([1.0, 2.0, 3.0])
        np.testing.assert_array_almost_equal(result.fn(test_x), expected)


class TestActivityMetadata:
    """Test ActivityMetadata dataclass."""

    @pytest.fixture
    def sample_results(self):
        """Create sample ActivityCoefficientResult objects."""
        results = [
            ActivityCoefficientResult(
                mol="water",
                x=np.array([0.0, 1.0]),
                y=np.array([1.0, 0.8]),
                property_type="derivative"
            ),
            ActivityCoefficientResult(
                mol="ethanol",
                x=np.array([0.0, 1.0]),
                y=np.array([1.0, 0.9]),
                property_type="derivative"
            ),
            ActivityCoefficientResult(
                mol="water",
                x=np.array([0.0, 1.0]),
                y=np.array([0.0, 0.2]),
                property_type="integrated"
            ),
            ActivityCoefficientResult(
                mol="ethanol",
                x=np.array([0.0, 1.0]),
                y=np.array([0.0, 0.1]),
                property_type="integrated"
            )
        ]
        return results

    def test_create_metadata(self, sample_results):
        """Test creating ActivityMetadata."""
        metadata = ActivityMetadata(results=sample_results)

        assert isinstance(metadata, ActivityMetadata)
        assert len(metadata.results) == 4

    def test_empty_results(self):
        """Test creating ActivityMetadata with empty results."""
        metadata = ActivityMetadata(results=[])

        assert len(metadata.results) == 0
        assert metadata.by_types == {}

    def test_by_types_property(self, sample_results):
        """Test by_types property groups results correctly."""
        metadata = ActivityMetadata(results=sample_results)

        by_types = metadata.by_types

        assert isinstance(by_types, dict)
        assert "derivative" in by_types
        assert "integrated" in by_types

        # Check derivative results
        assert "water" in by_types["derivative"]
        assert "ethanol" in by_types["derivative"]

        # Check integrated results
        assert "water" in by_types["integrated"]
        assert "ethanol" in by_types["integrated"]

    def test_by_types_structure(self, sample_results):
        """Test structure of by_types dictionary."""
        metadata = ActivityMetadata(results=sample_results)

        by_types = metadata.by_types

        # Check that values are ActivityCoefficientResult objects
        water_deriv = by_types["derivative"]["water"]
        assert isinstance(water_deriv, ActivityCoefficientResult)
        assert water_deriv.mol == "water"
        assert water_deriv.property_type == "derivative"

    def test_get_derivative_result(self, sample_results):
        """Test getting derivative result."""
        metadata = ActivityMetadata(results=sample_results)

        result = metadata.get("water", "derivative")

        assert isinstance(result, ActivityCoefficientResult)
        assert result.mol == "water"
        assert result.property_type == "derivative"

    def test_get_integrated_result(self, sample_results):
        """Test getting integrated result."""
        metadata = ActivityMetadata(results=sample_results)

        result = metadata.get("ethanol", "integrated")

        assert isinstance(result, ActivityCoefficientResult)
        assert result.mol == "ethanol"
        assert result.property_type == "integrated"

    def test_get_with_partial_property_type(self, sample_results):
        """Test get with partial property_type match."""
        metadata = ActivityMetadata(results=sample_results)

        # "d" should match "derivative"
        result = metadata.get("water", "d")
        assert result.property_type == "derivative"

        # "deriv" should match "derivative"
        result = metadata.get("water", "deriv")
        assert result.property_type == "derivative"

    def test_get_case_insensitive(self, sample_results):
        """Test that get is case-insensitive."""
        metadata = ActivityMetadata(results=sample_results)

        result1 = metadata.get("water", "Derivative")
        result2 = metadata.get("water", "DERIVATIVE")
        result3 = metadata.get("water", "derivative")

        assert result1.property_type == result2.property_type == result3.property_type

    def test_single_property_type(self):
        """Test with only one property type."""
        results = [
            ActivityCoefficientResult(
                mol="water",
                x=np.array([0.0, 1.0]),
                y=np.array([1.0, 0.8]),
                property_type="derivative"
            )
        ]

        metadata = ActivityMetadata(results=results)
        by_types = metadata.by_types

        assert len(by_types) == 1
        assert "derivative" in by_types

    def test_multiple_molecules_same_type(self):
        """Test with multiple molecules of same type."""
        results = [
            ActivityCoefficientResult(
                mol="water",
                x=np.array([0.0, 1.0]),
                y=np.array([1.0, 0.8]),
                property_type="derivative"
            ),
            ActivityCoefficientResult(
                mol="ethanol",
                x=np.array([0.0, 1.0]),
                y=np.array([1.0, 0.9]),
                property_type="derivative"
            ),
            ActivityCoefficientResult(
                mol="methanol",
                x=np.array([0.0, 1.0]),
                y=np.array([1.0, 0.85]),
                property_type="derivative"
            )
        ]

        metadata = ActivityMetadata(results=results)
        by_types = metadata.by_types

        assert len(by_types["derivative"]) == 3
        assert "water" in by_types["derivative"]
        assert "ethanol" in by_types["derivative"]
        assert "methanol" in by_types["derivative"]


class TestIntegration:
    """Integration tests combining both classes."""

    def test_workflow_with_functions(self):
        """Test typical workflow with polynomial functions."""
        def water_fn(x):
            return 1.0 - 0.2 * x

        def ethanol_fn(x):
            return 1.0 - 0.1 * x

        results = [
            ActivityCoefficientResult(
                mol="water",
                x=np.array([0.0, 0.5, 1.0]),
                y=np.array([1.0, 0.9, 0.8]),
                property_type="derivative",
                fn=water_fn
            ),
            ActivityCoefficientResult(
                mol="ethanol",
                x=np.array([0.0, 0.5, 1.0]),
                y=np.array([1.0, 0.95, 0.9]),
                property_type="derivative",
                fn=ethanol_fn
            )
        ]

        metadata = ActivityMetadata(results=results)

        # Get water result
        water_result = metadata.get("water", "derivative")
        assert water_result.has_fn is True
        assert water_result.y_eval is not None

        # Get ethanol result
        ethanol_result = metadata.get("ethanol", "derivative")
        assert ethanol_result.has_fn is True
        assert ethanol_result.y_eval is not None

    def test_mixed_with_and_without_functions(self):
        """Test with mix of results with and without functions."""
        def poly_fn(x):
            return 1.0 - 0.2 * x

        results = [
            ActivityCoefficientResult(
                mol="water",
                x=np.array([0.0, 1.0]),
                y=np.array([1.0, 0.8]),
                property_type="derivative",
                fn=poly_fn
            ),
            ActivityCoefficientResult(
                mol="ethanol",
                x=np.array([0.0, 1.0]),
                y=np.array([1.0, 0.9]),
                property_type="derivative"
                # No function
            )
        ]

        metadata = ActivityMetadata(results=results)

        water = metadata.get("water", "derivative")
        ethanol = metadata.get("ethanol", "derivative")

        assert water.has_fn is True
        assert ethanol.has_fn is False

    def test_complete_activity_coefficient_workflow(self):
        """Test complete workflow with both derivative and integrated."""
        # Create derivative results
        deriv_water = ActivityCoefficientResult(
            mol="water",
            x=np.linspace(0, 1, 11),
            y=np.linspace(1.0, 0.8, 11),
            property_type="derivative"
        )

        deriv_ethanol = ActivityCoefficientResult(
            mol="ethanol",
            x=np.linspace(0, 1, 11),
            y=np.linspace(1.0, 0.9, 11),
            property_type="derivative"
        )

        # Create integrated results
        integ_water = ActivityCoefficientResult(
            mol="water",
            x=np.linspace(0, 1, 11),
            y=np.linspace(0.0, 0.2, 11),
            property_type="integrated"
        )

        integ_ethanol = ActivityCoefficientResult(
            mol="ethanol",
            x=np.linspace(0, 1, 11),
            y=np.linspace(0.0, 0.1, 11),
            property_type="integrated"
        )

        metadata = ActivityMetadata(
            results=[deriv_water, deriv_ethanol, integ_water, integ_ethanol]
        )

        # Verify structure
        assert len(metadata.by_types) == 2
        assert len(metadata.by_types["derivative"]) == 2
        assert len(metadata.by_types["integrated"]) == 2

        # Get specific results
        water_deriv = metadata.get("water", "derivative")
        water_integ = metadata.get("water", "integrated")

        assert water_deriv.property_type == "derivative"
        assert water_integ.property_type == "integrated"
        assert water_deriv.mol == water_integ.mol == "water"


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_single_point_arrays(self):
        """Test with single-point arrays."""
        result = ActivityCoefficientResult(
            mol="water",
            x=np.array([0.5]),
            y=np.array([0.9]),
            property_type="derivative"
        )

        assert len(result.x) == 1
        assert len(result.y) == 1

    def test_large_arrays(self):
        """Test with large arrays."""
        n = 10000
        result = ActivityCoefficientResult(
            mol="water",
            x=np.linspace(0, 1, n),
            y=np.random.rand(n),
            property_type="derivative"
        )

        assert len(result.x) == n
        assert len(result.y) == n

    def test_function_with_complex_polynomial(self):
        """Test with complex polynomial function."""
        def complex_poly(x):
            return 1.0 - 0.5*x + 0.3*x**2 - 0.1*x**3

        result = ActivityCoefficientResult(
            mol="water",
            x=np.array([0.0, 0.5, 1.0]),
            y=np.array([1.0, 0.825, 0.7]),
            property_type="derivative",
            fn=complex_poly
        )

        assert result.has_fn is True
        y_eval = result.y_eval
        assert y_eval is not None

    def test_lambda_function(self):
        """Test with lambda function."""
        result = ActivityCoefficientResult(
            mol="water",
            x=np.array([0.0, 1.0]),
            y=np.array([1.0, 0.8]),
            property_type="derivative",
            fn=lambda x: 1.0 - 0.2 * x
        )

        assert result.has_fn is True
        assert callable(result.fn)

    def test_duplicate_molecules_different_types(self):
        """Test same molecule with different property types."""
        results = [
            ActivityCoefficientResult(
                mol="water",
                x=np.array([0.0, 1.0]),
                y=np.array([1.0, 0.8]),
                property_type="derivative"
            ),
            ActivityCoefficientResult(
                mol="water",
                x=np.array([0.0, 1.0]),
                y=np.array([0.0, 0.2]),
                property_type="integrated"
            )
        ]

        metadata = ActivityMetadata(results=results)

        deriv = metadata.get("water", "derivative")
        integ = metadata.get("water", "integrated")

        assert deriv.property_type == "derivative"
        assert integ.property_type == "integrated"
        np.testing.assert_array_equal(deriv.y, np.array([1.0, 0.8]))
        np.testing.assert_array_equal(integ.y, np.array([0.0, 0.2]))
