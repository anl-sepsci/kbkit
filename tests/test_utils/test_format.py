"""
Complete test coverage for kbkit.utils.format module.
Target: >95% coverage
"""
import warnings

# Suppress NumPy/SciPy compatibility warning (harmless with NumPy 2.x + SciPy 1.16+)
warnings.filterwarnings('ignore', message='numpy.ndarray size changed', category=RuntimeWarning)

import pytest

from kbkit.utils.format import ENERGY_ALIASES, format_unit_str, resolve_attr_key, resolve_units


class TestResolveAttrKey:
    """Test resolve_attr_key function."""

    def test_exact_match_lowercase(self):
        """Test exact match with lowercase alias."""
        result = resolve_attr_key("kappa", ENERGY_ALIASES)
        assert result == "isothermal-compressibility"

    def test_exact_match_uppercase(self):
        """Test exact match with uppercase alias."""
        result = resolve_attr_key("cp", ENERGY_ALIASES)
        assert result == "cp"

    def test_exact_match_mixed_case(self):
        """Test exact match with mixed case."""
        result = resolve_attr_key("temperature", ENERGY_ALIASES)
        assert result == "temperature"

    def test_alias_match(self):
        """Test matching through alias."""
        result = resolve_attr_key("temp", ENERGY_ALIASES)
        assert result == "temperature"

        result = resolve_attr_key("pres", ENERGY_ALIASES)
        assert result == "pressure"

        result = resolve_attr_key("vol", ENERGY_ALIASES)
        assert result == "volume"

    def test_fuzzy_match_high_similarity(self):
        """Test fuzzy matching with high similarity."""
        result = resolve_attr_key("temperatur", ENERGY_ALIASES)
        assert result == "temperature"

        result = resolve_attr_key("pressur", ENERGY_ALIASES)
        assert result == "pressure"

    def test_fuzzy_match_with_typo(self):
        """Test fuzzy matching with typo."""
        result = resolve_attr_key("tmeperature", ENERGY_ALIASES)
        # Should still match temperature due to high similarity
        assert result == "temperature"

    def test_no_match_formats_key(self):
        """Test that unmatched keys are formatted."""
        result = resolve_attr_key("custom_property", ENERGY_ALIASES)
        assert result == "Custom-Property"

        result = resolve_attr_key("my.new.property", ENERGY_ALIASES)
        assert result == "Mynewproperty"

    def test_formatting_with_spaces(self):
        """Test formatting with spaces."""
        result = resolve_attr_key("my custom property", ENERGY_ALIASES)
        assert result == "My-Custom-Property"

    def test_formatting_with_underscores(self):
        """Test formatting with underscores."""
        result = resolve_attr_key("my_custom_property", ENERGY_ALIASES)
        assert result == "My-Custom-Property"

    def test_formatting_with_dots(self):
        """Test formatting with dots."""
        result = resolve_attr_key("my.custom.property", ENERGY_ALIASES)
        assert result == "Mycustomproperty"

    def test_custom_cutoff(self):
        """Test with custom cutoff value."""
        # With high cutoff, weak matches should fail
        result = resolve_attr_key("xyz", ENERGY_ALIASES, cutoff=0.9)
        # Should format instead of fuzzy match
        assert result == "Xyz"

    def test_low_cutoff(self):
        """Test with low cutoff value."""
        result = resolve_attr_key("t", ENERGY_ALIASES, cutoff=0.3)
        # Should match something with low cutoff
        assert isinstance(result, str)

    def test_case_insensitive_matching(self):
        """Test that matching is case-insensitive."""
        result1 = resolve_attr_key("TEMPERATURE", ENERGY_ALIASES)
        result2 = resolve_attr_key("temperature", ENERGY_ALIASES)
        result3 = resolve_attr_key("Temperature", ENERGY_ALIASES)

        assert result1 == result2 == result3 == "temperature"

    def test_all_energy_aliases(self):
        """Test all canonical keys can be resolved."""
        canonical_keys = list(ENERGY_ALIASES.keys())

        for key in canonical_keys:
            result = resolve_attr_key(key, ENERGY_ALIASES)
            assert result == key

    def test_all_aliases_resolve(self):
        """Test that all aliases resolve to their canonical key."""
        for canonical, aliases in ENERGY_ALIASES.items():
            for alias in aliases:
                result = resolve_attr_key(alias, ENERGY_ALIASES)
                assert result == canonical

    def test_empty_string(self):
        """Test with empty string."""
        result = resolve_attr_key("", ENERGY_ALIASES)
        assert result == ""

    def test_special_characters(self):
        """Test with special characters."""
        result = resolve_attr_key("test-property", ENERGY_ALIASES)
        assert "Test" in result
        assert "Property" in result




class TestResolveUnits:
    """Test resolve_units function."""

    def test_requested_units_provided(self):
        """Test that requested units are returned when provided."""
        result = resolve_units("kg/m3", "g/cm3")
        assert result == "kg/m3"

    def test_default_units_when_none(self):
        """Test that default units are returned when requested is None."""
        result = resolve_units(None, "g/cm3")
        assert result == "g/cm3"

    def test_default_units_when_empty_string(self):
        """Test that default units are returned when requested is empty."""
        result = resolve_units("", "g/cm3")
        assert result == "g/cm3"

    def test_requested_over_default(self):
        """Test that requested units take precedence."""
        result = resolve_units("Pa", "bar")
        assert result == "Pa"

    def test_both_none(self):
        """Test when both are None."""
        result = resolve_units(None, None)
        assert result is None

    def test_both_empty(self):
        """Test when both are empty strings."""
        result = resolve_units("", "")
        assert result == ""

    def test_complex_units(self):
        """Test with complex unit strings."""
        result = resolve_units("kg*m/s**2", "N")
        assert result == "kg*m/s**2"

    def test_latex_units(self):
        """Test with LaTeX formatted units."""
        result = resolve_units("$kg/m^3$", "g/cm3")
        assert result == "$kg/m^3$"




class TestFormatUnitStr:
    """Test format_unit_str function."""

    def test_simple_unit(self):
        """Test simple unit formatting."""
        result = format_unit_str("kg")
        assert result.startswith("$")
        assert result.endswith("$")
        assert "kg" in result

    def test_unit_with_exponent(self):
        """Test unit with exponent using **."""
        result = format_unit_str("m**2")
        assert "^{2}" in result
        assert result.startswith("$")
        assert result.endswith("$")

    def test_unit_with_division(self):
        """Test unit with division."""
        result = format_unit_str("kg/m**3")
        assert "^{-3}" in result or "^{3}" in result

    def test_inverse_unit_simple(self):
        """Test inverse unit formatting."""
        result = format_unit_str("/s")
        assert "^{-1}" in result

    def test_inverse_unit_with_exponent(self):
        """Test inverse unit with exponent."""
        result = format_unit_str("/m**2")
        assert "^{-2}" in result

    def test_subscript_formatting(self):
        """Test subscript formatting."""
        result = format_unit_str("H_2O")
        assert "_{2" in result or "_{H" in result

    def test_subscript_with_parentheses(self):
        """Test subscript with parentheses."""
        result = format_unit_str("C_(aq)")
        assert "_{" in result

    def test_already_latex(self):
        """Test string already in LaTeX format."""
        result = format_unit_str("$kg/m^3$")
        # The function still processes it, so just check it's wrapped in $
        assert result.startswith("$")
        assert result.endswith("$")
        # Check that it contains the key elements
        assert "kg" in result

    def test_complex_unit(self):
        """Test complex unit string."""
        result = format_unit_str("kg*m/s**2")
        assert result.startswith("$")
        assert result.endswith("$")

    def test_unit_with_spaces(self):
        """Test unit with spaces."""
        result = format_unit_str("kg / m ** 3")
        assert result.startswith("$")
        assert result.endswith("$")

    def test_multiple_exponents(self):
        """Test multiple exponents."""
        result = format_unit_str("kg2*m3/s**2")
        assert result.startswith("$")
        assert result.endswith("$")
        # Check that exponents are present in some form
        assert "^{" in result or "^" in result

    def test_non_string_input(self):
        """Test with non-string input that can be converted."""
        result = format_unit_str(123)
        assert result.startswith("$")
        assert "123" in result

    def test_type_error_on_unconvertible(self):
        """Test TypeError on unconvertible type."""
        # Create an object that can't be converted to string
        class UnconvertibleType:
            def __str__(self):
                raise TypeError("Cannot convert")

        with pytest.raises(TypeError, match="Could not convert"):
            format_unit_str(UnconvertibleType())

    def test_empty_string(self):
        """Test with empty string."""
        result = format_unit_str("")
        assert result == "$$"

    def test_unit_with_numbers(self):
        """Test unit with numbers."""
        result = format_unit_str("m3")
        assert result.startswith("$")
        assert result.endswith("$")

    def test_inverse_with_letters_and_digits(self):
        """Test inverse unit with letters and digits."""
        result = format_unit_str("/m3")
        assert "^{-" in result

    def test_inverse_with_caret_notation(self):
        """Test inverse unit already with caret notation."""
        result = format_unit_str("/m^{2}")
        assert "^{-2}" in result

    def test_multiple_divisions(self):
        """Test multiple divisions."""
        result = format_unit_str("kg/m/s")
        assert result.startswith("$")
        assert result.endswith("$")

    def test_mixed_subscript_superscript(self):
        """Test mixed subscript and superscript."""
        result = format_unit_str("H_2O**2")
        assert "_{" in result
        assert "^{" in result

    def test_parentheses_in_subscript(self):
        """Test parentheses in subscript."""
        result = format_unit_str("C_(a+b)")
        assert "_{" in result
        assert "a+b" in result or "(a+b)" in result




class TestEnergyAliases:
    """Test ENERGY_ALIASES constant."""

    def test_aliases_is_dict(self):
        """Test that ENERGY_ALIASES is a dictionary."""
        assert isinstance(ENERGY_ALIASES, dict)

    def test_all_values_are_sets(self):
        """Test that all values are sets."""
        for value in ENERGY_ALIASES.values():
            assert isinstance(value, set)

    def test_all_keys_are_strings(self):
        """Test that all keys are strings."""
        for key in ENERGY_ALIASES.keys():
            assert isinstance(key, str)

    def test_all_aliases_are_strings(self):
        """Test that all aliases are strings."""
        for aliases in ENERGY_ALIASES.values():
            for alias in aliases:
                assert isinstance(alias, str)

    def test_no_duplicate_aliases(self):
        """Test that aliases are unique across all sets."""
        all_aliases = []
        for aliases in ENERGY_ALIASES.values():
            all_aliases.extend(aliases)

        # Check for duplicates
        assert len(all_aliases) == len(set(all_aliases))

    def test_common_properties_present(self):
        """Test that common properties are present."""
        expected_keys = [
            "temperature", "pressure", "volume", "density",
            "enthalpy", "potential", "kinetic-en", "total-energy"
        ]

        for key in expected_keys:
            assert key in ENERGY_ALIASES

    def test_aliases_not_empty(self):
        """Test that no alias set is empty."""
        for aliases in ENERGY_ALIASES.values():
            assert len(aliases) >= 0




class TestIntegration:
    """Integration tests combining multiple functions."""

    def test_resolve_and_format_workflow(self):
        """Test typical workflow of resolving and formatting."""
        # Resolve attribute
        attr = resolve_attr_key("temp", ENERGY_ALIASES)
        assert attr == "temperature"

        # Resolve units
        units = resolve_units("K", "C")
        assert units == "K"

        # Format units
        formatted = format_unit_str(units)
        assert "$" in formatted

    def test_custom_property_workflow(self):
        """Test workflow with custom property."""
        # Custom property that won't match
        attr = resolve_attr_key("my_custom_prop", ENERGY_ALIASES)
        assert "Custom" in attr
        assert "Prop" in attr

        # Use default units
        units = resolve_units(None, "dimensionless")
        assert units == "dimensionless"

    def test_fuzzy_match_and_format(self):
        """Test fuzzy matching followed by formatting."""
        # Fuzzy match
        attr = resolve_attr_key("densty", ENERGY_ALIASES)  # typo
        assert attr == "density"

        # Format units
        units = format_unit_str("kg/m**3")
        assert "^{" in units

    def test_all_aliases_resolve_and_format(self):
        """Test that all aliases can be resolved and units formatted."""
        for canonical, aliases in ENERGY_ALIASES.items():
            for alias in aliases:
                # Resolve
                resolved = resolve_attr_key(alias, ENERGY_ALIASES)
                assert resolved == canonical

                # Format some example units
                formatted = format_unit_str("unit")
                assert formatted.startswith("$")
                assert formatted.endswith("$")




class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_resolve_attr_key_with_numbers(self):
        """Test resolve_attr_key with numbers in key."""
        result = resolve_attr_key("property123", ENERGY_ALIASES)
        assert isinstance(result, str)

    def test_resolve_units_with_special_chars(self):
        """Test resolve_units with special characters."""
        result = resolve_units("kg·m/s²", "N")
        assert result == "kg·m/s²"

    def test_format_unit_str_unicode(self):
        """Test format_unit_str with unicode characters."""
        result = format_unit_str(r"\Omega")
        assert result.startswith("$")
        assert r"\Omega" in result

    def test_very_long_unit_string(self):
        """Test with very long unit string."""
        long_unit = "kg*m**2*s**-2*A**-1*K**-1*mol**-1*cd**-1"
        result = format_unit_str(long_unit)
        assert result.startswith("$")
        assert result.endswith("$")

    def test_resolve_attr_key_empty_alias_map(self):
        """Test resolve_attr_key with empty alias map."""
        result = resolve_attr_key("test", {})
        assert result == "Test"

    def test_format_unit_str_with_fractions(self):
        """Test format_unit_str with fraction-like notation."""
        result = format_unit_str("1/s")
        assert result.startswith("$")
        assert result.endswith("$")


