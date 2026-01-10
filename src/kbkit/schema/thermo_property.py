"""Structured representation of scalar properties with units and semantic tags."""

from dataclasses import dataclass, field
from functools import cached_property, wraps
from typing import Any

from kbkit.config.unit_registry import load_unit_registry


@dataclass
class ThermoProperty:
    """
    Container for a scalar property with units and semantic annotations.

    Designed to store a value alongside its physical units and optional tags
    for classification, filtering, or metadata enrichment.

    Attributes
    ----------
    name: str
        Name of the computed property.
    value : Any
        The raw property value (e.g., float, int, or derived object).
    units : str
        Units associated with the value (e.g., "kJ/mol", "nm", "mol/L").
    """

    name: str
    value: Any
    units: str = field(default_factory=str)

    def to(self, new_units: str):
        """
        Unit conversion for property.

        Parameters
        ----------
        new_units: str
            Units for desired property

        Returns
        -------
        Any
            Value in new units.
        """
        ureg = load_unit_registry()
        Q_ = ureg.Quantity
        return Q_(self.value, self.units).to(new_units).magnitude


def register_property(name: str, units: str):
    """
    Method decorator for associating metadata and units with a ThermoProperty.

    Parameters
    ----------
    name : str
        Property name.
    units : str
        Property units.

    Returns
    -------
    Callable
        The resulting decorator produces a cached property containing a ThermoProperty instance.
    """

    def decorator(func):
        """Recieve decorated method and applies the wrapping logic."""

        @cached_property
        @wraps(func)
        def wrapper(self):
            """Create and return the ThermoProperty object for a given function."""
            return ThermoProperty(name=name, value=func(self), units=units)

        return wrapper

    return decorator
