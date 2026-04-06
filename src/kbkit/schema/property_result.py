"""Structured representation of scalar properties with units and semantic tags."""

from dataclasses import dataclass
from typing import Any

import numpy as np

from kbkit.config.unit_registry import load_unit_registry


@dataclass
class PropertyResult:
    """
    Container for calculated thermodynamic properties with metadata.

    Parameters
    ----------
    name : str
        Name of the property (e.g., "density", "kbi", "activity_coefficient")
    value : np.ndarray
        Calculated property values.
    property_type : str, optional
        Type of property calculated.
    units : str, optional
        Units of the property (e.g., "kg/m^3", "kJ/mol").
    metadata : dict
        Additional calculation metadata (e.g., mixing rules, KBI metadata).
    """

    name: str
    value: np.ndarray
    metadata: dict[str, Any] | None = None
    property_type: str | None = None
    units: str | None = None

    def __post_init__(self):
        """Validate inputs after initialization."""
        if not isinstance(self.value, np.ndarray):
            self.value = np.asarray(self.value)
        if self.metadata is None:
            self.metadata = {}

    def to(self, units: str) -> "PropertyResult":
        """
        Convert property to desired units.

        For KBI properties, also converts KBIMetadata values.

        Parameters
        ----------
        units : str
            Target units for conversion.

        Returns
        -------
        PropertyResult
            New PropertyResult with converted values and units.
        """
        if self.units is None:
            raise ValueError(
                f"Cannot convert PropertyResult '{self.name}' without units. Original units were not specified."
            )

        if units == self.units:
            # No conversion needed, return self
            return self

        ureg = load_unit_registry()

        try:
            # Convert the main value
            quantity = ureg.Quantity(self.value, self.units)
            new_value = quantity.to(units).magnitude
        except Exception as e:
            raise ValueError(f"Cannot convert '{self.name}' from '{self.units}' to '{units}': {e}") from e

        # Return a new PropertyResult with converted values
        return PropertyResult(
            name=self.name, value=new_value, property_type=self.property_type, units=units, metadata=self.metadata
        )

    def __repr__(self) -> str:
        """String representation of PropertyResult."""
        value_str = f"shape={self.value.shape}" if self.value.ndim > 0 else f"value={self.value}"
        units_str = f", units='{self.units}'" if self.units else ""
        type_str = f", type='{self.property_type}'" if self.property_type else ""
        return f"PropertyResult(name='{self.name}', {value_str}{units_str}{type_str})"

    def __str__(self) -> str:
        """User-friendly string representation."""
        return f"{self.name}: {self.value} {self.units or '(dimensionless)'}"
