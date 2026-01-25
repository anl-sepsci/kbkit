"""Structured representation of scalar properties with units and semantic tags."""

from dataclasses import dataclass
from typing import Any

import numpy as np

from kbkit.config.unit_registry import load_unit_registry
from kbkit.schema.kbi_metadata import KBIMetadata


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
                f"Cannot convert PropertyResult '{self.name}' without units. "
                "Original units were not specified."
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
            raise ValueError(
                f"Cannot convert '{self.name}' from '{self.units}' to '{units}': {e}"
            ) from e

        # Handle metadata conversion
        new_metadata: dict[str, Any] = {}
        if self.metadata:
            if self.name.lower() == "kbi":
                # Special handling for KBI metadata
                new_metadata = self._convert_kbi_metadata(self.metadata, units, ureg)
            else:
                # For other properties, just copy metadata
                new_metadata = self.metadata.copy() if self.metadata else {}

        # Return a new PropertyResult with converted values
        return PropertyResult(
            name=self.name,
            value=new_value,
            property_type=self.property_type,
            units=units,
            metadata=new_metadata
        )

    def _convert_kbi_metadata(self, property_meta: dict[str, Any], target_units: str, ureg) ->  dict[str, dict[str, KBIMetadata]]:
        """
        Convert KBI metadata to target units.

        Parameters
        ----------
        target_units : str
            Target units for KBI values.
        ureg : UnitRegistry
            Pint unit registry.

        Returns
        -------
        dict
            Metadata with converted KBIMetadata objects.
        """
        new_metadata: dict[str, Any] = {}

        for system_name, pair_dict in property_meta.items():
            if not isinstance(pair_dict, dict):
                # Not the expected structure, keep as-is
                new_metadata[system_name] = pair_dict
                continue

            new_metadata[system_name] = {}
            for pair_name, kbi_meta in pair_dict.items():
                if isinstance(kbi_meta, KBIMetadata):
                    # Convert KBIMetadata
                    new_metadata[system_name][pair_name] = self._convert_single_kbi_metadata(
                        kbi_meta, target_units, ureg
                    )
                else:
                    # Keep non-KBIMetadata as-is
                    new_metadata[system_name][pair_name] = kbi_meta

        return new_metadata

    def _convert_single_kbi_metadata(
        self, kbi_meta: KBIMetadata, target_units: str, ureg
    ) -> KBIMetadata:
        """
        Convert a single KBIMetadata object to target units.

        Parameters
        ----------
        kbi_meta : KBIMetadata
            Original KBIMetadata object.
        target_units : str
            Target units for KBI values.
        ureg : UnitRegistry
            Pint unit registry.

        Returns
        -------
        KBIMetadata
            New KBIMetadata with converted values.
        """
        # Convert volume-based quantities (KBI values)
        new_rkbi = ureg.Quantity(kbi_meta.rkbi, self.units).to(target_units).magnitude
        new_scaled_rkbi = ureg.Quantity(kbi_meta.scaled_rkbi, self.units).to(target_units).magnitude
        new_scaled_rkbi_fit = ureg.Quantity(kbi_meta.scaled_rkbi_fit, self.units).to(target_units).magnitude
        new_scaled_rkbi_est = ureg.Quantity(kbi_meta.scaled_rkbi_est, self.units).to(target_units).magnitude
        new_kbi_limit = ureg.Quantity(kbi_meta.kbi_limit, self.units).to(target_units).magnitude

        # Create new KBIMetadata with converted values
        return KBIMetadata(
            mols=kbi_meta.mols,
            r=kbi_meta.r,  # Distance units (nm) - unchanged
            g=kbi_meta.g,  # Dimensionless - unchanged
            rkbi=new_rkbi,
            scaled_rkbi=new_scaled_rkbi,
            r_fit=kbi_meta.r_fit,  # Distance units - unchanged
            scaled_rkbi_fit=new_scaled_rkbi_fit,
            scaled_rkbi_est=new_scaled_rkbi_est,
            kbi_limit=new_kbi_limit,
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
