"""Structured representation of thermodynamic and state properties with units and semantic tags."""

from dataclasses import dataclass

from kbkit.schema.thermo_property import ThermoProperty


@dataclass
class ThermoState:
    """
    Structured container for thermodynamic and state properties.

    This dataclass aggregates all computed thermodynamic and state properties from a KBPipeline run.
    Each attribute is a `ThermoProperty` instance, providing the value, units, and metadata for a specific property.
    """
    # from KBThermo
    kbis: ThermoProperty
    A_inv_matrix: ThermoProperty
    A_matrix: ThermoProperty
    l_stability: ThermoProperty
    dmui_dxj: ThermoProperty
    dmui_dnj: ThermoProperty
    dmui_dxi: ThermoProperty
    hessian: ThermoProperty
    det_hessian: ThermoProperty
    dlngammas_dxs: ThermoProperty
    lngammas: ThermoProperty
    ge: ThermoProperty
    gid: ThermoProperty
    gm: ThermoProperty
    se: ThermoProperty
    s0_x: ThermoProperty
    s0_kappa: ThermoProperty
    s0_cc: ThermoProperty
    s0_nc: ThermoProperty
    s0_nn: ThermoProperty
    s0_x_e: ThermoProperty
    s0_kappa_e: ThermoProperty
    s0_cc_e: ThermoProperty
    s0_nc_e: ThermoProperty
    s0_nn_e: ThermoProperty
    s0_e: ThermoProperty
    i0_x: ThermoProperty
    i0_kappa: ThermoProperty
    i0_cc: ThermoProperty
    i0_nc: ThermoProperty
    i0_nn: ThermoProperty
    i0: ThermoProperty
    # from SystemState
    top_molecules: ThermoProperty
    salt_pairs: ThermoProperty
    unique_molecules: ThermoProperty
    total_molecules: ThermoProperty
    molecule_info: ThermoProperty
    molecule_counts: ThermoProperty
    pure_molecules: ThermoProperty
    pure_mol_fr: ThermoProperty
    electron_map: ThermoProperty
    n_electrons: ThermoProperty
    electron_bar: ThermoProperty
    mol_fr: ThermoProperty
    temperature: ThermoProperty
    volume: ThermoProperty
    molar_volume_map: ThermoProperty
    molar_volume: ThermoProperty
    enthalpy: ThermoProperty
    heat_capacity: ThermoProperty
    isothermal_compressibility: ThermoProperty
    pure_enthalpy: ThermoProperty
    ideal_enthalpy: ThermoProperty
    h_mix: ThermoProperty
    volume_bar: ThermoProperty
    volume_mix: ThermoProperty
    excess_volume: ThermoProperty


    def to_dict(self) -> dict:
        """Convert the ThermoState dataclass to a dictionary."""
        state_dict = {field.name: getattr(self, field.name) for field in self.__dataclass_fields__.values()}
        return {name: thermo.value for name, thermo in state_dict.items()}

    def get(self, property_name: str) -> ThermoProperty:
        """
        Retrieve a specific ThermoProperty by name.

        Parameters
        ----------
        property_name : str
            The name of the property to retrieve.

        Returns
        -------
        ThermoProperty
            The requested ThermoProperty instance.
        """
        if not hasattr(self, property_name):
            raise AttributeError(f"ThermoState has no attribute '{property_name}'")
        return getattr(self, property_name)
