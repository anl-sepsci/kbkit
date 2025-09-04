"""Structured representation of thermodynamic and state properties with units and semantic tags."""

from dataclasses import dataclass

from kbkit.schema.thermo_property import ThermoProperty


@dataclass
class ThermoState:
    """
    Structured container for thermodynamic and state properties.

    This dataclass aggregates all computed thermodynamic and state properties from a KBPipeline run.
    Each attribute is a `ThermoProperty` instance, providing the value, units, and metadata for a specific property.

    Attributes
    ----------
    kbis : ThermoProperty
        Kirkwood-Buff integrals for each system.
    A_inv_matrix : ThermoProperty
        Inverse of the Helmholtz stability matrix.
    A_matrix : ThermoProperty
        Helmholtz stability matrix.
    l_stability : ThermoProperty
        Stability array for the mixture.
    dmui_dxj : ThermoProperty
        Chemical potential derivatives with respect to mol fraction.
    dmui_dnj : ThermoProperty
        Chemical potential derivatives with respect to molecule count.
    dmui_dxi : ThermoProperty
        Chemical potential derivatives with respect to own mol fraction.
    isothermal_compressibility : ThermoProperty
        Isothermal compressibility of the mixture.
    hessian : ThermoProperty
        Hessian matrix of Gibbs mixing free energy.
    det_hessian : ThermoProperty
        Determinant of the Hessian matrix.
    dlngammas_dxs : ThermoProperty
        Derivatives of activity coefficients with respect to composition.
    lngammas : ThermoProperty
        Activity coefficients as a function of composition.
    ge : ThermoProperty
        Gibbs excess free energy.
    gid : ThermoProperty
        Gibbs ideal free energy.
    gm : ThermoProperty
        Gibbs mixing free energy.
    se : ThermoProperty
        Excess entropy.
    i0 : ThermoProperty
        X-ray intensity as q → 0.
    s0_e : ThermoProperty
        Electron density structure factor as q → 0.
    s0_x_e : ThermoProperty
        Concentration-concentration contribution to electron density structure factor.
    s0_xp_e : ThermoProperty
        Concentration-density contribution to electron density structure factor.
    s0_p_e : ThermoProperty
        Density-density contribution to electron density structure factor.
    s0_x : ThermoProperty
        Concentration-concentration contribution to structure factor.
    s0_xp : ThermoProperty
        Concentration-density contribution to structure factor.
    s0_p : ThermoProperty
        Density-density contribution to structure factor.
    molecules : ThermoProperty
        List of molecule names in the system.
    mol_fr : ThermoProperty
        Mole fractions for each component.
    temperature : ThermoProperty
        System temperature.
    volume : ThermoProperty
        System volume.
    molar_volume : ThermoProperty
        Molar volume of the system.
    n_electrons : ThermoProperty
        Number of electrons in the system.
    h_mix : ThermoProperty
        Enthalpy of mixing.
    volume_bar : ThermoProperty
        Average volume per molecule.
    molecule_rho : ThermoProperty
        Number density for each molecule type.
    molecule_counts : ThermoProperty
        Count of each molecule type in the system.
    """

    kbis: ThermoProperty
    A_inv_matrix: ThermoProperty
    A_matrix: ThermoProperty
    l_stability: ThermoProperty
    dmui_dxj: ThermoProperty
    dmui_dnj: ThermoProperty
    dmui_dxi: ThermoProperty
    isothermal_compressibility: ThermoProperty
    hessian: ThermoProperty
    det_hessian: ThermoProperty
    dlngammas_dxs: ThermoProperty
    lngammas: ThermoProperty
    ge: ThermoProperty
    gid: ThermoProperty
    gm: ThermoProperty
    se: ThermoProperty
    i0: ThermoProperty
    s0_e: ThermoProperty
    s0_x_e: ThermoProperty
    s0_xp_e: ThermoProperty
    s0_p_e: ThermoProperty
    s0_x: ThermoProperty
    s0_xp: ThermoProperty
    s0_p: ThermoProperty
    molecules: ThermoProperty
    mol_fr: ThermoProperty
    temperature: ThermoProperty
    volume: ThermoProperty
    molar_volume: ThermoProperty
    n_electrons: ThermoProperty
    h_mix: ThermoProperty
    volume_bar: ThermoProperty
    molecule_rho: ThermoProperty
    molecule_counts: ThermoProperty
