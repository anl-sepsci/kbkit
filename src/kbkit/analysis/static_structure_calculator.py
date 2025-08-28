"""
Calculator for static structure.

Provides method to compute structure factors and x-ray scattering intensity.
"""

import numpy as np
from numpy.typing import NDArray

from kbkit.config.unit_registry import load_unit_registry


class StaticStructureCalculator:
    """
    Computes static structure properties for molecular systems using thermodynamic properties.

    Parameters
    ----------
    molar_volume: np.ndarray
        Molar volume of pure components in cm^3/mol.
    n_electrons: np.ndarray
        Number of electrons in pure components.
    mol_fr: np.ndarray
        Mol fraction array.
    """

    def __init__(
        self,
        molar_volume: NDArray[np.float64],  # units = cm^3/mol
        n_electrons: NDArray[np.float64],
        mol_fr: NDArray[np.float64],
    ) -> None:
        # add unit registry
        self.ureg = load_unit_registry()
        self.Q_ = self.ureg.Quantity

        # pure component properties; make sure values are arrays
        self.molar_volume = np.asarray(
            self.Q_(molar_volume, "cm^3/mol").to("cm^3/molecule").magnitude
        )  # convert to cm^3/molecule
        self.n_electrons = np.asarray(n_electrons)
        self.mol_fr = np.asarray(mol_fr)

    @property
    def volume_bar(self) -> NDArray[np.float64]:
        """np.ndarray: Linear combination of molar volume."""
        return self.mol_fr @ self.molar_volume

    @property
    def n_electrons_bar(self) -> NDArray[np.float64]:
        """np.ndarray: Linear combination of electron numbers."""
        return self.mol_fr @ self.n_electrons

    @property
    def delta_volume(self) -> NDArray[np.float64]:
        """np.ndarray: Molar volume difference."""
        return self.molar_volume[:-1] - self.molar_volume[-1]

    @property
    def delta_n_electrons(self) -> NDArray[np.float64]:
        """np.ndarray: Electron number difference."""
        return self.n_electrons[:-1] - self.n_electrons[-1]

    @property
    def re(self) -> float:
        """float: Electron radius (cm)."""
        re_val = self.ureg("re").to("cm").magnitude
        return float(re_val)

    @property
    def gas_constant(self) -> float:
        """float: Gas constant (kJ/mol/K)."""
        R_val = self.ureg("R").to("kJ/mol/K").magnitude
        return float(R_val)

    def s0_x(self, T: float, hessian: NDArray[np.float64]) -> NDArray[np.float64]:
        r"""
        Structure factor as q :math:`\rightarrow` 0 for composition-composition fluctuations.

        Parameters
        ----------
        T: float
            Temperature (K) of system.
        hessian: np.ndarray
            Hessian of Gibbs mixing free energy (kJ/mol).

        Returns
        -------
        np.ndarray
            A 3D matrix of shape ``(n_sys, n_comp-1, n_comp-1)``

        Notes
        -----
        The structure factor, :math:`\hat{S}_{ij}^{x}(0)`, is calculated as follows:

        .. math::
            \hat{S}_{ij}^{x}(0) = RT H_{ij}^{-1}

        where:
            - :math:`H_{ij}` is the Hessian of molecules :math:`i,j`
        """
        return self.gas_constant * T / np.asarray(hessian)

    def s0_xp(self, T: float, hessian: NDArray[np.float64]) -> NDArray[np.float64]:
        r"""
        Structure factor as q :math:`\rightarrow` 0 for composition-density fluctuations.

        Parameters
        ----------
        T: float
            Temperature (K) of system.
        hessian: np.ndarray
            Hessian of Gibbs mixing free energy (kJ/mol).

        Returns
        -------
        np.ndarray
            2D array of shape ``(n_sys, n_comp-1)``.

        Notes
        -----
        The structure factor, :math:`\hat{S}_{i}^{x\rho}(0)`, is calculated as follows:

        .. math::
            \hat{S}_{i}^{x\rho}(0) = - \sum_{j=1}^{n-1} \left(\frac{V_j - V_n}{\bar{V}}\right) \hat{S}_{ij}^{x}(0)

        where:
            - :math:`V_j` is the molar volume of molecule :math:`j`
        """
        v_ratio = self.delta_volume[np.newaxis, :] / self.volume_bar[:, np.newaxis]
        s0_xp_calc = self.s0_x(T, hessian) * v_ratio
        s0_xp_sum = np.nansum(s0_xp_calc, axis=2)
        return s0_xp_sum

    def s0_p(
        self, T: float, hessian: NDArray[np.float64], isothermal_compressability: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        r"""
        Structure factor as q :math:`\rightarrow` 0 for density-density fluctuations.

        Parameters
        ----------
        T: float
            Temperature (K) of system.
        hessian: np.ndarray
            Hessian of Gibbs mixing free energy (kJ/mol).
        isothermal_compressability: np.ndarray
            Isothermal compressability (1/kPa).

        Returns
        -------
        np.ndarray
            2D array of shape ``(n_sys, n_comp-1)``.

        Notes
        -----
        The structure factor, :math:`\hat{S}^{\rho}(0)`, is calculated as follows:

        .. math::
            \hat{S}^{\rho}(0) = \frac{RT \kappa}{\bar{V}} + \sum_{i=1}^{n-1} \sum_{j=1}^{n-1} \left(\frac{V_i - V_n}{\bar{V}}\right) \left(\frac{V_j - V_n}{\bar{V}}\right) \hat{S}_{ij}^{x}(0)

        where:
            - :math:`V_i` is the molar volume of molecule :math:`i`
            - :math:`\kappa` is the isothermal compressability
        """
        R_units = float(self.Q_(self.gas_constant, "kJ/mol/K").to("kPa*cm^3/molecule/K").magnitude)
        term1 = R_units * T * isothermal_compressability / self.volume_bar
        v_ratio = self.delta_volume[np.newaxis, :] / self.volume_bar[:, np.newaxis]
        term2 = v_ratio[:, :, np.newaxis] * v_ratio[:, np.newaxis, :] * self.s0_x(T, hessian)
        term2_sum = np.nansum(term2, axis=tuple(range(1, term2.ndim)))
        return term1 + term2_sum

    def s0_x_e(self, T: float, hessian: NDArray[np.float64]) -> NDArray[np.float64]:
        r"""
        Contribution of concentration-concentration structure factor to electron density structure factor.

        Parameters
        ----------
        T: float
            Temperature (K) of system.
        hessian: np.ndarray
            Hessian of Gibbs mixing free energy (kJ/mol).

        Returns
        -------
        np.ndarray
            1D array of shape ``(n_sys)``.

        Notes
        -----
        The contribution of concentration-concentration structure factor to electron density, :math:`\hat{S}^{x,e}(0)`, is calculated as follows:

        .. math::
            \hat{S}^{x,e}(0) = \sum_{i=1}^{n-1} \sum_{j=1}^{n-1} \left(Z_i - Z_n\right) \left(Z_j - Z_n\right) \hat{S}_{ij}^{x}(0)
        """
        s0_x_calc = (
            self.delta_n_electrons[np.newaxis, :, np.newaxis]
            * self.delta_n_electrons[np.newaxis, np.newaxis, :]
            * self.s0_x(T, hessian)
        )
        return np.nansum(s0_x_calc, axis=tuple(range(1, s0_x_calc.ndim)))

    def s0_xp_e(self, T: float, hessian: NDArray[np.float64]) -> NDArray[np.float64]:
        r"""
        Contribution of concentration-density structure factor to electron density structure factor.

        Parameters
        ----------
        T: float
            Temperature (K) of system.
        hessian: np.ndarray
            Hessian of Gibbs mixing free energy (kJ/mol).

        Returns
        -------
        np.ndarray
            1D array of shape ``(n_sys)``.

        Notes
        -----
        The contribution of concentration-density structure factor to electron density, :math:`\hat{S}^{x\rho,e}(0)`, is calculated as follows:

        .. math::
            \hat{S}^{x\rho,e}(0) = 2 \bar{Z} \sum_{i=1}^{n-1} \left(Z_i - Z_n\right) \hat{S}_{i}^{x\rho}(0)
        """
        s0_xp_calc = self.delta_n_electrons[np.newaxis, :] * self.s0_xp(T, hessian)
        return 2 * self.n_electrons_bar * np.nansum(s0_xp_calc, axis=1)

    def s0_p_e(
        self, T: float, hessian: NDArray[np.float64], isothermal_compressability: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        r"""
        Contribution of density-density structure factor to electron density structure factor.

        Parameters
        ----------
        T: float
            Temperature (K) of system.
        hessian: np.ndarray
            Hessian of Gibbs mixing free energy (kJ/mol).

        Returns
        -------
        np.ndarray
            1D array of shape ``(n_sys)``.

        Notes
        -----
        The contribution of density-density structure factor to electron density, :math:`\hat{S}^{\rho,e}(0)`, is calculated as follows:

        .. math::
            \hat{S}^{\rho,e}(0) = \bar{Z}^2 \hat{S}^{\rho}(0)
        """
        return self.n_electrons_bar**2 * self.s0_p(T, hessian, isothermal_compressability)

    def s0_e(
        self, T: float, hessian: NDArray[np.float64], isothermal_compressability: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        r"""
        Structure factor of electron density as q :math:`\rightarrow` 0.

        Returns
        -------
        np.ndarray
            1D array of shape ``(n_sys)``.

        Notes
        -----
        The electron density structure factor (:math:`\hat{S}^e(0)`), is calculated from the sum of the structure factor contributions to electron density.

        .. math::
            \hat{S}^e(0) = \hat{S}^{x,e}(0) + \hat{S}^{x\rho,e}(0) + \hat{S}^{\rho,e}(0)
        """
        return self.s0_x_e(T, hessian) + self.s0_xp_e(T, hessian) + self.s0_p_e(T, hessian, isothermal_compressability)

    def i0(
        self, T: float, hessian: NDArray[np.float64], isothermal_compressability: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        r"""
        Small angle x-ray scattering (SAXS) intensity as q :math:`\rightarrow` 0.

        Returns
        -------
        np.ndarray
            A 1D array with shape ``(n_sys)``

        Notes
        -----
        The scattering intensity at as q :math:`\rightarrow` 0 (I(0)), is calculated from electron density structure factor (:math:`\hat{S}^e`):

        .. math::
            I(0) = r_e^2 \rho \hat{S}^e
        """
        return self.re**2 * (1 / self.volume_bar) * self.s0_e(T, hessian, isothermal_compressability)
