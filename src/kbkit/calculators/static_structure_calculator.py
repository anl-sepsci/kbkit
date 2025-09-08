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

    Attributes
    ----------
    T: float
        Temperature in Kelvin. Initialized to None.
    hessian: np.ndarray
        Hessian of Gibbs mixing free energy 3D array with shape ``(n_sys, n_comp, n_comp)``. Initialized to None.
    isothermal_compressibility: np.ndarray
        Isothermal compressibility 1D array with shape ``(n_sys)``. Initialzed to None.


    .. note::
        Run :func:`update_conditions` to set and update the T, hessian, and isothermal compressibility attributes for structure calculations.

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

        # validate input shapes
        if not (len(molar_volume) == len(n_electrons) == np.asarray(mol_fr).shape[1]):
            raise ValueError("Input arrays: molar_volume, n_electrons, and mol_fr must have the same length.")

        # pure component properties; make sure values are arrays
        self.molar_volume = np.asarray(
            self.Q_(molar_volume, "cm^3/mol").to("cm^3/molecule").magnitude
        )  # convert to cm^3/molecule
        self.n_electrons = np.asarray(n_electrons)
        self.mol_fr = np.asarray(mol_fr)

        # initialize thermodynamic conditions
        self.T = 0.0
        self.hessian = np.zeros((mol_fr.shape[0], mol_fr.shape[1] - 1, mol_fr.shape[1] - 1))
        self.isothermal_compressibility = np.zeros(mol_fr.shape[0])

    def update_conditions(
        self,
        T: float | None = None,
        hessian: NDArray[np.float64] | None = None,
        isothermal_compressibility: NDArray[np.float64] | None = None,
    ) -> None:
        """
        Update thermodynamic conditions for the system.

        Parameters
        ----------
        T : float
            Temperature in Kelvin. Must be strictly positive.
        hessian : np.ndarray
            Third-order tensor representing the Hessian matrix. Must be a 3D array of shape ``(n_sys, n_comp, n_comp)``.
        isothermal_compressibility : np.ndarray
            Array of isothermal compressibility values. All entries must be non-negative.
        """
        if T is not None:
            if not isinstance(T, float):
                try:
                    if isinstance(T, (list, np.ndarray)):
                        T = float(np.mean(T))
                except Exception as e:
                    raise TypeError(f"T of type({type(T)}), expected type float.") from e
            if T <= 0:
                raise ValueError("Temperature must be positive.")
            self.T = float(T)

        if hessian is not None:
            MAGIC_THREE = 3
            if hessian.ndim != MAGIC_THREE:
                raise ValueError("Hessian must be a 3D array.")
            self.hessian = np.asarray(hessian)

        if isothermal_compressibility is not None:
            self.isothermal_compressibility = np.asarray(isothermal_compressibility)

        # check that all variables are not None
        missing = [
            name
            for name, val in {
                "T": self.T,
                "hessian": self.hessian,
                "isothermal_compressibility": self.isothermal_compressibility,
            }.items()
            if val is None
        ]

        if missing:
            raise ValueError(f"Missing required condition(s): {', '.join(missing)}.")

    def summarize_conditions(self) -> str:
        """
        Return a summary of the current thermodynamic conditions.

        Returns
        -------
        str
            A formatted string containing the temperature in Kelvin,
            the shape of the Hessian tensor, and the shape of the
            isothermal compressibility array.

        Examples
        --------
        >>> obj.summarize_conditions()
        'T = 300.0 K, hessian shape = (10, 3, 3), compressibility shape = (10,)'
        """
        return f"T = {self.T} K, hessian shape = {self.hessian.shape}, compressibility shape = {self.isothermal_compressibility.shape}"

    @property
    def volume_bar(self) -> NDArray[np.float64]:
        """np.ndarray: Molar volume of the mixture."""
        return self.mol_fr @ self.molar_volume

    @property
    def n_electrons_bar(self) -> NDArray[np.float64]:
        """np.ndarray: Number of electrons in the mixture."""
        return self.mol_fr @ self.n_electrons

    @property
    def _delta_volume(self) -> NDArray[np.float64]:
        """np.ndarray: Molar volume difference."""
        return self.molar_volume[:-1] - self.molar_volume[-1]

    @property
    def _delta_n_electrons(self) -> NDArray[np.float64]:
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

    def s0_x(self) -> NDArray[np.float64]:
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
        with np.errstate(divide="ignore", invalid="ignore"):
            return self.gas_constant * self.T / self.hessian

    def s0_xp(self) -> NDArray[np.float64]:
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
            - :math:`\bar{V}` is the molar volume of mixture
        """
        v_ratio = self._delta_volume[np.newaxis, :] / self.volume_bar[:, np.newaxis]
        s0_xp_calc = -1 * self.s0_x() * v_ratio[:, :, np.newaxis]
        s0_xp_sum = np.nansum(s0_xp_calc, axis=2)
        return s0_xp_sum

    def s0_p(self) -> NDArray[np.float64]:
        r"""
        Structure factor as q :math:`\rightarrow` 0 for density-density fluctuations.

        Parameters
        ----------
        T: float
            Temperature (K) of system.
        hessian: np.ndarray
            Hessian of Gibbs mixing free energy (kJ/mol).
        isothermal_compressibility: np.ndarray
            Isothermal compressibility (1/kPa).

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
            - :math:`\bar{V}` is the molar volume of mixture
            - :math:`\kappa` is the isothermal compressibility
        """
        R_units = float(self.Q_(self.gas_constant, "kJ/mol/K").to("kPa*cm^3/molecule/K").magnitude)
        term1 = R_units * self.T * self.isothermal_compressibility / self.volume_bar
        v_ratio = self._delta_volume[np.newaxis, :] / self.volume_bar[:, np.newaxis]
        term2 = v_ratio[:, :, np.newaxis] * v_ratio[:, np.newaxis, :] * self.s0_x()
        term2_sum = np.nansum(term2, axis=tuple(range(1, term2.ndim)))
        return term1 + term2_sum

    def s0_x_e(self) -> NDArray[np.float64]:
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

        where:
            - :math:`Z_i` is the number of electrons in molecule :math:`i`
        """
        s0_x_calc = (
            self._delta_n_electrons[np.newaxis, :, np.newaxis]
            * self._delta_n_electrons[np.newaxis, np.newaxis, :]
            * self.s0_x()
        )
        return np.nansum(s0_x_calc, axis=tuple(range(1, s0_x_calc.ndim)))

    def s0_xp_e(self) -> NDArray[np.float64]:
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

        where:
            - :math:`Z_i` is the number of electrons in molecule :math:`i`
            - :math:`\bar{Z}` is the number of electrons in the mixture
        """
        s0_xp_calc = self._delta_n_electrons[np.newaxis, :] * self.s0_xp()
        return 2 * self.n_electrons_bar * np.nansum(s0_xp_calc, axis=1)

    def s0_p_e(self) -> NDArray[np.float64]:
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

        where:
            - :math:`\bar{Z}` is the number of electrons in the mixture
        """
        return self.n_electrons_bar**2 * self.s0_p()

    def s0_e(self) -> NDArray[np.float64]:
        r"""
        Structure factor of electron density as q :math:`\rightarrow` 0.

        Returns
        -------
        np.ndarray
            1D array of shape ``(n_sys)``.

        Notes
        -----
        The electron density structure factor, :math:`\hat{S}^e(0)`, is calculated from the sum of the structure factor contributions to electron density.

        .. math::
            \hat{S}^e(0) = \hat{S}^{x,e}(0) + \hat{S}^{x\rho,e}(0) + \hat{S}^{\rho,e}(0)
        """
        return self.s0_x_e() + self.s0_xp_e() + self.s0_p_e()

    def i0(self) -> NDArray[np.float64]:
        r"""
        Small angle x-ray scattering (SAXS) intensity as q :math:`\rightarrow` 0.

        Returns
        -------
        np.ndarray
            A 1D array with shape ``(n_sys)``

        Notes
        -----
        The scattering intensity at as q :math:`\rightarrow` 0, I(0), is calculated from electron density structure factor (:math:`\hat{S}^e`):

        .. math::
            I(0) = r_e^2 \rho \hat{S}^e(0)

        where:
            - :math:`r_e` is the radius of an electron in cm
        """
        return self.re**2 * (1 / self.volume_bar) * self.s0_e()

    def i0_calc(self, s0: NDArray[np.float64]) -> NDArray[np.float64]:
        r"""
        Small angle x-ray scattering (SAXS) intensity as q :math:`\rightarrow` 0, for any structure factor contribution to electron density.

        Parameters
        ----------
        s0: np.ndarray
            Electron density structure factor to use for i0 calculation.

        Returns
        -------
        np.ndarray
            A 1D array with shape ``(n_sys)``

        See Also
        --------
        see :meth:`i0` for the derivation and formula.
        """
        return self.re**2 * (1 / self.volume_bar) * s0
