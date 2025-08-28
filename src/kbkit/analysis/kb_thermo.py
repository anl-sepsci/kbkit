"""Constructs thermodynamic property matrices from KBIs across multiple systems."""

import warnings
from functools import partial
from itertools import product
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import cumulative_trapezoid

from kbkit.analysis.system_state import SystemState
from kbkit.calculators.static_structure_calculator import StaticStructureCalculator
from kbkit.schema.property_cache import PropertyCache

# Suppress only the specific RuntimeWarning from numpy.linalg
warnings.filterwarnings("ignore", category=RuntimeWarning, module="numpy.linalg")


class KBThermo:
    """Apply Kirkwood-Buff (KB) theory to calculate thermodynamic properties from KBI matrix.

    This class inherits system properties from :class:`KBICalculator` and uses them for the calculation of thermodynamic properties.

    Parameters
    ----------
    state: SystemState
        SystemState at a constant temperature.
    kbi_matrix: np.ndarray
        Matrix of KBI values for each pairwise interaction.

    Attributes
    ----------
    structure_calculator: StaticStructureCalculator
        Calculator for calculating static structure.
    kbis: np.ndarray
        Matrix of KBI values.
    state: SystemState
        Initialized SystemState object.
    """

    def __init__(self, state: SystemState, kbi_matrix: NDArray[np.float64]) -> None:
        # initialize SystemAnalyzer with config.
        self.state = state

        # initialize static structure calculator
        self.structure_calculator = StaticStructureCalculator(
            molar_volume=self.state.molar_volume("cm^3/mol"),
            n_electrons=self.state.n_electrons,
            mol_fr=self.state.pure_mol_fr,
        )

        # initialize cache to store units
        self._cache: dict[str, PropertyCache] = {}

        # add kbi matrix to cache
        self.kbis = kbi_matrix
        self._populate_cache("kbis", self.kbis, "nm^3/molecule")

    def _populate_cache(self, name: str, value: Any, units: str = "", tags: list | None = None) -> None:
        """Update the value of property in cache."""
        self._cache[name] = PropertyCache(value=value, units=units, tags=tags if tags else [])

    def kd(self) -> NDArray[np.float64]:
        """Kronecker delta between pairs of unique molecules."""
        return np.eye(self.state.n_comp)

    def b_matrix(self) -> NDArray[np.float64]:
        r"""
        Construct a symmetric matrix **B** for each system based on the number densities and KBIs.

        Returns
        -------
        np.ndarray
            A 3D matrix with shape ``(n_sys, n_comp, n_comp)``,
            where ``n_sys`` is the number of systems and ``n_comp`` is the number of unique components.

        Notes
        -----
        Elements of **B** are calculated for molecules :math:`i,j`, using the formula:

        .. math::
            B_{ij} = \rho_{ij} G_{ij} + \rho_i \delta_{i,j}

        where:
            - :math:`\rho_{ij}` is the pairwise number density of molecules in each system.
            - :math:`G_{ij}` is the KBI for the pair of molecules.
            - :math:`\rho_i` is the number density of molecule :math:`i`.
            - :math:`\delta_{i,j}` is the Kronecker delta for molecules :math:`i,j`.
        """
        if "_b_matrix" not in self.__dict__:
            self._b_matrix = (
                self.state.rho_ij(units="molecule/nm^3") * self.kbis
                + self.state.molecule_rho(units="molecule/nm^3")[:, :, np.newaxis] * self.kd()[np.newaxis, :, :]
            )
        return self._b_matrix

    @property
    def _b_inv(self) -> NDArray[np.float64]:
        """np.ndarray: Inverse of the B matrix."""
        # Create an empty array to store the inverse matrices
        b_mat = self.b_matrix()
        inverses = np.zeros_like(b_mat)
        # Iterate and compute the inverse for each 2x2 matrix
        for i in range(b_mat.shape[0]):
            inverses[i] = np.linalg.inv(b_mat[i, :, :])
        return inverses

    @property
    def _b_det(self) -> NDArray[np.float64]:
        """np.ndarray: Determinant of the B matrix."""
        return np.asarray(np.linalg.det(self.b_matrix()))

    def b_cofactors(self) -> NDArray[np.float64]:
        r"""
        Get the cofactors of **B** for each system.

        Returns
        -------
        np.ndarray
            A 3D matrix representing the cofactors of **B** with shape ``(n_sys, n_comp, n_comp)``,

        Notes
        -----
        The cofactors of **B**, :math:`Cof(\mathbf{B})`, are calculated as:

        .. math::
            Cof(\mathbf{B}) = |\mathbf{B}| \cdot \mathbf{B}^{-1}

        where:
            - :math:`|\mathbf{B}|` is the determinant of **B**
            - :math:`\mathbf{B}^{-1}` is the inverse of **B**
        """
        if "_b_cofactors" not in self.__dict__:
            self._b_cofactors = self._b_det[:, np.newaxis, np.newaxis] * self._b_inv
        return self._b_cofactors

    def a_matrix(self) -> NDArray[np.float64]:
        r"""
        Construct a symmetric matrix **A** for each system from compositions and **G**.

        Returns
        -------
        np.ndarray
            A 3D matrix with shape ``(n_sys, n_comp, n_comp)``,
            where ``n_sys`` is the number of systems and ``n_comp`` is the number of unique components.

        Notes
        -----
        Elements of **A** are calculated for molecules :math:`i,j`, using the formula:

        .. math::
            A_{ij} = \rho x_i x_j G_{ij} + x_i \delta_{i,j}

        where:
            - :math:`\rho` is the average mixture density.
            - :math:`G_{ij}` is the KBI for the pair of molecules.
            - :math:`x_i` is the mol fraction of molecule :math:`i`.
            - :math:`\delta_{i,j}` is the Kronecker delta for molecules :math:`i,j`.
        """
        if "_a_matrix" not in self.__dict__:
            self._a_matrix = (1 / self.state.volume_bar(units="nm^3/molecule"))[
                :, np.newaxis, np.newaxis
            ] * self.state.mol_fr[:, :, np.newaxis] * self.state.mol_fr[
                :, np.newaxis, :
            ] * self.kbis + self.state.mol_fr[:, :, np.newaxis] * self.kd()[np.newaxis, :, :]
        return self._a_matrix

    @property
    def gas_constant(self) -> float:
        """float: Gas constant in kJ/mol/K."""
        return float(self.state.ureg("R").to("kJ/mol/K").magnitude)

    def isothermal_compressability(self) -> NDArray[np.float64]:
        r"""
        Calculate the isothermal compressability, :math:`\kappa`, for each system.

        Parameters
        ----------
        units : str
            Units of energy to report values in. Default is 'kJ/mol'.

        Returns
        -------
        np.ndarray
            Isothermal compressability values for each system, with shape ``(n_sys)``

        Notes
        -----
        Isothermal compressability (:math:`\kappa`) is calculated by:

        .. math::
            \kappa RT = \sum_{j=1}^n V_j A_{ij}^{-1}

        where:
            - :math:`V_j` is the molar volume of molecule :math:`j`
            - :math:`A_{ij}^{-1}` is the inverse of **A** for molecules :math:`i,j`

        """
        if "_isothermal_compressability" not in self.__dict__:
            kT = (1 / (self.gas_constant * self.state.temperature())) * (
                self.state.molar_volume()[np.newaxis, :] / self.a_matrix()[:, 0, :]
            ).sum(axis=1)
            # convert units
            kT_converted = self.state.Q_(kT, units="mol/kJ * nm^3/molecule").to("1/kPa").magnitude
            # check array
            self._isothermal_compressability = np.asarray(kT_converted)
            self._populate_cache("isothermal_compressability", kT_converted, "1/kPa")
        return self._isothermal_compressability

    def dmu_dn(self) -> NDArray[np.float64]:
        r"""
        Compute the derivative of the chemical potential of molecule :math:`i` with respect to the number of molecules of molecule :math:`j`.

        Parameters
        ----------
        units : str
            Units of energy to report values in. Default is 'kJ/mol'.

        Returns
        -------
        np.ndarray
            A 3D matrix of shape ``(n_sys, n_comp, n_comp)``

        Notes
        -----
        Derivative of chemical potential with respect to molecule number (:math:`\frac{\partial \mu_i}{\partial n_j}`) is calculated as follows:

        .. math::
           \frac{\partial \mu_i}{\partial n_j} = \frac{k_bT}{\left<V\right> |\mathbf{B}|}\left(\frac{\sum_{a=1}^n\sum_{b=1}^n \rho_a\rho_b\left|B^{ij}B^{ab}-B^{ai}B^{bj}\right|}{\sum_{a=1}^n\sum_{b=1}^n \rho_a\rho_b B^{ab}}\right)

        where:
            - :math:`\mu_i` is the chemical potential of molecule :math:`i`
            - :math:`n_j` is the molecule number of molecule :math:`j`
            - :math:`k_b` is the Boltmann constant
            - :math:`\left<V\right>` is the ensemble average box volume
            - :math:`B^{ij}` is the element of :math:`Cof(\mathbf{B})` (the cofactors of **B**) for molecules :math:`i,j`

        """
        if "_dmu_dn_mat" not in self.__dict__:
            # get cofactors x number density
            cofactors_rho = self.b_cofactors() * self.state.rho_ij(units="molecule/nm^3")

            # get denominator of matrix calculation
            b_lower = cofactors_rho.sum(axis=tuple(range(1, cofactors_rho.ndim)))  # sum over dimensions 1:end

            # get numerator of matrix calculation
            B_prod = np.empty(
                (
                    self.state.n_sys,
                    self.state.n_comp,
                    self.state.n_comp,
                    self.state.n_comp,
                    self.state.n_comp,
                )
            )
            for a, b, i, j in product(range(self.state.n_comp), repeat=4):
                B_prod[:, a, b, i, j] = self.state.rho_ij(units="molecule/nm^3")[:, i, j] * (
                    self.b_cofactors()[:, a, b] * self.b_cofactors()[:, i, j]
                    - self.b_cofactors()[:, i, a] * self.b_cofactors()[:, j, b]
                )
            b_upper = B_prod.sum(axis=tuple(range(3, B_prod.ndim)))

            # get chemical potential with respect to mol number in target units
            b_frac = b_upper / b_lower[:, np.newaxis, np.newaxis]
            dmu_dn_mat = (
                self.gas_constant
                * self.state.temperature()[:, np.newaxis, np.newaxis]
                * b_frac
                / (self.state.volume() * self._b_det)[:, np.newaxis, np.newaxis]
            )
            self._dmu_dn_mat = np.asarray(dmu_dn_mat)
            self._populate_cache("dmu_dn", self._dmu_dn_mat, "kJ/mol")
        return self._dmu_dn_mat

    def _matrix_setup(self, matrix: np.ndarray) -> NDArray[np.float64]:
        """Set up matrices for multicomponent analysis."""
        n = self.state.n_comp - 1
        mat_ij = matrix[:, :n, :n]
        mat_in = matrix[:, :n, n][:, :, np.newaxis]
        mat_jn = matrix[:, n, :n][:, np.newaxis, :]
        mat_nn = matrix[:, n, n][:, np.newaxis, np.newaxis]
        return np.asarray(mat_ij - mat_in - mat_jn + mat_nn)

    def hessian(self) -> NDArray[np.float64]:
        r"""
        Hessian of Gibbs mixing free energy for molecules :math:`i,j`.

        Parameters
        ----------
        units: str
            Units of energy to report values in. Default is 'kJ/mol'.

        Returns
        -------
        np.ndarray
            A 3D matrix of shape ``(n_sys, n_comp-1, n_comp-1)``

        Notes
        -----
        Hessian matrix, **H**, with elements for molecules :math:`i,j` is calculated as follows:

        .. math::
            H_{ij} = M_{ij} - M_{in} - M_{jn} + M_{nn}

        .. math::
            M_{ij} = \frac{RT \Delta_{ij}^{-1}}{\rho x_i x_j}

        .. math::
            \Delta_{ij} = \frac{\delta_{ij}}{\rho x_i} + \frac{1}{\rho x_n} + G_{ij} - G_{in} - G_{jn} + G_{nn}

        where:
            - **H** is the Hessian matrix
            - **G** is the KBI matrix
            - :math:`x_i` is mol fraction of molecule :math:`i`
            - :math:`\rho` is the density of each system

        """
        if "_H_ij" not in self.__dict__:
            # difference between ij interactions with each other and last component
            delta_G = self._matrix_setup(self.kbis)
            mol_fraction = self.state.mol_fr.copy()
            mol_fraction[mol_fraction == 0] = np.nan

            # get Delta matrix for Hessian calc
            Delta_ij = (
                self.kd()[np.newaxis, :]
                * self.state.volume_bar()[:, np.newaxis, np.newaxis]
                / mol_fraction[:, np.newaxis]
                + (self.state.volume_bar() / (mol_fraction[:, self.state.n_comp - 1]))[:, np.newaxis, np.newaxis]
                + delta_G
            )

            # Create an empty array to store the inverse matrices
            Delta_ij_inv = np.zeros_like(Delta_ij)
            # Iterate and compute the inverse for each N x N matrix
            for i in range(Delta_ij.shape[0]):
                Delta_ij_inv[i] = np.linalg.inv(Delta_ij[i, :, :])

            # get M matrix for hessian calculation
            M_ij = (
                Delta_ij_inv
                * self.gas_constant
                * self.state.temperature()[:, np.newaxis, np.newaxis]
                * self.state.volume_bar()[:, np.newaxis, np.newaxis]
                / (mol_fraction[:, :, np.newaxis] * mol_fraction[:, np.newaxis, :])
            )

            self._H_ij = self._matrix_setup(M_ij)
            self._populate_cache("hessian", self._H_ij, "kJ/mol")
        return self._H_ij

    def det_hessian(self) -> NDArray[np.float64]:
        r"""
        Compute the determinant, :math:`|\mathbf{H}|`, of the Hessian matrix.

        Parameters
        ----------
        units: str
            Units of energy to report values in. Default is 'kJ/mol'.

        Returns
        -------
        np.ndarray
            A 1D array of shape ``(n_sys)``
        """
        det_h = np.asarray(np.linalg.det(self.hessian()))
        self._populate_cache("det_hessian", det_h, "kJ/mol")
        return det_h

    def i0(self) -> NDArray[np.float64]:
        r"""
        Compute the X-ray scattering intensity as q :math:`\rightarrow` 0.

        Returns
        -------
        np.ndarray
            A 1D array of shape ``(n_sys)``
        """
        T_avg = float(self.state.temperature().mean())
        i0_calc = self.structure_calculator.i0(
            T=T_avg,
            hessian=self.hessian(),
            isothermal_compressability=self.isothermal_compressability(),
        )
        self._populate_cache("i0", i0_calc, "1/cm")
        return i0_calc

    def s0_e(self) -> NDArray[np.float64]:
        r"""
        Compute the structure factor of electron density as q :math:`\rightarrow` 0.

        Returns
        -------
        np.ndarray
            A 1D array of shape ``(n_sys)``
        """
        T_avg = float(self.state.temperature().mean())
        s0_calc = self.structure_calculator.s0_e(
            T=T_avg,
            hessian=self.hessian(),
            isothermal_compressability=self.isothermal_compressability(),
        )
        self._populate_cache("s0_e", s0_calc, "")
        return s0_calc

    def s0_x_e(self) -> NDArray[np.float64]:
        r"""
        Compute the contribution from concentration-concentration fluctuations to structure factor of electron density as q :math:`\rightarrow` 0.

        Returns
        -------
        np.ndarray
            A 1D array of shape ``(n_sys)``
        """
        T_avg = float(self.state.temperature().mean())
        s0_x_e_calc = self.structure_calculator.s0_x_e(T=T_avg, hessian=self.hessian())
        self._populate_cache("s0_x_e", s0_x_e_calc, "")
        return s0_x_e_calc

    def s0_xp_e(self) -> NDArray[np.float64]:
        r"""
        Compute the contribution from concentration-density fluctuations to structure factor of electron density as q :math:`\rightarrow` 0.

        Returns
        -------
        np.ndarray
            A 1D array of shape ``(n_sys)``
        """
        T_avg = float(self.state.temperature().mean())
        s0_xp_e_calc = self.structure_calculator.s0_xp_e(T=T_avg, hessian=self.hessian())
        self._populate_cache("s0_xp_e", s0_xp_e_calc, "")
        return s0_xp_e_calc

    def s0_p_e(self) -> NDArray[np.float64]:
        r"""
        Compute the contribution from density-density fluctuations to structure factor of electron density as q :math:`\rightarrow` 0.

        Returns
        -------
        np.ndarray
            A 1D array of shape ``(n_sys)``
        """
        T_avg = float(self.state.temperature().mean())
        s0_p_e_calc = self.structure_calculator.s0_p_e(
            T=T_avg,
            hessian=self.hessian(),
            isothermal_compressability=self.isothermal_compressability(),
        )
        self._populate_cache("s0_p_e", s0_p_e_calc, "")
        return s0_p_e_calc

    def dmu_dxs(self) -> NDArray[np.float64]:
        r"""
        Compute the derivative of the chemical potential of molecule :math:`i` with respect to mol fraction of molecule :math:`j`.

        Parameters
        ----------
        units : str
            Units of energy to report values in. Default is 'kJ/mol'.

        Returns
        -------
        np.ndarray
            A 3D matrix of shape ``(n_sys, n_comp, n_comp)``

        Notes
        -----
        Derivative of chemical potential with respect to mol fraction (:math:`\frac{\partial \mu_i}{\partial x_j}`) is calculated as:

        .. math::
        \frac{\partial \mu_i}{\partial x_j} = n_T \left( \frac{\partial \mu_i}{\partial n_j} - \frac{\partial \mu_i}{\partial n_n} \right)

        where:
            - :math:`\mu_i`: chemical potential of molecule :math:`i`
            - :math:`n_j`: molecule number of molecule :math:`j`
            - :math:`x_j`: mol fraction of molecule :math:`j`
            - :math:`n_T`: total number of molecules in system
        """
        if "_dmu_dxs" not in self.__dict__:
            n = self.state.n_comp - 1
            total_mol = self.state.total_molecules[:, np.newaxis, np.newaxis]
            dmu_dn = self.dmu_dn()
            mol_fraction = self.state.mol_fr.copy()
            mol_fraction[mol_fraction == 0] = np.nan

            dmu_dxs = total_mol * (dmu_dn[:, :n, :n] - dmu_dn[:, :n, -1][:, :, np.newaxis])
            dmui_dxi = np.full_like(mol_fraction, np.nan)
            dmui_dxi[:, :-1] = np.diagonal(dmu_dxs, axis1=1, axis2=2)

            sum_xi_dmui = (mol_fraction[:, :-1] * dmui_dxi[:, :-1]).sum(axis=1)
            dmui_dxi[:, -1] = sum_xi_dmui / mol_fraction[:, -1]

            self._dmu_dxs = dmui_dxi
            self._populate_cache("dmu_dxs", self._dmu_dxs, "kJ/mol")

        return self._dmu_dxs

    def dlngammas_dxs(self) -> NDArray[np.float64]:
        r"""
        Compute the derivative of natural logarithm of the activity coefficient of molecule :math:`i` with respect to its mol fraction.

        Returns
        -------
        np.ndarray
            A 3D matrix with shape ``(n_sys, n_comp, n_comp)``

        Notes
        -----
        Activity coefficient derivatives, :math:`\frac{\partial \gamma_i}{\partial x_i}` are calculated as follows:

        .. math::
            \frac{\partial \ln{\gamma_i}}{\partial x_i} = \frac{1}{k_b T}\left(\frac{\partial \mu_i}{\partial x_i}\right) - \frac{1}{x_i}

        where:
            - :math:`\mu_i` is the chemical potential of molecule :math:`i`
            - :math:`\gamma_i` is the activity coefficient of molecule :math:`i`
            - :math:`x_i` is the mol fraction of molecule :math:`i`
            - :math:`k_b` is the Boltzmann constant
        """
        if "_dlngammas_dxs" not in self.__dict__:
            # Avoid ZeroDivisionError by replacing zeros with NaN
            mol_fraction = self.state.mol_fr.copy()
            mol_fraction[mol_fraction == 0] = np.nan

            # Compute derivative of ln(gamma) with respect to composition
            temperature = self.state.temperature()
            dmu_dxs = self.dmu_dxs()
            factor = 1 / (self.gas_constant * temperature)
            self._dlngammas_dxs = np.asarray(factor[:, np.newaxis] * dmu_dxs - 1 / mol_fraction)
            self._populate_cache("dlngammas_dxs", self._dlngammas_dxs)

        return self._dlngammas_dxs

    def _get_ref_state_dict(self, mol: str) -> dict[str, object]:
        """Get reference state parameters for each molecule."""
        # get max mol fr at each composition
        z0 = self.state.mol_fr.copy()
        z0[np.isnan(z0)] = 0
        comp_max = z0.max(axis=1)
        # get mol index
        i = self.state._get_mol_idx(mol, self.state.unique_molecules)
        # get mask for max mol frac at each composition
        is_max = z0[:, i] == comp_max

        # create dict for ref. state values
        # if mol is max at any composition; it cannot be a 'solute'
        if np.any(is_max):
            ref_state_dict = {
                "ref_state": "pure_component",
                "x_initial": 1.0,
                "sorted_idx_val": -1,
                "weight_fn": partial(self._weight_fn, exp_mult=1),
            }
        # if solute, use inf. dil. ref state
        else:
            ref_state_dict = {
                "ref_state": "inf_dilution",
                "x_initial": 0.0,
                "sorted_idx_val": 1,
                "weight_fn": partial(self._weight_fn, exp_mult=-1),
            }
        return ref_state_dict

    def _weight_fn(self, x: NDArray[np.float64], exp_mult: float) -> NDArray[np.float64]:
        try:
            return 100 ** (exp_mult * np.log10(x))
        except ValueError as ve:
            raise ValueError(f"Cannot take log of negative value. Details: {ve}.") from ve

    def _x_initial(self, mol: str) -> float:
        value = self._get_ref_state_dict(mol)["x_initial"]
        if isinstance(value, (float, int)):
            return float(value)
        raise TypeError(f"x_initial value must be numeric, got {type(value)}")

    def _sort_idx_val(self, mol: str) -> int:
        value = self._get_ref_state_dict(mol)["sorted_idx_val"]
        if isinstance(value, int):
            return value
        raise TypeError(f"sorted_idx_val must be int, got {type(value)}")

    def _weights(self, mol: str, x: NDArray[np.float64]) -> NDArray[np.float64]:
        w = self._get_ref_state_dict(mol)["weight_fn"]
        if callable(w):
            return w(x)  # type: ignore
        raise TypeError(f"Expected a callable for weight_fn, got {type(w)}")

    def lngammas(self, integration_type: str, polynomial_degree: int = 5) -> NDArray[np.float64]:
        r"""
        Integrate the derivative of activity coefficients.

        Parameters
        ----------
        integration_type: str
            This determines how the integration will be performed. Options include: numerical, polynomial.
        polynomial_degree: int
            For the '`polynomial`' integration, this specifies the degree of polynomial to fit the derivatives to.

        Returns
        -------
        np.ndarray
            A 2D array with shape ``(n_sys, n_comp)``

        Notes
        -----
        Numerical integration of activity coefficient derivatives occurs through:

        .. math::
            \ln{\gamma_i}(x_i) = \int_{a_0}^{x_i} \left(\frac{\partial \ln{\gamma_i}}{\partial x_i}\right) dx_i \approx \sum_{a=a_0}^{x_i} \frac{\Delta x}{2} \left[\left(\frac{\partial \ln{\gamma_i}}{\partial x_i}\right)_{a} + \left(\frac{\partial \ln{\gamma_i}}{\partial x_i}\right)_{a \pm \Delta x}\right]

        where:
            - :math:`\gamma_i` is the activity coefficient of molecule :math:`i`
            - :math:`x_i` is the mol fraction of molecule :math:`i`
            - :math:`\Delta x` is the step size in :math:`x` between points

        .. note::
            The integral is approximated by a summation using the trapezoidal rule, where the upper limit of summation is :math:`x_i` and the initial condition (or reference state) is :math:`a_0`. Note that the term :math:`a \pm \Delta x` behaves differently based on the value of :math:`a_0`: if :math:`a_0 = 1` (pure component reference state), it becomes :math:`a - \Delta x`, and if :math:`a_0 = 0` (infinite dilution reference state), it becomes :math:`a + \Delta x`.


        Analytical integration of activity coefficient derivatives thorough polynomial fitting occurs by fitting an n-order polynomial function to :math:`\frac{\partial \ln{\gamma_i}}{\partial x_i}`.

        .. note::
            This method takes a set of mole fractions (`xi`) and the corresponding derivatives of :math:`\ln{\gamma}`, fits a polynomial of a specified degree to the derivative data, integrates the polynomial to reconstruct :math:`\ln{\gamma}`, and evaluates :math:`\ln{\gamma}` at the given mol fractions. The integration constant is chosen such that :math:`\ln{\gamma}` obeys boundary conditions of reference state.

        """
        integration_type = integration_type.lower()
        dlng_dxs = self.dlngammas_dxs()  # avoid repeated calls

        ln_gammas = np.full_like(self.state.mol_fr, fill_value=np.nan)
        for i, mol in enumerate(self.state.unique_molecules):
            # get x & dlng for molecule
            xi0 = self.state.mol_fr[:, i]
            dlng0 = dlng_dxs[:, i]
            lng_i = np.full(len(xi0), fill_value=np.nan)

            # filter nan
            nan_mask = (~np.isnan(xi0)) & (~np.isnan(dlng0))
            xi, dlng = xi0[nan_mask], dlng0[nan_mask]

            # if len of True values == 0; no valid mols dln gamma/dxs is found.
            if sum(nan_mask) == 0:
                raise ValueError(f"No real values found for molecule {mol} in dlngammas_dxs.")

            # search for x-initial
            x_initial_found = np.any(np.isclose(xi, self._x_initial(mol)))
            if not x_initial_found:
                xi = np.append(xi, self._x_initial(mol))
                dlng = np.append(dlng, 0)

            # sort by mol fr.
            sorted_idxs = np.argsort(xi)[:: self._sort_idx_val(mol)]
            xi, dlng = xi[sorted_idxs], dlng[sorted_idxs]

            # integrate
            if integration_type == "polynomial":
                lng = self._polynomial_integration(xi, dlng, mol, polynomial_degree)
            elif integration_type == "numerical":
                lng = self._numerical_integration(xi, dlng, mol)
            else:
                raise ValueError(
                    f"Integration type not recognized. Must be `polynomial` or `numerical`, {integration_type} was provided."
                )

            # now prepare data for saving
            inverse_permutation = np.argsort(sorted_idxs)
            lng = lng[inverse_permutation]

            # remove ref. state if added
            if not x_initial_found:
                x_initial_idx = np.where(lng == 0)[0][0]
                lng = np.delete(lng, x_initial_idx)

            try:
                # force shape of lng is same as xi
                lng_i[nan_mask] = lng
                ln_gammas[:, i] = lng_i
            except ValueError as ve:
                if len(lng) != ln_gammas.shape[0]:
                    raise ValueError(
                        f"Length mismatch between lngammas: {len(lng)} and lngammas matrix: {ln_gammas.shape[0]}. Details: {ve}."
                    ) from ve

        self._populate_cache("lngammas", ln_gammas, "", [integration_type])
        return ln_gammas

    def _polynomial_integration(
        self, xi: np.ndarray, dlng: np.ndarray, mol: str, polynomial_degree: int = 5
    ) -> NDArray[np.float64]:
        # use polynomial to integrate dlng_dxs.
        try:
            dlng_fit = np.poly1d(np.polyfit(xi, dlng, polynomial_degree, w=self._weights(mol, xi)))
        except ValueError as ve:
            if polynomial_degree > len(xi):
                raise ValueError(
                    f"Not enough data points for polynomial fit. Required degree < number points. Details: {ve}."
                ) from ve
            elif len(xi) != len(dlng):
                raise ValueError(
                    f"Length mismatch! Shapes of xi {(len(xi))} and dlng {(len(xi))} do not match. Details: {ve}."
                ) from ve

        # integrate polynomial function to get ln gammas
        lng_fn = dlng_fit.integ(k=0)
        yint = 0 - lng_fn(1)  # adjust for lng=0 at x=1.
        lng_fn = dlng_fit.integ(k=yint)

        # check if _lngamma_fn has been initialized
        if "_lngamma_fn_dict" not in self.__dict__:
            self._lngamma_fn_dict = {}
        if "_dlngamma_fn_dict" not in self.__dict__:
            self._dlngamma_fn_dict = {}

        # add func. to dict
        self._lngamma_fn_dict[mol] = lng_fn
        self._dlngamma_fn_dict[mol] = dlng_fit

        # evalutate lng at xi
        lng = lng_fn(xi)
        return lng

    def _numerical_integration(self, xi: np.ndarray, dlng: np.ndarray, mol: str) -> NDArray[np.float64]:
        # using numerical integration via trapezoid method
        try:
            return np.asarray(cumulative_trapezoid(dlng, xi, initial=0))
        except Exception as e:
            raise Exception(f"Could not perform numerical integration for {mol}. Details: {e}.") from e

    def ge(self, lngammas: NDArray[np.float64]) -> NDArray[np.float64]:
        r"""
        Gibbs excess free energy calculated from activity coefficients.

        Notes
        -----
        Excess free energy, :math:`G^E`, is calculated according to:

        .. math::
            \frac{G^E}{RT} = \sum_{i=1}^n x_i \ln{\gamma_i}

        where:
            - :math:`x_i` is mol fraction of molecule :math:`i`
            - :math:`\gamma_i` is activity coefficient of molecule :math:`i`
        """
        temp = self.state.temperature()
        mol_fraction = self.state.mol_fr

        ge = self.gas_constant * temp * (mol_fraction * lngammas).sum(axis=1)
        self._populate_cache("ge", ge, "kJ/mol")

        return ge

    def gid(self) -> NDArray[np.float64]:
        r"""
        Ideal free energy calculated from mol fractions.

        Notes
        -----
        Ideal free energy, :math:`G^{id}`, is calculated according to:

        .. math::
            \frac{G^{id}}{RT} = \sum_{i=1}^n x_i \ln{x_i}

        where:
            - :math:`x_i` is mol fraction of molecule :math:`i`
        """
        if "_gibbs_ideal" not in self.__dict__:
            # to prevent error thrown for np.log10(0)
            mfr = self.state.mol_fr.copy()
            mfr[mfr == 0] = np.nan

            GID = self.gas_constant * self.state.temperature(units="K") * (mfr * np.log(mfr)).sum(axis=1)

            self._gibbs_ideal = np.asarray(GID)
            self._populate_cache("gid", self._gibbs_ideal, "kJ/mol")

        return self._gibbs_ideal

    def gm(self, lngammas: NDArray[np.float64]) -> NDArray[np.float64]:
        r"""
        Gibbs mixing free energy calculated from excess and ideal contributions.

        Notes
        -----
        Gibbs mixing free energy, :math:`\Delta G_{mix}`, is calculated according to:

        .. math::
            \Delta G_{mix} = G^E + G^{id}
        """
        gm = self.ge(lngammas) + self.gid()
        self._populate_cache("gm", gm, "kJ/mol")
        return gm

    def se(self, lngammas: NDArray[np.float64]) -> NDArray[np.float64]:
        r"""
        Excess entropy determined from Gibbs relation between enthlapy and free energy.

        Notes
        -----
        Excess entropy, :math:`S^{E}`, is calculated according to:

        .. math::
            S^E = \frac{\Delta H_{mix} - G^E}{T}
        """
        temp = self.state.temperature()

        self._se = (self.state.h_mix() - self.ge(lngammas)) / temp
        self._populate_cache("se", self._se, "kJ/mol/K")
        return self._se

    def build_cache(self, gamma_integration_type: str) -> None:
        """
        Build property cache with default units.

        Parameters
        ----------
        gamma_integration_type: str
            Integration type of activity coefficient derivatives. Options: numerical, polynomial.
        """
        lngammas = self.lngammas(gamma_integration_type)
        self.gm(lngammas)
        self.se(lngammas)
        self.i0()
        self.s0_e()
        self.s0_x_e()
        self.s0_xp_e()
        self.s0_p_e()
        self.det_hessian()
