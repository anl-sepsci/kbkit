"""Constructs thermodynamic property matrices from KBIs across multiple systems."""

import warnings

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import cumulative_trapezoid

from kbkit.schema.thermo_property import ThermoProperty, register_property
from kbkit.systems.state import SystemState

# Suppress only the specific RuntimeWarning from numpy.linalg
warnings.filterwarnings("ignore", category=RuntimeWarning, module="numpy.linalg")


class KBThermo:
    """Apply Kirkwood-Buff (KB) theory to calculate thermodynamic properties from KBI matrix.

    This class inherits system properties from :class:`KBICalculator` and uses them for the calculation of thermodynamic properties.

    Parameters
    ----------
    state: SystemState
        SystemState at a constant temperature.
    kbi_matrix: NDArray[np.float64]
        Matrix of KBI values for each pairwise interaction.
    gamma_integration_type: str, optional.
        How to perform activity coefficient integration. Options: numerical, polynomial (default: numerical).
    gamma_polynomial_degree: int, optional.
        If integration type is polynomial, what degree to use in fitting? (default: 5).

    Attributes
    ----------
    state: SystemState
        Initialized SystemState object.
    structure_calc: StaticStructureCalculator
        Calculator for calculating static structure.
    """

    def __init__(
        self,
        state: SystemState,
        kbi_matrix: NDArray[np.float64],
        gamma_integration_type: str = "numerical",
        gamma_polynomial_degree: int = 5,
    ) -> None:
        # initialize SystemAnalyzer with config.
        self.state = state

        # create attribute from kbi_matrix
        self._kbi_matrix = kbi_matrix

        # how to integrate activity coefficients and what polynomial degree to be used if type=="polynomial"
        self.gamma_integration_type = gamma_integration_type
        self.gamma_polynomial_degree = gamma_polynomial_degree

    @register_property("kbi_matrix", "nm^3/molecule")
    def kbi_matrix(self) -> NDArray[np.float64]:
        """ThermoProperty: Matrix of KBI values."""
        return self._kbi_matrix

    @property
    def gas_constant(self) -> float:
        """float: Gas constant in kJ/mol/K."""
        return float(self.state.ureg("R").to("kJ/mol/K").magnitude)

    def kronecker_delta(self) -> NDArray[np.float64]:
        """Kronecker delta between pairs of unique molecules."""
        return np.eye(self.state.n_comp)

    @register_property("A_inv_matrix", "")
    def A_inv_matrix(self) -> NDArray[np.float64]:
        r"""
        ThermoProperty: Inverse of matrix **A** corresponding to fluctuations in Helmholtz free energy representation, from compositions and KBI matrix, **G**.

        Returns
        -------
        np.ndarray
            A 3D matrix with shape ``(n_sys, n_comp, n_comp)``,
            where ``n_sys`` is the number of systems and ``n_comp`` is the number of unique components.

        Notes
        -----
        Elements of **A** :math:`^{-1}` are calculated for molecules :math:`i,j`, using the formula:

        .. math::
            A_{ij}^{-1} = \rho x_i x_j G_{ij} + x_i \delta_{i,j}

        where:
            - :math:`\rho` is the mixture number density.
            - :math:`G_{ij}` is the KBI for the pair of molecules.
            - :math:`x_i` is the mol fraction of molecule :math:`i`.
            - :math:`\delta_{i,j}` is the Kronecker delta for molecules :math:`i,j`.
        """
        mfr_3d = self.state.mol_fr[:, :, np.newaxis]  # reshape mol_fr array to 3d
        mfr_3d_sq = (
            self.state.mol_fr[:, :, np.newaxis] * self.state.mol_fr[:, np.newaxis, :]
        )  # compute square of 3d array
        rho = self.state.mixture_number_density("molecule/nm^3")[:, np.newaxis, np.newaxis]  # compute mixture number density
        Aij_inv = (
            mfr_3d * self.kronecker_delta()[np.newaxis, :] + rho * mfr_3d_sq * self.kbi_matrix.value
        )  # inverse of
        return Aij_inv

    @register_property("A_matrix", "")
    def A_matrix(self) -> NDArray[np.float64]:
        """ThermoProperty: Stability matrix (**A**) of a thermodynamic system in the Helmholtz free energy representation."""
        A_inv = self.A_inv_matrix.value
        try:
            return np.array([np.linalg.inv(block) for block in A_inv])
        except np.linalg.LinAlgError as e:
            raise ValueError("One or more A_inv blocks are singular and cannot be inverted.") from e

    @register_property("l_stability", "")
    def l_stability(self) -> NDArray[np.float64]:
        r"""
        ThermoProperty: Stability array :math:`l`, quantifies the stability of a multicomponent fluid mixture.

        Returns
        -------
        np.ndarray
            A 1D array with shape ``(n_sys)``,
            where ``n_sys`` is the number of systems.

        Notes
        -----
        Array :math:`l` is computed using the formula:

        .. math::
            l = \sum_{m=1}^n\sum_{n=1}^n x_m x_n A_{mn}

        where:
            - :math:`\mathbf{A}_{mn}` is the Helmholtz stability matrix for molecules :math:`m,n`.
            - :math:`x_m` is the mol fraction of molecule :math:`m`.
        """
        mfr_3d_sq = self.state.mol_fr[:, :, np.newaxis] * self.state.mol_fr[:, np.newaxis, :]
        l_arr_calc = mfr_3d_sq * self.A_matrix.value
        return np.nansum(l_arr_calc, axis=(2, 1))

    @register_property("dmui_dxj", "kJ/mol")
    def dmui_dxj(self) -> NDArray[np.float64]:
        r"""
        ThermoProperty: Chemical potential derivatives, **M**, corresponding to composition fluctuations in Gibbs free energy representation.

        Returns
        -------
        np.ndarray
            A 3D matrix with shape ``(n_sys, n_comp, n_comp)``,
            where ``n_sys`` is the number of systems and ``n_comp`` is the number of unique components.

        Notes
        -----
        Elements of **M** are calculated for molecules :math:`i,j`, using the formula:

        .. math::
            \frac{M_{ij}}{RT} = \frac{1}{RT}\left(\frac{\partial \mu_i}{\partial x_j}\right)_{T,P,x_k} = A_{ij} - \frac{\left(\sum_{k=1}^n x_k A_{ik}\right) \left(\sum_{k=1}^n x_k A_{jk}\right)}{l}

        where:
            - :math:`\mathbf{A}_{ij}` is the Helmholtz stability matrix for molecules :math:`i,j`.
            - :math:`x_k` is the mol fraction of molecule :math:`k`.
            - :math:`l` is the stability array (see :meth:`l_stability`).
        """
        A_mat = self.A_matrix.value
        l_arr = self.l_stability.value

        upper_calc = self.state.mol_fr[:, :, np.newaxis] * A_mat
        upper = np.nansum(upper_calc, axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            term2 = (upper[:, :, np.newaxis] * upper[:, np.newaxis, :]) / l_arr[:, np.newaxis, np.newaxis]

        RT = self.gas_constant * self.state.temperature()[:, np.newaxis, np.newaxis]
        M_mat = RT * (A_mat - term2)
        return M_mat

    @register_property("isothermal_compressibility", "1/kPa")
    def isothermal_compressibility(self) -> NDArray[np.float64]:
        r"""
        ThermoProperty: Isothermal compressibility, :math:`\kappa`, of mixture.

        Returns
        -------
        np.ndarray
            A 1D array with shape ``(n_sys)``,
            where ``n_sys`` is the number of systems.

        Notes
        -----
        Array :math:`\kappa` is computed using the formula:

        .. math::
            RT\kappa = \frac{\bar{V}}{\sum_{j=1}^n \sum_{k=1}^n x_j x_k A_{jk}}

        where:
            - :math:`\bar{V}` is the molar volume of the mixture.
            - :math:`A_{ij}` is the stability matrix (see :meth:`A_matrix`).
        """
        RT = self.gas_constant * self.state.temperature()
        RTkT = self.state.mixture_molar_volume("m^3/mol") / self.l_stability.value
        return RTkT / RT

    def _subtract_nth_elements(self, matrix: NDArray[np.float64]) -> NDArray[np.float64]:
        """Set up matrices for multicomponent analysis."""
        n = self.state.n_comp - 1
        mat_ij = matrix[:, :n, :n]
        mat_in = matrix[:, :n, n][:, :, np.newaxis]
        mat_jn = matrix[:, n, :n][:, np.newaxis, :]
        mat_nn = matrix[:, n, n][:, np.newaxis, np.newaxis]
        return np.asarray(mat_ij - mat_in - mat_jn + mat_nn)

    @register_property("hessian", "kJ/mol")
    def hessian(self) -> NDArray[np.float64]:
        r"""
        ThermoProperty: Hessian matrix, **H**, of Gibbs mixing free energy.

        Returns
        -------
        np.ndarray
            A 3D matrix with shape ``(n_sys, n_comp-1, n_comp-1)``,
            where ``n_sys`` is the number of systems and ``n_comp`` is the number of unique components.

        Notes
        -----
        Elements of **H** are calculated for molecules :math:`i,j`, using the formula:

        .. math::
            H_{ij} = M_{ij} - M_{in} - M_{jn} + M_{nn}

        where:
            - :math:`M_{ij}` is matrix **M** for molecules :math:`i,j`
            - :math:`n` represents the last element in **M** matrix
        """
        return self._subtract_nth_elements(self.dmui_dxj.value)

    @register_property("hessian_determinant", "kJ/mol")
    def hessian_determinant(self) -> NDArray[np.float64]:
        r"""
        ThermoProperty: Determinant of the Hessian, :math:`|\mathbf{H}|`, of Gibbs free energy of mixing.

        Returns
        -------
        np.ndarray
            A 1D array of shape ``(n_sys)``

        Notes
        -----
        The determinant, :math:`|\mathbf{H}|`, quantifies the curvature of the Gibbs mixing free energy surface and is used to assess mixture stability.

        See Also
        --------
        :meth:`hessian`
        """
        return np.asarray([np.linalg.det(block) for block in self.hessian.value])



    def _set_pure_to_zero(self, array: NDArray[np.float64]) -> NDArray[np.float64]:
        """Set value of array to zero where value is pure component."""
        array[np.where(self.state.mol_fr == 1)] = 0
        return array

    @register_property("dmui_dxi", "kJ/mol")
    def dmui_dxi(self) -> NDArray[np.float64]:
        r"""
        ThermoProperty: Derivative of the chemical potential of each component with respect to its own mol fraction, enforcing thermodynamic consistency.

        Returns
        -------
        np.ndarray
            A 2D array of shape ``(n_sys, n_comp)``,
            where ``n_sys`` is the number of systems and ``n_comp`` is the number of unique components.

        Notes
        -----
        For each system, the chemical potential derivative matrix :math:`M_{ij}` is used to construct the derivatives:

        * For components ``i = 1, \ldots, n-1``:

        .. math::
            \left(\frac{\partial \mu_i}{\partial x_i}\right) = \mathrm{diag}\left(\frac{\partial \mu_i}{\partial x_j} - \frac{\partial \mu_i}{\partial x_n}\right)_{j=1}^{n-1}

        This is implemented as:

        .. math::
            dmui\_dxi[:, :-1] = \mathrm{diag}\left(\frac{\partial \mu_i}{\partial x_j} - \frac{\partial \mu_i}{\partial x_n}\right)

        * For the last component ``n`` (by Gibbs-Duhem):

        .. math::
            \left(\frac{\partial \mu_n}{\partial x_n}\right) = \frac{1}{x_n} \sum_{j=1}^{n-1} x_j \left(\frac{\partial \mu_j}{\partial x_j}\right)

        This ensures the sum of mol fraction derivatives is thermodynamically consistent.
        """
        mfr = self.state.mol_fr.copy()
        n = self.state.n_comp - 1
        M = self.dmui_dxj.value

        # compute dmu_dxs; shape n-1 x n-1
        dmu_dxs = M[:, :n, :n] - M[:, :n, -1][:, :, np.newaxis]

        dmui_dxi = np.full_like(mfr, np.nan)
        dmui_dxi[:, :-1] = np.diagonal(dmu_dxs, axis1=1, axis2=2)
        with np.errstate(divide="ignore", invalid="ignore"):  # avoids zeros in mfr
            mfr_dmui_product = mfr[:, :-1] * dmui_dxi[:, :-1]
            dmui_dxi[:, -1] = mfr_dmui_product.sum(axis=1) / mfr[:, -1]
        return self._set_pure_to_zero(dmui_dxi)  # replace values of pure component with 0

    @register_property("dlngammas_dxs", "")
    def dlngammas_dxs(self) -> NDArray[np.float64]:
        r"""
        ThermoProperty: Derivative of natural logarithm of the activity coefficient of molecule :math:`i` with respect to its mol fraction.

        Returns
        -------
        np.ndarray
            A 3D matrix with shape ``(n_sys, n_comp, n_comp)``

        Notes
        -----
        Activity coefficient derivatives, :math:`\frac{\partial \gamma_i}{\partial x_i}` are calculated as follows:

        .. math::
            \frac{\partial \ln{\gamma_i}}{\partial x_i} = \frac{1}{R T}\left(\frac{\partial \mu_i}{\partial x_i}\right) - \frac{1}{x_i}

        where:
            - :math:`\mu_i` is the chemical potential of molecule :math:`i`
            - :math:`\gamma_i` is the activity coefficient of molecule :math:`i`
            - :math:`x_i` is the mol fraction of molecule :math:`i`
        """
        # Compute derivative of ln(gamma) with respect to composition
        factor = 1 / (self.gas_constant * self.state.temperature()[:, np.newaxis])
        with np.errstate(divide="ignore", invalid="ignore"):
            lng_dx = factor * self.dmui_dxi.value - 1 / self.state.mol_fr
        return self._set_pure_to_zero(lng_dx)

    def _get_ref_state_dict(self, mol: str) -> dict[str, object]:
        """Return reference state parameters for a molecule."""
        z0 = np.nan_to_num(self.state.mol_fr.copy())
        comp_max = z0.max(axis=1)
        i = self.state._get_mol_idx(mol, self.state.unique_molecules)
        is_max = z0[:, i] == comp_max
        if np.any(is_max):
            return {
                "ref_state": "pure_component",
                "x_initial": 1.0,
                "sorted_idx_val": -1,
                "weight_fn": lambda x: 100 ** (np.log10(x)),
            }
        else:
            return {
                "ref_state": "inf_dilution",
                "x_initial": 0.0,
                "sorted_idx_val": 1,
                "weight_fn": lambda x: 100 ** (-np.log10(x)),
            }

    def _x_initial(self, mol: str) -> float:
        """Return initial mol fraction for reference state."""
        val = self._get_ref_state_dict(mol)["x_initial"]
        if isinstance(val, (float, int)):
            return float(val)
        else:
            raise TypeError(f"Could not convert value of type({type(val)}) to float.")

    def _sort_idx_val(self, mol: str) -> int:
        """Return sorting direction for reference state."""
        val = self._get_ref_state_dict(mol)["sorted_idx_val"]
        if isinstance(val, (float, int)):
            return int(val)
        else:
            raise TypeError(f"Could not convert value of type({type(val)}) to int.")

    def _weights(self, mol: str, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """Return weights for polynomial fitting based on reference state."""
        fn = self._get_ref_state_dict(mol)["weight_fn"]
        if callable(fn):
            return fn(x)
        else:
            raise TypeError("Could not exctract callable from weight_fn for mol.")

    @register_property("lngammas", "")
    def lngammas(self) -> NDArray[np.float64]:
        r"""
        ThermoProperty: Activity coefficients as a function of composition and molecule.

        See Also
        --------
        :meth:`compute_lngammas` for full derivation and formulas.
        """
        return self.compute_lngammas(
            integration_type=self.gamma_integration_type, polynomial_degree=self.gamma_polynomial_degree
        )

    def compute_lngammas(self, integration_type: str, polynomial_degree: int = 5) -> NDArray[np.float64]:
        r"""
        Integrate the derivative of activity coefficients to obtain :math:`\ln{\gamma_i}` for each component.

        Parameters
        ----------
        integration_type: str
            Integration method: "numerical" (trapezoidal rule) or "polynomial" (fit and integrate polynomial).
        polynomial_degree: int
            Degree of polynomial for fitting if using polynomial integration.

        Returns
        -------
        np.ndarray
            A 2D array with shape ``(n_sys, n_comp)``

        Notes
        -----
        The general formula for activity coefficient integration is:

        .. math::
            \ln{\gamma_i}(x_i) = \int_{a_0}^{x_i} \left(\frac{\partial \ln{\gamma_i}}{\partial x_i}\right) dx_i

        The integration method is chosen by the `integration_type` argument:
            * "numerical": trapezoidal rule (see :meth:`dlngammas_numerical_integration`)
            * "polynomial": polynomial fit and integration (see :meth:`dlngammas_polynomial_integration`)
        """
        integration_type = integration_type.lower()
        dlng_dxs = self.dlngammas_dxs.value  # avoid repeated calls

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
                lng = self.dlngammas_polynomial_integration(xi, dlng, mol, polynomial_degree)
            elif integration_type == "numerical":
                lng = self.dlngammas_numerical_integration(xi, dlng, mol)
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
        return self._set_pure_to_zero(ln_gammas)

    def dlngammas_polynomial_integration(
        self, xi: NDArray[np.float64], dlng: NDArray[np.float64], mol: str, polynomial_degree: int = 5
    ) -> NDArray[np.float64]:
        r"""
        Analytical integration of activity coefficient derivatives using polynomial fitting.

        Parameters
        ----------
        xi: NDArray[np.float64]
            Mol fraction 1D array to integrate over.
        dlng: NDArray[np.float64]
            Natural log of activity coefficients with respect to mol fraction.
        mol: str
            Molecule ID of mol fraction and activity coefficient derivative.
        polynomial_degree: int, optional.
            Polynomial degree for activity coefficient derivative fit (default: 5).

        Returns
        -------
        np.ndarray
            A 1D array with shape ``(n_sys)``

        Notes
        -----
        The method fits a polynomial :math:`P(x_i)` to the derivative data and integrates:

        .. math::
            \ln{\gamma_i}(x_i) = \int_{a_0}^{x_i} P(x_i) dx_i

        The integration constant is chosen so that :math:`\ln{\gamma_i}` obeys the boundary condition at the reference state.
        """
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

    def dlngammas_numerical_integration(
        self, xi: NDArray[np.float64], dlng: NDArray[np.float64], mol: str
    ) -> NDArray[np.float64]:
        r"""
        Numerical integration of activity coefficient derivatives using the trapezoidal rule.

        Parameters
        ----------
        xi: NDArray[np.float64]
            Mol fraction 1D array to integrate over.
        dlng: NDArray[np.float64]
            Natural log of activity coefficients with respect to mol fraction.
        mol: str
            Molecule ID of mol fraction and activity coefficient derivative.

        Returns
        -------
        np.ndarray
            A 1D array with shape ``(n_sys)``

        Notes
        -----
        The trapezoidal rule is used to approximate the integral because an analytical
        solution is not available.  The integral is approximated as:

        .. math::
           \ln{\gamma_i}(x_i) \approx \sum_{a=a_0}^{x_i} \frac{\Delta x}{2} \left[\left(\frac{\partial \ln{\gamma_i}}{\partial x_i}\right)_{a} + \left(\frac{\partial \ln{\gamma_i}}{\partial x_i}\right)_{a \pm \Delta x}\right]

        where:
            *  :math:`\ln{\gamma_i}(x_i)` is the natural logarithm of the activity coefficient of component `i` at mole fraction :math:`x_i`.
            *  :math:`a` is the index of summation
            *  :math:`a_0` is the starting value for the index of summation
            *  :math:`x_i` is the mole fraction of component :math:`i`.
            *  :math:`\Delta x` is the step size in :math:`x` between points.
            *  :math:`\left(\frac{\partial \ln{\gamma_i}}{\partial x_i}\right)_{a}` is the derivative of the natural logarithm of the activity coefficient of component `i` with respect to its mole fraction, evaluated at point `a`.

        The integration starts at a reference state where :math:`x_i = a_0` and
        :math:`\ln{\gamma_i}(a_0) = 0`.  The step size :math:`\Delta x` is determined
        by the spacing of the input data points.
        """
        try:
            return np.asarray(cumulative_trapezoid(dlng, xi, initial=0))
        except Exception as e:
            raise Exception(f"Could not perform numerical integration for {mol}. Details: {e}.") from e

    @register_property("g_ex", "kJ/mol")
    def g_ex(self) -> NDArray[np.float64]:
        r"""
        ThermoProperty: Gibbs excess energy from activity coefficients.

        Notes
        -----
        Excess free energy, :math:`G^E`, is calculated according to:

        .. math::
            \frac{G^E}{RT} = \sum_{i=1}^n x_i \ln{\gamma_i}

        where:
            - :math:`x_i` is mol fraction of molecule :math:`i`
            - :math:`\gamma_i` is activity coefficient of molecule :math:`i`
        """
        ge = self.gas_constant * self.state.temperature() * (self.state.mol_fr * self.lngammas.value).sum(axis=1)
        # where any system contains a pure component, set excess to zero
        ge[np.array(np.where(self.state.mol_fr == 1))[0, :]] = 0
        return ge
    
    @register_property("h_mix", "kJ/mol")
    def h_mix(self) -> NDArray[np.float64]:
        r"""
        ThermoProperty: Enthalpy of mixing from pure component simulations.

        See Also
        --------
        :meth:`~kbkit.systems.state.mixture_enthalpy` for calculation from simulation properties.
        """
        return self.state.mixture_enthalpy(units="kJ/mol")

    @register_property("s_ex", "kJ/mol/K")
    def s_ex(self) -> NDArray[np.float64]:
        r"""ThermoProperty: Excess entropy from mixing enthalpy and Gibbs excess energy.
        
        Notes
        -----
        Excess entropy, :math:`S^E`, is calculated according to:

        .. math::
            \frac{S^E} = \frac{\Delta H_{mix} - G^E}{T}

        where:
            - :math:`x_i` is mol fraction of molecule :math:`i`
        """
        se = (self.state.mixture_enthalpy("kJ/mol") - self.g_ex.value) / self.state.temperature("K")
        se[np.array(np.where(self.state.mol_fr == 1))[0, :]] = 0
        return se

    @register_property("g_id", "kJ/mol")
    def g_id(self) -> NDArray[np.float64]:
        r"""
        ThermoProperty: Ideal free energy calculated from mol fractions.

        Notes
        -----
        Ideal free energy, :math:`G^{id}`, is calculated according to:

        .. math::
            \frac{G^{id}}{RT} = \sum_{i=1}^n x_i \ln{x_i}

        where:
            - :math:`x_i` is mol fraction of molecule :math:`i`
        """
        with np.errstate(divide="ignore", invalid="ignore"):
            gid = (
                self.gas_constant
                * self.state.temperature()
                * (self.state.mol_fr * np.log(self.state.mol_fr)).sum(axis=1)
            )
        gid[np.array(np.where(self.state.mol_fr == 1))[0, :]] = 0
        return gid

    @register_property("g_mix", "kJ/mol")
    def g_mix(self) -> NDArray[np.float64]:
        r"""
        ThermoProperty: Gibbs mixing free energy calculated from excess and ideal contributions.

        Notes
        -----
        Gibbs mixing free energy, :math:`\Delta G_{mix}`, is calculated according to:

        .. math::
            \Delta G_{mix} = G^E + G^{id}
        """
        return self.g_ex.value + self.g_id.value

    @property
    def _delta_z(self) -> NDArray[np.float64]:
        r"""Calculate the difference in electrons."""
        return self.state.unique_electrons[:-1] - self.state.unique_electrons[-1]

    @property
    def _zbar(self) -> NDArray[np.float64]:
        r"""Calculate the linear combination of electrons for each system."""
        return self.state.total_electrons

    @register_property("s0_ij", "")
    def s0_ij(self) -> NDArray[np.float64]:
        r"""ThermoProperty: Partial structure factors for pairwise interaction between components.

        Notes
        -----
        Partial structure factor, :math:`\hat{S}_{ij}(0)`, is calculated via:

        .. math::
            \hat{S}_{ij}(0) = \frac{A_{ij}^{-1}}{(x_i x_j)^{1/2}}
        """
        xi = self.state.mol_fr[:, :, np.newaxis]
        xj = self.state.mol_fr[:, np.newaxis, :]
        return self.A_inv_matrix.value / np.sqrt(xi * xj)

    @register_property("s0_cc", "")
    def s0_cc(self) -> NDArray[np.float64]:
        r"""ThermoProperty: Contribution from concentration-concentration fluctuations to structure factor as q :math:`\rightarrow` 0.

        Notes
        -----
        Structure factor, :math:`\hat{S}_{CC}(0)`, is calcuted via:

        .. math::
            \hat{S}_{CC}(0) = A_{ij}^{-1} - x_i \sum_{k=1}^n A_{kj}^{-1} - x_j \sum_{k=1}^n A_{ki}^{-1} + x_i x_j \sum_{k=1}^n \sum_{l=1}^n A_{kl}^{-1}

        for i and j from 1 to n-1.
        """
        mfr = self.state.mol_fr
        xi = mfr[:, :, np.newaxis]
        xj = mfr[:, np.newaxis, :]
        t1 = self.A_inv_matrix.value
        t2 = -xi * np.nansum(self.A_inv_matrix.value, axis=2)[:, :, np.newaxis]
        t3 = -xj * np.nansum(self.A_inv_matrix.value, axis=1)[:, :, np.newaxis]
        t4 = xi * xj * np.nansum(self.A_inv_matrix.value, axis=(2, 1))[:, np.newaxis, np.newaxis]
        n = self.state.n_comp - 1
        return (t1 + t2 + t3 + t4)[:, :n, :n]

    @register_property("s0_nc", "")
    def s0_nc(self) -> NDArray[np.float64]:
        r"""ThermoProperty: Contribution from number-concentration fluctuations to structure factor as q :math:`\rightarrow` 0.

        Notes
        -----
        Structure factor, :math:`\hat{S}_{NC}(0)`, is calcuted via:

        .. math::
            \hat{S}_{NC}(0) = \sum_{k=1}^n A_{ik}^{-1}  - x_i \sum_{k=1}^n \sum_{l=1}^n A_{kl}^{-1}

        for i from 1 to n-1.
        """
        t1 = np.nansum(self.A_inv_matrix.value, axis=2)
        t2 = -self.state.mol_fr * np.nansum(self.A_inv_matrix.value, axis=(2, 1))[:, np.newaxis]
        n = self.state.n_comp - 1
        dz_sign = np.sign(self._delta_z)
        return dz_sign[np.newaxis, :] * (t1 + t2)[:, :n]

    @register_property("s0_nn", "")
    def s0_nn(self) -> NDArray[np.float64]:
        r"""ThermoProperty: Contribution from number-number fluctuations to structure factor as q :math:`\rightarrow` 0.

        Notes
        -----
        Structure factor, :math:`\hat{S}_{NN}(0)`, is calcuted via:

        .. math::
            \hat{S}_{NN}(0) = \sum_{k=1}^n \sum_{l=1}^n A_{kl}^{-1}
        """
        return np.nansum(self.A_inv_matrix.value, axis=(2, 1))

    @register_property("s0_kappa", "")
    def s0_kappa(self) -> NDArray[np.float64]:
        r"""ThermoProperty: Contribution from isothermal compressibility to density-density fluctuations structure factor as q :math:`\rightarrow` 0.

        Notes
        -----
        Structure factor, :math:`\hat{S}_{\kappa_T}(0)`, is calcuted via:

        .. math::
            \hat{S}_{\kappa_T}(0) = \frac{RT \kappa_T}{\bar{V}}
        """
        return (
            self.gas_constant
            * self.state.temperature()
            * self.isothermal_compressibility.value
            / self.state.ideal_molar_volume("m^3/mol")
        )

    @register_property("s0_x", "")
    def s0_x(self) -> NDArray[np.float64]:
        r"""ThermoProperty: Contribution from concentration fluctuations to structure factor as q :math:`\rightarrow` 0.

        Notes
        -----
        Structure factor, :math:`\hat{S}_x(0)`, is calcuted via:

        .. math::
            \hat{S}_x(0) = \sum_{i=1}^{n-1} \sum_{j=1}^{n-1} \hat{S}_{CC}(0) + \sum_{i=1}^{n-1} \hat{S}_{NC}(0) + \hat{S}_{NN}(0) - \hat{S}_{\kappa_T}(0)

        """
        return (
            np.nansum(self.s0_cc.value, axis=(2, 1))
            + np.nansum(self.s0_nc.value, axis=1)
            + self.s0_nn.value
            - self.s0_kappa.value
        )

    @register_property("s0_cc_e", "")
    def s0_cc_e(self) -> NDArray[np.float64]:
        r"""ThermoProperty: Contribution from concentration-concentration fluctuations to electron density structure factor as q :math:`\rightarrow` 0.

        Notes
        -----
        Structure factor, :math:`\hat{S}_{CC}^e(0)`, is calcuted via:

        .. math::
            \hat{S}_{CC}^e(0) = \sum_{i=1}^{n-1}\sum_{j=1}^{n-1} \left( Z_i - Z_n \right) \left( Z_j - Z_n \right) \hat{S}_{CC}(0)
        """
        dz_sq = self._delta_z[:, np.newaxis] * self._delta_z[np.newaxis, :]
        t1 = dz_sq[np.newaxis, :, :] * self.s0_cc.value
        return np.nansum(t1, axis=(2, 1))

    @register_property("s0_nc_e", "")
    def s0_nc_e(self) -> NDArray[np.float64]:
        r"""ThermoProperty: Contribution from number-concentration fluctuations to electron density structure factor as q :math:`\rightarrow` 0.

        Notes
        -----
        Structure factor, :math:`\hat{S}_{NC}^e(0)`, is calcuted via:

        .. math::
            \hat{S}_{NC}^e(0) = 2 \bar{Z} \sum_{i=1}^{n-1} \left( Z_i - Z_n \right)  \hat{S}_{NC}(0)
        """
        dz_sign = np.sign(self._delta_z)
        t1 = self._delta_z[np.newaxis, :] * dz_sign[np.newaxis, :] * self.s0_nc.value
        return 2 * self._zbar * np.nansum(t1, axis=1)

    @register_property("s0_nn_e", "")
    def s0_nn_e(self) -> NDArray[np.float64]:
        r"""ThermoProperty: Contribution from number-number fluctuations to electron density structure factor as q :math:`\rightarrow` 0.

        Notes
        -----
        Structure factor, :math:`\hat{S}_{NN}^e(0)`, is calcuted via:

        .. math::
            \hat{S}_{NN}^e(0) = \bar{Z}^2 \hat{S}_{NN}(0)
        """
        return self._zbar**2 * self.s0_nn.value

    @register_property("s0_kappa_e", "")
    def s0_kappa_e(self) -> NDArray[np.float64]:
        r"""ThermoProperty: Contribution from isothermal compressibility to density-density fluctuations electron density structure factor as q :math:`\rightarrow` 0.

        Notes
        -----
        Structure factor, :math:`\hat{S}_{\kappa_T}^e(0)`, is calcuted via:

        .. math::
            \hat{S}_{\kappa_T}^e(0) = \bar{Z}^2 \hat{S}_{\kappa_T}(0)
        """
        return self._zbar**2 * self.s0_kappa.value

    @register_property("s0_x_e", "")
    def s0_x_e(self) -> NDArray[np.float64]:
        r"""ThermoProperty: Contribution from concentration fluctuations to electron density structure factor as q :math:`\rightarrow` 0.

        Notes
        -----
        Structure factor, :math:`\hat{S}_x^e(0)`, is calcuted via:

        .. math::
            \hat{S}_x^e(0) = \hat{S}_{CC}^e(0) + \hat{S}_{NC}^e(0) + \hat{S}_{NN}^e(0) - \hat{S}_{\kappa_T}^e(0)
        """
        return self.s0_cc_e.value + self.s0_nc_e.value + self.s0_nn_e.value - self.s0_kappa_e.value

    @register_property("s0_e", "")
    def s0_e(self) -> NDArray[np.float64]:
        r"""ThermoProperty: Electron density structure factor as q :math:`\rightarrow` 0 for the entire mixture.

        Notes
        -----
        Structure factor, :math:`\hat{S}^{e}(0)`, is calcuted via:

        .. math::
            \hat{S}^{e}(0) = \sum_{i=1}^n \sum_{j=1}^n Z_i Z_j A_{ij}^{-1}

        """
        ne = self.state.unique_electrons
        ne_sq = ne[:, np.newaxis] * ne[np.newaxis, :]
        return np.nansum(ne_sq * self.A_inv_matrix.value, axis=(2, 1))

    def _calculate_i0_from_s0e(self, s0_elec) -> NDArray[np.float64]:
        r"""Calculates x-ray scattering intensity from electron density contribution of structure factor."""
        re = float(self.state.ureg("re").to("cm").magnitude)
        vbar = self.state.ideal_molar_volume(units="cm^3/mol")
        N_A = float(self.state.ureg("N_A").to("1/mol").magnitude)
        return re**2 * (1 / vbar) * N_A * s0_elec

    @register_property("i0_cc", "1/cm")
    def i0_cc(self) -> NDArray[np.float64]:
        r"""ThermoProperty:  Contribution from concentration-concentration fluctuations to x-ray intensity as q :math:`\rightarrow` 0.

        Notes
        -----
        X-ray intensity, :math:`I_{CC}(0)`, is calcuted via:

        .. math::
            I_{CC}(0) = r_e^2 \rho N_A \hat{S}_{CC}^e(0)
        """
        return self._calculate_i0_from_s0e(self.s0_cc_e.value)

    @register_property("i0_nc", "1/cm")
    def i0_nc(self) -> NDArray[np.float64]:
        r"""ThermoProperty:  Contribution from number-concentration fluctuations to x-ray intensity as q :math:`\rightarrow` 0.

        Notes
        -----
        X-ray intensity, :math:`I_{NC}(0)`, is calcuted via:

        .. math::
            I_{NC}(0) = r_e^2 \rho N_A \hat{S}_{NC}^e(0)
        """
        return self._calculate_i0_from_s0e(self.s0_nc_e.value)

    @register_property("i0_nn", "1/cm")
    def i0_nn(self) -> NDArray[np.float64]:
        r"""ThermoProperty:  Contribution from number-concentration fluctuations to x-ray intensity as q :math:`\rightarrow` 0.

        Notes
        -----
        X-ray intensity, :math:`I_{NN}(0)`, is calcuted via:

        .. math::
            I_{NN}(0) = r_e^2 \rho N_A \hat{S}_{NN}^e(0)
        """
        return self._calculate_i0_from_s0e(self.s0_nn_e.value)

    @register_property("i0_kappa", "1/cm")
    def i0_kappa(self) -> NDArray[np.float64]:
        r"""ThermoProperty:  Contribution from isothermal compressibility to density-density fluctuations x-ray intensity as q :math:`\rightarrow` 0.

        Notes
        -----
        X-ray intensity, :math:`I_{\kappa_T}(0)`, is calcuted via:

        .. math::
            I_{\kappa_T}(0) = r_e^2 \rho N_A \hat{S}_{\kappa_T}^e(0)
        """
        return self._calculate_i0_from_s0e(self.s0_kappa_e.value)

    @register_property("i0_x", "1/cm")
    def i0_x(self) -> NDArray[np.float64]:
        r"""ThermoProperty:  Contribution from concentration fluctuations to x-ray intensity as q :math:`\rightarrow` 0.

        Notes
        -----
        X-ray intensity, :math:`I_{x}(0)`, is calcuted via:

        .. math::
            I_{x}(0) = r_e^2 \rho N_A \hat{S}_x^e(0)
        """
        return self._calculate_i0_from_s0e(self.s0_x_e.value)

    @register_property("i0", "1/cm")
    def i0(self) -> NDArray[np.float64]:
        r"""ThermoProperty: X-ray intensity as q :math:`\rightarrow` 0 for entire mixture.

        Notes
        -----
        X-ray intensity, :math:`I(0)`, is calcuted via:

        .. math::
            I(0) = r_e^2 \rho N_A \hat{S}^e
        """
        return self._calculate_i0_from_s0e(self.s0_e.value)

