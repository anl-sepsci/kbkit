"""Constructs thermodynamic property matrices from KBIs across multiple systems."""

import warnings

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import cumulative_trapezoid

from kbkit.analysis.system_state import SystemState
from kbkit.calculators.static_structure_calculator import StaticStructureCalculator
from kbkit.schema.thermo_property import ThermoProperty, register_property

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
    gamma_integration_type: str, optional.
        How to perform activity coefficient integration. Options: numerical, polynomial (default: numerical).
    gamma_polynomial_degree: int, optional.
        If integration type is polynomial, what degree to use in fitting? (default: 5).

    Attributes
    ----------
    structure_calculator: StaticStructureCalculator
        Calculator for calculating static structure.
    kbis: np.ndarray
        Matrix of KBI values.
    state: SystemState
        Initialized SystemState object.
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
        self.kbi_matrix = kbi_matrix

        # how to integrate activity coefficients and what polynomial degree to be used if type=="polynomial"
        self.gamma_integration_type = gamma_integration_type
        self.gamma_polynomial_degree = gamma_polynomial_degree

        # initialize static structure calculator & set conditions
        self.structure_calculator = StaticStructureCalculator(
            molar_volume=self.state.molar_volume("cm^3/mol"),
            n_electrons=self.state.n_electrons,
            mol_fr=self.state.pure_mol_fr,
        )
        self.structure_calculator.update_conditions(
            T=float(self.state.temperature().mean()),
            hessian=self.hessian.value,
            isothermal_compressibility=self.isothermal_compressibility.value,
        )

    @register_property("kbis", "nm^3/molecule")
    def kbis(self) -> NDArray[np.float64]:
        """ThermoProperty: Matrix of KBI values."""
        return self.kbi_matrix

    @property
    def gas_constant(self) -> float:
        """float: Gas constant in kJ/mol/K."""
        return float(self.state.ureg("R").to("kJ/mol/K").magnitude)

    def kd(self) -> NDArray[np.float64]:
        """Kronecker delta between pairs of unique molecules."""
        return np.eye(self.state.n_comp)

    @register_property("A_inv_matrix", "")
    def A_inv_matrix(self) -> NDArray[np.float64]:
        r"""
        ThermoProperty: Inverse of matrix **A** corresponding to fluctuations in Helmholtz free energy representation.

        See Also
        --------
        :meth:`compute_A_inv_matrix` for full derivation and formula.
        """
        return self.compute_A_inv_matrix()

    def compute_A_inv_matrix(self) -> NDArray[np.float64]:
        r"""
        Compute the inverse of matrix **A** for each system from compositions and KBI matrix, **G**.

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
        mfr_3d = self.state.mol_fr[:, :, np.newaxis]  # reshape mol_fr array to 3d
        mfr_3d_sq = (
            self.state.mol_fr[:, :, np.newaxis] * self.state.mol_fr[:, np.newaxis, :]
        )  # compute square of 3d array
        rho_bar = self.state.rho_bar("molecule/nm^3")[:, np.newaxis, np.newaxis]  # compute mixture number density
        Aij_inv = mfr_3d * self.kd()[np.newaxis, :] + rho_bar * mfr_3d_sq * self.kbis.value  # inverse of
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
        ThermoProperty: Stability array (L), quantifies the stability of a multicomponent fluid mixture.

        See Also
        --------
        :meth:`compute_l_stability` for full derivation.
        """
        return self.compute_l_stability()

    def compute_l_stability(self) -> NDArray[np.float64]:
        r"""
        Compute stability array :math:`l` from compositions and Helmholtz stability matrix, **A**.

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
        A_mat = self.A_matrix.value
        mfr_3d_sq = self.state.mol_fr[:, :, np.newaxis] * self.state.mol_fr[:, np.newaxis, :]
        l_arr_calc = mfr_3d_sq * A_mat
        l_arr = np.nansum(l_arr_calc, axis=tuple(range(1, A_mat.ndim)))
        return l_arr

    @register_property("dmui_dxj", "kJ/mol")
    def dmui_dxj(self) -> NDArray[np.float64]:
        r"""
        ThermoProperty: Chemical potential derivatives, **M**, corresponding to composition fluctuations in Gibbs free energy representation.

        See Also
        --------
        :meth:`compute_dmui_dxj` for full derivation and formula.
        """
        return self.compute_dmui_dxj()

    def compute_dmui_dxj(self) -> NDArray[np.float64]:
        r"""
        Compute chemical potential derivatives, **M**, with respect to mol fraction for each system from compositions and Helmholtz stability matrix, **A**.

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
        ThermoProperty: Isothermal compressibility of mixture.

        See Also
        --------
        :meth:`compute_isothermal_compressibility` for full derivation and formula.
        """
        return self.compute_isothermal_compressibility()

    def compute_isothermal_compressibility(self) -> NDArray[np.float64]:
        r"""
        Compute the isothermal compressibility, :math:`\kappa`, from the Helmholtz stability matrix, **A**.

        Returns
        -------
        np.ndarray
            A 1D array with shape ``(n_sys)``,
            where ``n_sys`` is the number of systems.

        Notes
        -----
        Array :math:`\kappa` is computed using the formula:

        .. math::
            RT\kappa = \frac{\bar{V}}{l}

        where:
            - :math:`\bar{V}` is the mixing volume as a function of composition.
            - :math:`l` is the stability array (see :meth:`l_stability`).
        """
        upper = self.state.volume_bar() / (self.gas_constant * self.state.temperature())
        with np.errstate(divide="ignore", invalid="ignore"):
            kT = upper / self.l_stability.value
            kT_converted = np.asarray(self.state.Q_(kT, units="mol/kJ * nm^3/molecule").to("1/kPa").magnitude)
        return kT_converted

    def _matrix_setup(self, matrix: np.ndarray) -> NDArray[np.float64]:
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
        ThermoProperty: Hessian matrix of Gibbs mixing free energy.

        See Also
        --------
        :meth:`compute_hessian` for full derivation and formula.
        """
        return self.compute_hessian()

    def compute_hessian(self) -> NDArray[np.float64]:
        r"""
        Compute the Hessian matrix, **H**, of Gibbs mixing free energy.

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
        return self._matrix_setup(self.dmui_dxj.value)

    @register_property("det_hessian", "kJ/mol")
    def det_hessian(self) -> NDArray[np.float64]:
        r"""
        ThermoProperty: Determinant of the Hessian of Gibbs free energy of mixing.

        See Also
        --------
        :meth:`compute_det_hessian` for full derivation and formula.
        """
        return self.compute_det_hessian()

    def compute_det_hessian(self) -> NDArray[np.float64]:
        r"""
        Compute the determinant, :math:`|\mathbf{H}|`, of the Hessian matrix.

        Returns
        -------
        np.ndarray
            A 1D array of shape ``(n_sys)``

        Notes
        -----
        The determinant, :math:`|\mathbf{H}|`, quantifies the curvature of the Gibbs mixing free energy surface and is used to assess mixture stability.

        See Also
        --------
        :meth:`compute_hessian`
        """
        return np.asarray([np.linalg.det(block) for block in self.hessian.value])

    @register_property("dmui_dnj", "kJ/mol")
    def dmui_dnj(self) -> NDArray[np.float64]:
        r"""
        ThermoProperty: Chemical potential derivatives of molecule :math:`i` with respect to the number of residues of molecule :math:`j`.

        See Also
        --------
        :meth:`compute_dmui_dnj`
        """
        return self.compute_dmui_dnj()

    def compute_dmui_dnj(self) -> NDArray[np.float64]:
        r"""
        Compute chemical potential derivatives, with respect to residue numbers for each system.

        Returns
        -------
        np.ndarray
            A 3D matrix with shape ``(n_sys, n_comp, n_comp)``,
            where ``n_sys`` is the number of systems and ``n_comp`` is the number of unique components.

        Notes
        -----
        Elements in the matrix are calculated for molecules :math:`i,j`, using the formula:

        .. math::
           \left(\frac{\partial \mu_i}{\partial n_j}\right)_{T,P,n_k} = \frac{1}{n_T} \left(\frac{\partial \mu_i}{\partial x_j}\right)_{T,P,x_k}

        where:
            - :math:`n_T` is the total number of molecules present.
        """
        return self.dmui_dxj.value / self.state.total_molecules[:, np.newaxis, np.newaxis]

    @register_property("dmui_dxi", "kJ/mol")
    def dmui_dxi(self) -> NDArray[np.float64]:
        r"""
        ThermoProperty: Derivative of chemical potential of molecule :math:`i` with respect to mol fraction of molecule :math:`i`.

        See Also
        --------
        :meth:`compute_dmui_dxi` for full derivation and formula.
        """
        return self.compute_dmui_dxi()

    def compute_dmui_dxi(self) -> NDArray[np.float64]:
        r"""
        Compute the derivative of the chemical potential of each component with respect to its own mol fraction, enforcing thermodynamic consistency.

        Returns
        -------
        np.ndarray
            A 2D array of shape ``(n_sys, n_comp)``,
            where ``n_sys`` is the number of systems and ``n_comp`` is the number of unique components.

        Notes
        -----
        For each system, the chemical potential derivative matrix :math:`M_{ij}` is used to construct the derivatives:

        - For components ``i = 1, \ldots, n-1``:

            .. math::
                \left(\frac{\partial \mu_i}{\partial x_i}\right) = \mathrm{diag}\left(M_{ij} - M_{i,n}\right)_{j=1}^{n-1}

            This is implemented as:

            .. math::
                dmui\_dxi[:, :-1] = \mathrm{diag}\left(M_{ij} - M_{i,n}\right)

        - For the last component ``n`` (by Gibbs-Duhem):

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
        return dmui_dxi

    @register_property("dlngammas_dxs", "")
    def dlngammas_dxs(self) -> NDArray[np.float64]:
        r"""
        ThermoProperty: Activity coefficient derivatives with respect to mol fraction.

        See Also
        --------
        :meth:`compute_dlngammas_dxs` for full derivation and formulas.
        """
        return self.compute_dlngammas_dxs()

    def compute_dlngammas_dxs(self) -> NDArray[np.float64]:
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
        # Compute derivative of ln(gamma) with respect to composition
        factor = 1 / (self.gas_constant * self.state.temperature()[:, np.newaxis])
        with np.errstate(divide="ignore", invalid="ignore"):
            lng_dx = factor * self.dmui_dxi.value - 1 / self.state.mol_fr
        return lng_dx

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
        - "numerical": trapezoidal rule (see :meth:`dlngammas_numerical_integration`)
        - "polynomial": polynomial fit and integration (see :meth:`dlngammas_polynomial_integration`)
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
        return ln_gammas

    def dlngammas_polynomial_integration(
        self, xi: np.ndarray, dlng: np.ndarray, mol: str, polynomial_degree: int = 5
    ) -> NDArray[np.float64]:
        r"""
        Analytical integration of activity coefficient derivatives using polynomial fitting.

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

    def dlngammas_numerical_integration(self, xi: np.ndarray, dlng: np.ndarray, mol: str) -> NDArray[np.float64]:
        r"""
        Numerical integration of activity coefficient derivatives using the trapezoidal rule.

        The method approximates the integral as:

        .. math::
            \ln{\gamma_i}(x_i) \approx \sum_{a=a_0}^{x_i} \frac{\Delta x}{2} \left[\left(\frac{\partial \ln{\gamma_i}}{\partial x_i}\right)_{a} + \left(\frac{\partial \ln{\gamma_i}}{\partial x_i}\right)_{a \pm \Delta x}\right]

        where :math:`\Delta x` is the step size in :math:`x` between points.
        """
        try:
            return np.asarray(cumulative_trapezoid(dlng, xi, initial=0))
        except Exception as e:
            raise Exception(f"Could not perform numerical integration for {mol}. Details: {e}.") from e

    @register_property("ge", "kJ/mol")
    def ge(self) -> NDArray[np.float64]:
        r"""
        ThermoProperty: Gibbs excess energy from activity coefficients.

        See Also
        --------
        :meth:`compute_ge` for full formula.
        """
        return self.compute_ge()

    def compute_ge(self) -> NDArray[np.float64]:
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
        return self.gas_constant * self.state.temperature() * (self.state.mol_fr * self.lngammas.value).sum(axis=1)

    @register_property("gid", "kJ/mol")
    def gid(self) -> NDArray[np.float64]:
        r"""
        ThermoProperty: Gibbs ideal energy from mol fractions.

        See Also
        --------
        :meth:`compute_gid` for full formula.
        """
        return self.compute_gid()

    def compute_gid(self) -> NDArray[np.float64]:
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
        with np.errstate(divide="ignore", invalid="ignore"):
            GID = (
                self.gas_constant
                * self.state.temperature()
                * (self.state.mol_fr * np.log(self.state.mol_fr)).sum(axis=1)
            )
        return np.asarray(GID)

    @register_property("gm", "kJ/mol")
    def gm(self) -> NDArray[np.float64]:
        r"""
        ThermoProperty: Gibbs mixing energy from mol fractions and activity coefficients.

        See Also
        --------
        :meth:`compute_gm` for full formula.
        """
        return self.compute_gm()

    def compute_gm(self) -> NDArray[np.float64]:
        r"""
        Gibbs mixing free energy calculated from excess and ideal contributions.

        Notes
        -----
        Gibbs mixing free energy, :math:`\Delta G_{mix}`, is calculated according to:

        .. math::
            \Delta G_{mix} = G^E + G^{id}
        """
        return self.ge.value + self.gid.value

    @register_property("se", "kJ/mol/K")
    def se(self) -> NDArray[np.float64]:
        r"""
        ThermoProperty: Excess entropy from Gibbs excess property relationship.

        See Also
        --------
        :meth:`compute_se` for full formula.
        """
        return self.compute_se()

    def compute_se(self) -> NDArray[np.float64]:
        r"""
        Excess entropy determined from Gibbs relation between enthlapy and free energy.

        Notes
        -----
        Excess entropy, :math:`S^{E}`, is calculated according to:

        .. math::
            S^E = \frac{\Delta H_{mix} - G^E}{T}
        """
        return (self.state.h_mix() - self.ge.value) / self.state.temperature()

    @register_property("i0", "1/cm")
    def i0(self) -> NDArray[np.float64]:
        r"""
        X-ray intensity as q :math:`\rightarrow` 0.

        See Also
        --------
        :meth:`StaticStructureCalculator.i0()` for full derivation and calculation.
        """
        return self.structure_calculator.i0()

    @register_property("s0_e", "")
    def s0_e(self) -> NDArray[np.float64]:
        r"""
        Structure factor contribution to electron density as q :math:`\rightarrow` 0.

        See Also
        --------
        :meth:`StaticStructureCalculator.s0_e()` for full derivation and calculation.
        """
        return self.structure_calculator.s0_e()

    @register_property("s0_x_e", "")
    def s0_x_e(self) -> NDArray[np.float64]:
        r"""
        Contribution from concentration-concentration fluctuations to electron density structure factor as q :math:`\rightarrow` 0.

        See Also
        --------
        :meth:`StaticStructureCalculator.s0_x_e()` for full derivation and calculation.
        """
        return self.structure_calculator.s0_x_e()

    @register_property("s0_xp_e", "")
    def s0_xp_e(self) -> NDArray[np.float64]:
        r"""
        Contribution from concentration-density fluctuations to electron density structure factor as q :math:`\rightarrow` 0.

        See Also
        --------
        :meth:`StaticStructureCalculator.s0_xp_e()` for full derivation and calculation.
        """
        return self.structure_calculator.s0_xp_e()

    @register_property("s0_p_e", "")
    def s0_p_e(self) -> NDArray[np.float64]:
        r"""
        Contribution from density-density fluctuations to electron density structure factor as q :math:`\rightarrow` 0.

        See Also
        --------
        :meth:`StaticStructureCalculator.s0_p_e()` for full derivation and calculation.
        """
        return self.structure_calculator.s0_p_e()

    @register_property("s0_x", "")
    def s0_x(self) -> NDArray[np.float64]:
        r"""
        Contribution from concentration-concentration fluctuations to structure factor as q :math:`\rightarrow` 0.

        See Also
        --------
        :meth:`StaticStructureCalculator.s0_x()` for full derivation and calculation.
        """
        return self.structure_calculator.s0_x()

    @register_property("s0_xp", "")
    def s0_xp(self) -> NDArray[np.float64]:
        r"""
        Contribution from concentration-density fluctuations to structure factor as q :math:`\rightarrow` 0.

        See Also
        --------
        :meth:`StaticStructureCalculator.s0_xp()` for full derivation and calculation.
        """
        return self.structure_calculator.s0_xp()

    @register_property("s0_p", "")
    def s0_p(self) -> NDArray[np.float64]:
        r"""
        Contribution from density-density fluctuations to structure factor as q :math:`\rightarrow` 0.

        See Also
        --------
        :meth:`StaticStructureCalculator.s0_p()` for full derivation and calculation.
        """
        return self.structure_calculator.s0_p()

    def computed_properties(self) -> list[ThermoProperty]:
        """
        Collects all computed thermodynamic properties for the current state.

        Returns
        -------
        List[ThermoProperty]
            A list of `ThermoProperty` instances defined on this object. Each entry contains
            the name, value, units, and description of a property that has been computed and
            cached for the current thermodynamic state.

        Notes
        -----
        This method inspects the instance for attributes that are instances of `ThermoProperty`,
        allowing dynamic discovery of all registered quantities. It is useful for exporting,
        summarizing, or validating the full set of derived thermodynamic results.
        """
        return [getattr(self, attr) for attr in dir(self) if isinstance(getattr(self, attr), ThermoProperty)]
