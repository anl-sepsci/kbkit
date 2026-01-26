"""
Compute thermodynamic properties and structure factors from Kirkwood-Buff integrals (KBIs) across multicomponent systems.

`KBThermo` applies Kirkwood-Buff theory to a matrix of pairwise KB integrals and constructs thermodynamic property matrices such as:
    * hessians of Gibbs mixing free energy,
    * activity coefficient derivatives,
    * decouples enthalpic vs. entropic contribution to Gibbs mixing free energy,
    * structure factors (partial, Bhatia-Thornton),
    * and related x-ray intensities.

The class operates at constant temperature and uses system metadata (densities, compositions, species identities) provided by a :class:`~kbkit.systems.collection.SystemCollection` object.
It supports multiple strategies for integrating activity coefficient derivatives, including numerical integration and polynomial fitting.


.. note::
    * KBThermo does not compute KB integrals itself; it consumes a precomputed KBI matrix (e.g., from :class:`~kbkit.kbi.calculator.KBICalculator`).
    * All thermodynamic quantities are computed consistently across mixtures, enabling comparison of multicomponent systems or concentration series.
    * Designed for automated workflows within the KBKit analysis pipeline.
"""

from functools import cached_property
from typing import TYPE_CHECKING, Literal

import numpy as np
from scipy.integrate import cumulative_trapezoid

from kbkit.config.unit_registry import load_unit_registry
from kbkit.schema.activity_metadata import ActivityCoefficientResult, ActivityMetadata
from kbkit.schema.property_result import PropertyResult
from kbkit.utils.decorators import cached_property_value
from kbkit.visualization.thermo import ThermoPlotter

if TYPE_CHECKING:
    from kbkit.systems.collection import SystemCollection


class KBThermo:
    """
    Apply Kirkwood-Buff (KB) theory to calculate thermodynamic properties.

    This class inherits system properties from :class:`~kbkit.analysis.collection.SystemCollection` and uses them for the calculation of thermodynamic properties.

    Parameters
    ----------
    systems : SystemCollection
        SystemCollection at a constant temperature.
    kbi : PropertyResult
        KBI values for each pairwise interaction.
    activity_integration_type: str, optional
        Method for performing integration of activity coefficient derivatives.
    activity_polynomial_degree: int, optional
        Polynomial degree for fitting activity coefficient derivatives, if ``activity_integration_type`` is `polynomial`.
    """

    def __init__(
        self,
        systems: "SystemCollection",
        kbi: PropertyResult,
        activity_integration_type: Literal["numerical", "polynomial"] = "numerical",
        activity_polynomial_degree: int = 5,
    ) -> None:
        self.systems = systems
        self.kbi_res = kbi
        self.activity_integration_type = activity_integration_type.lower()
        self.activity_polynomial_degree = activity_polynomial_degree

        # get unit registry
        self.ureg = load_unit_registry()
        self.Q_ = self.ureg.Quantity

        # create cache for expensive calculations
        self._cache: dict[str, PropertyResult] = {}
        self._lngamma_fn_dict: dict[str, np.poly1d] = {}
        self._dlngamma_fn_dict: dict[str, np.poly1d] = {}
        self._activity_coef_meta: list[ActivityCoefficientResult] = []

    @cached_property_value(default_units="nm^3/molecule")
    def kbi(self, units: str = "nm^3/molecule") -> np.ndarray:
        """KBI values in desired units."""
        return self.kbi_res.to(units).value

    def R(self, units: str = "kJ/mol/K") -> float:
        """float: Gas constant."""
        return float(self.ureg("R").to(units).magnitude)

    def temperature(self, units: str = "K") -> np.ndarray:
        """np.ndarray: 1D array of Temperatures of each system."""
        return self.systems.simulated_property(name="Temperature", units=units).value

    def RT(self, units: str = "kJ/mol") -> np.ndarray:
        """np.ndarray: Gas constant (kJ/mol/K) x simulation Temperature."""
        return self.R(units + "/K") * self.temperature()

    def rho(self, units: str = "molecule/nm^3") -> np.ndarray:
        """np.ndarray: 1D array of number density of each system."""
        return self.systems.simulated_property(name="number_density", units=units).value

    def v_bar(self, units: str = "cm^3/mol") -> np.ndarray:
        r"""Ideal molar volumes.

        .. math::
            \bar{V} = \sum_i x_i V_i^{pure}

        Returns
        -------
        np.ndarray
        """
        return self.systems.ideal_property(name="molar_volume", units=units).value

    @property
    def z_i(self) -> np.ndarray:
        """np.ndarray: Electrons present in the system mapped to ``molecules``."""
        return self.systems.pure_property(name="electron_count").value

    @property
    def z_bar(self) -> np.ndarray:
        r"""Ideal electrons as a function of composition.

        .. math::
            \bar{Z} = \sum_i x_i Z_i^{pure}

        Returns
        -------
        np.ndarray
        """
        return self.systems.ideal_property(name="electron_count").value

    @property
    def z_i_diff(self) -> np.ndarray:
        r"""Difference in electrons from the last element.

        .. math::
            \Delta Z_i = Z_i - Z_n

        where:
            - :math:`Z_n` is the last element in :meth:`Z_i`

        from :math:`i=1 \rightarrow n-1` where :math:`n` is the number of molecule types present.

        Returns
        -------
        np.ndarray
        """
        return self.z_i[:-1] - self.z_i[-1]

    @property
    def delta_ij(self) -> np.ndarray:
        """np.ndarray: Kronecker delta between pairs of unique molecules (n x n array)."""
        return np.eye(self.systems.n_i)

    def _get_from_cache(self, cache_key: str, units: str):
        """Retrieve cached result and convert to requested units if available."""
        if cache_key in self._cache:
            return self._cache[cache_key].to(units)
        return None

    @property
    def _x_3d(self) -> np.ndarray:
        """Convert mole fractions to a 3d array."""
        return self.systems.x[:, :, np.newaxis]

    @property
    def _x_3d_sq(self) -> np.ndarray:
        """Calculate the square of mole fraction for pairwise combinations."""
        return self.systems.x[:, :, np.newaxis] * self.systems.x[:, np.newaxis, :]

    @cached_property_value()
    def A_inv(self) -> np.ndarray:
        r"""
        Inverse of matrix **A** corresponding to fluctuations in Helmholtz free energy representation, from compositions and KBI matrix, **G**.

        .. note::
            In literature, this is also referred to as the **B** matrix. Here we use **A** and **A** :math:`^{-1}` for clarity on relation between both matrices.

        Returns
        -------
        np.ndarray
            A 3D matrix with shape ``(n_sys, n_i, n_i)``,
            where ``n_sys`` is the number of systems and ``n_i`` is the number of unique components.

        Notes
        -----
        Elements of **A** :math:`^{-1}` are calculated for molecules :math:`i,j`, using the formula:

        .. math::
            A_{ij}^{-1} = B_{ij} = \rho x_i x_j G_{ij} + x_i \delta_{i,j}

        where:
            - :math:`\rho` is the mixture number density.
            - :math:`G_{ij}` is the KBI for the pair of molecules.
            - :math:`x_i` is the mole fraction of molecule :math:`i`.
            - :math:`\delta_{i,j}` is the Kronecker delta for molecules :math:`i,j`.
        """
        return (
            self._x_3d * self.delta_ij[np.newaxis, :]
            + self.rho("molecule/nm^3")[:, np.newaxis, np.newaxis] * self._x_3d_sq * self.kbi("nm^3/molecule")
        )

    @cached_property_value()
    def A(self) -> np.ndarray:
        """Stability matrix (**A**) of a thermodynamic system in the Helmholtz free energy representation."""
        try:
            return np.array([np.linalg.inv(block) for block in self.A_inv()])
        except np.linalg.LinAlgError as e:
            raise ValueError("One or more A_inv blocks are singular and cannot be inverted.") from e

    @cached_property_value()
    def _l(self) -> np.ndarray:
        r"""
        Stability array :math:`l`, quantifies the stability of a multicomponent fluid mixture.

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
            - :math:`x_m` is the mole fraction of molecule :math:`m`.
        """
        value = self._x_3d_sq * self.A()
        return value.sum(axis=(2, 1))

    @cached_property_value(default_units="kJ/mol")
    def chemical_potential_deriv(self, units: str = "kJ/mol") -> np.ndarray:
        r"""
        Chemical potential derivatives, **M**, corresponding to composition fluctuations in Gibbs free energy representation.

        Returns
        -------
        np.ndarray
            A 3D matrix with shape ``(n_sys, n_i, n_i)``,
            where ``n_sys`` is the number of systems and ``n_i`` is the number of unique components.

        Notes
        -----
        Elements of **M** are calculated for molecules :math:`i,j`, using the formula:

        .. math::
            \frac{M_{ij}}{RT} = \frac{1}{RT}\left(\frac{\partial \mu_i}{\partial x_j}\right)_{T,P,x_k} = A_{ij} - \frac{\left(\sum_{k=1}^n x_k A_{ik}\right) \left(\sum_{k=1}^n x_k A_{jk}\right)}{\sum_{m=1}^n\sum_{n=1}^n x_m x_n A_{mn}}

        where:
            - :math:`\mathbf{A}_{ij}` is the Helmholtz stability matrix for molecules :math:`i,j`.
            - :math:`x_k` is the mole fraction of molecule :math:`k`.
        """
        upper = (self._x_3d * self.A()).sum(axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            term2 = (upper[:, :, np.newaxis] * upper[:, np.newaxis, :]) / self._l()[:, np.newaxis, np.newaxis]

        return self.RT(units)[:, np.newaxis, np.newaxis] * (self.A() - term2)

    @cached_property_value(default_units="1/kPa")
    def isothermal_compressibility(self, units: str = "1/kPa") -> np.ndarray:
        r"""
        Isothermal compressibility, :math:`\kappa`, of mixture.

        Returns
        -------
        np.ndarray
            A 1D array with shape ``(n_sys)``,
            where ``n_sys`` is the number of systems.

        Notes
        -----
        Array :math:`\kappa` is computed using the formula:

        .. math::
            RT\kappa = \frac{1}{\rho \sum_{j=1}^n \sum_{k=1}^n x_j x_k A_{jk}}

        where:
            - :math:`\rho` is the mixture number density.
            - :math:`A_{ij}` is the stability matrix (see :meth:`A`).
        """
        return 1 / (self.rho(units="mol/m^3") * self.RT("kJ/mol") * self._l())

    def _subtract_nth_elements(self, matrix: np.ndarray) -> np.ndarray:
        """Set up matrices for multicomponent analysis."""
        n = self.systems.n_i - 1
        mat_ij = matrix[:, :n, :n]
        mat_in = matrix[:, :n, n][:, :, np.newaxis]
        mat_jn = matrix[:, n, :n][:, np.newaxis, :]
        mat_nn = matrix[:, n, n][:, np.newaxis, np.newaxis]
        return np.asarray(mat_ij - mat_in - mat_jn + mat_nn)

    @cached_property_value(default_units="kJ/mol")
    def hessian(self, units: str = "kJ/mol") -> np.ndarray:
        r"""
        Hessian matrix, **H**, of Gibbs mixing free energy.

        Returns
        -------
        np.ndarray
            A 3D matrix with shape ``(n_sys, n_i-1, n_i-1)``,
            where ``n_sys`` is the number of systems and ``n_i`` is the number of unique components.

        Notes
        -----
        Elements of **H** are calculated for molecules :math:`i,j`, using the formula:

        .. math::
            H_{ij} = M_{ij} - M_{in} - M_{jn} + M_{nn}

        where:
            - :math:`M_{ij}` is matrix **M** for molecules :math:`i,j`
            - :math:`n` represents the last element in **M** matrix
        """
        return self._subtract_nth_elements(self.chemical_potential_deriv(units))

    @cached_property_value(default_units="kJ/mol")
    def hessian_determinant(self, units: str = "kJ/mol") -> np.ndarray:
        r"""
        Determinant of the Hessian, :math:`|\mathbf{H}|`, of Gibbs free energy of mixing (units: kJ/mol).

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
        with np.errstate(divide="ignore", invalid="ignore"):  # avoids zeros in np.ndarray
            return np.asarray([np.linalg.det(block) for block in self.hessian(units)])

    def _set_ref_to_zero(self, array: np.ndarray, ref: float = 1) -> np.ndarray:
        """Set value of array to zero where value is pure component."""
        if array.ndim == 1:
            array[np.array(np.where(self.systems.x == ref))[0, :]] = 0
        else:
            array[np.where(self.systems.x == ref)] = 0
        return array

    @cached_property_value(default_units="kJ/mol")
    def chemical_potential_deriv_diag(self, units: str = "kJ/mol") -> np.ndarray:
        r"""
        Derivative of the chemical potential of each component with respect to its own mole fraction, enforcing thermodynamic consistency.

        Returns
        -------
        np.ndarray
            A 2D array of shape ``(n_sys, n_i)``,
            where ``n_sys`` is the number of systems and ``n_i`` is the number of unique components.

        Notes
        -----
        For each system, the chemical potential derivative matrix :math:`M_{ij}` is used to construct the derivatives:

        * For components :math:`i = 1, \ldots, n-1`:

        .. math::
            \left(\frac{\partial \mu_i}{\partial x_i}\right) = \mathrm{diag}\left(\frac{\partial \mu_i}{\partial x_j} - \frac{\partial \mu_i}{\partial x_n}\right)_{j=1}^{n-1}

        This is implemented as:

        .. math::
            dmui\_dxi[:, :-1] = \mathrm{diag}\left(\frac{\partial \mu_i}{\partial x_j} - \frac{\partial \mu_i}{\partial x_n}\right)

        * For the last component ``n`` (by Gibbs-Duhem):

        .. math::
            \left(\frac{\partial \mu_n}{\partial x_n}\right) = \frac{1}{x_n} \sum_{j=1}^{n-1} x_j \left(\frac{\partial \mu_j}{\partial x_j}\right)

        This ensures the sum of mole fraction derivatives is thermodynamically consistent.
        """
        n = self.systems.n_i - 1
        M = self.chemical_potential_deriv(units)

        # compute dmu_dxs; shape n-1 x n-1
        dmu_dxs = M[:, :n, :n] - M[:, :n, -1][:, :, np.newaxis]

        dmui_dxi = np.full_like(self.systems.x, np.nan)
        dmui_dxi[:, :-1] = np.diagonal(dmu_dxs, axis1=1, axis2=2)
        with np.errstate(divide="ignore", invalid="ignore"):  # avoids zeros in np.ndarray
            dmui_product = self.systems.x[:, :-1] * dmui_dxi[:, :-1]
            dmui_dxi[:, -1] = dmui_product.sum(axis=1) / self.systems.x[:, -1]

        # replace values of reference state with 0.
        return self._set_ref_to_zero(dmui_dxi, ref=1)

    @cached_property_value()
    def ln_activity_coef_deriv(self) -> np.ndarray:
        r"""
        Derivative of natural logarithm of the activity coefficient of molecule :math:`i` with respect to its own mole fraction.

        Returns
        -------
        np.ndarray
            A 3D matrix with shape ``(n_sys, n_i, n_i)``

        Notes
        -----
        Activity coefficient derivatives are calculated as follows:

        .. math::
            \frac{\partial \ln{\gamma_i}}{\partial x_i} = \frac{1}{R T}\left(\frac{\partial \mu_i}{\partial x_i}\right) - \frac{1}{x_i}

        where:
            - :math:`\mu_i` is the chemical potential of molecule :math:`i`
            - :math:`\gamma_i` is the activity coefficient of molecule :math:`i`
            - :math:`x_i` is the mole fraction of molecule :math:`i`
        """
        with np.errstate(divide="ignore", invalid="ignore"):
            return (1 / self.RT("kJ/mol"))[:, np.newaxis] * self.chemical_potential_deriv_diag("kJ/mol") - 1 / self.systems.x

    def _get_ref_state(self, mol: str) -> float:
        """Return reference state for a molecule; 1: `pure component`, 0: `infinite dilution`."""
        z0 = np.nan_to_num(self.systems.x.copy())
        comp_max = z0.max(axis=1)
        i = self.systems.get_mol_index(mol)
        is_max = z0[:, i] == comp_max
        return 1. if np.any(is_max) else 0.

    def _get_weights(self, mol: str, x: np.ndarray) -> np.ndarray:
        """Get fitting weights based on reference state."""
        weight_fns_mapped = {
            1: lambda x: 100 ** (np.log10(np.clip(x, 1e-10, 1.0))),
            0: lambda x: 100 ** (-np.log10(np.clip(x, 1e-10, 1.0))),
        }
        ref_state = self._get_ref_state(mol)
        return weight_fns_mapped[int(ref_state)](x)

    @cached_property_value()
    def ln_activity_coef(self) -> np.ndarray:
        r"""
        Natural logarithm of activity coefficients.

        Integrate the derivative of activity coefficients to obtain :math:`\ln{\gamma_i}` for each component.
        Use either numerical methods (trapezoidal rule) or polynomial fitting for integration.
        These parameters are chosen by the ``activity_integration_type`` and ``activity_polynomial_degree`` in `KBThermo` initialization.

        Returns
        -------
        np.ndarray
            A 2D array with shape ``(n_sys, n_i)``

        Notes
        -----
        The general formula for activity coefficient integration is:

        .. math::
            \ln{\gamma_i}(x_i) = \int \left(\frac{\partial \ln{\gamma_i}}{\partial x_i}\right) dx_i


        **Polynomial integration**: the method fits a polynomial, :math:`P(x_i)`, to the derivative data and integrates:

        .. math::
            \ln{\gamma_i}(x_i) = \int P(x_i) dx_i

        The integration constant is chosen so that :math:`\ln{\gamma_i}` obeys the boundary condition at the reference state.


        **Numerical Integration**: The trapezoidal rule is used to approximate the integral because an analytical solution is not available. The integral is approximated as:

        .. math::
           \ln{\gamma_i}(x_i) \approx \sum_{a=a_0}^{N-1} \frac{(x_i)_{a+1}-(x_i)_a}{2} \left[\left(\frac{\partial \ln{\gamma_i}}{\partial x_i}\right)_{a} + \left(\frac{\partial \ln{\gamma_i}}{\partial x_i}\right)_{a+1}\right]

        where:
            *  :math:`\ln{\gamma_i}(x_i)` is the natural logarithm of the activity coefficient of component `i` at mole fraction :math:`x_i`.
            *  :math:`a` is the index of summation.
            *  :math:`a_0` is the starting value for index of summation.
            *  :math:`N` is the number of data points to sum over.
            *  :math:`x_i` is the mole fraction of component :math:`i`.
            *  :math:`\left(\frac{\partial \ln{\gamma_i}}{\partial x_i}\right)_{a}` is the derivative of the natural logarithm of the activity coefficient of component `i` with respect to its mole fraction, evaluated at point `a`.

        The integration starts at a reference state where :math:`x_i = a_0` and :math:`\ln{\gamma_i}(a_0) = 0`.
        """
        # now for the calculation
        dlng_dxs = self.ln_activity_coef_deriv()

        ln_gammas = np.full_like(self.systems.x, fill_value=np.nan)
        for i, mol in enumerate(self.systems.molecules):
            xi = self.systems.x[:, i]
            dlng = dlng_dxs[:, i]

            # Filter valid data
            valid = (~np.isnan(xi)) & (~np.isnan(dlng))
            if not valid.any():
                raise ValueError(f"No valid data for molecule {mol}")

            # Get reference state info once
            x_ref = self._get_ref_state(mol)

            # Integrate
            if self.activity_integration_type == "polynomial":
                lng = self._integrate_polynomial(xi[valid], dlng[valid], x_ref, mol, self.activity_polynomial_degree)
            else:
                lng = self._integrate_numerical(xi[valid], dlng[valid], x_ref)

            ln_gammas[valid, i] = lng

            # update metadatalog
            self._activity_coef_meta.extend(
                [
                    ActivityCoefficientResult(
                        mol=mol, x=xi, y=dlng, property_type="derivative", fn=self._dlngamma_fn_dict.get(mol)
                    ),
                    ActivityCoefficientResult(
                        mol=mol, x=xi, y=lng, property_type="integrated", fn=self._lngamma_fn_dict.get(mol)
                    ),
                ]
            )

        return ln_gammas

    def _integrate_polynomial(
        self, xi: np.ndarray, dlng: np.ndarray, x_ref: float, mol: str, degree: int = 5
    ) -> np.ndarray:
        """Fit polynomial to dlng/dx and integrate analytically."""
        # Include reference point in fit
        xi_fit = np.append(xi, x_ref)
        dlng_fit_data = np.append(dlng, 0.0)  # dlng = 0 at reference

        # Compute weights
        weights = self._get_weights(mol, xi_fit)

        # Fit polynomial
        if len(xi_fit) <= degree:
            degree = len(xi_fit) - 1

        poly_coeffs = np.polyfit(xi_fit, dlng_fit_data, degree, w=weights)
        dlng_poly = np.poly1d(poly_coeffs)

        # Integrate: ∫ dlng/dx dx
        lng_poly = dlng_poly.integ()

        # Set integration constant: lng(x_ref) = 0
        C = -lng_poly(x_ref)
        lng_poly = dlng_poly.integ(k=C)

        # Store for later use
        mol_key = ".".join(list(mol)) if isinstance(mol, (tuple, list)) else str(mol)
        self._lngamma_fn_dict[mol_key] = lng_poly
        self._dlngamma_fn_dict[mol_key] = dlng_poly

        # Evaluate only at original points
        return lng_poly(xi)

    def _integrate_numerical(self, xi: np.ndarray, dlng: np.ndarray, x_ref: float) -> np.ndarray:
        """Numerically integrate using trapezoidal rule with proper reference."""
        # Sort data
        sort_idx = np.argsort(xi)
        xi_sorted = xi[sort_idx]
        dlng_sorted = dlng[sort_idx]

        # Find or insert reference point
        ref_idx = np.searchsorted(xi_sorted, x_ref)
        if ref_idx < len(xi_sorted) and np.isclose(xi_sorted[ref_idx], x_ref):
            # Reference point exists
            lng_sorted = cumulative_trapezoid(dlng_sorted, xi_sorted, initial=0)
            lng_sorted -= lng_sorted[ref_idx]  # Set lng(x_ref) = 0
        else:
            # Insert reference point
            xi_with_ref = np.insert(xi_sorted, ref_idx, x_ref)
            dlng_with_ref = np.insert(dlng_sorted, ref_idx, 0.0)
            lng_sorted = cumulative_trapezoid(dlng_with_ref, xi_with_ref, initial=0)
            lng_sorted = np.delete(lng_sorted, ref_idx)  # Remove inserted point

        # Unsort to match original order
        unsort_idx = np.argsort(sort_idx)
        return lng_sorted[unsort_idx]

    @property
    def activity_metadata(self) -> ActivityMetadata:
        """ActivityMetadata: Container for results from activity coefficient integration."""
        return ActivityMetadata(self._activity_coef_meta)

    @cached_property_value(default_units="kJ/mol")
    def g_ex(self, units: str = "kJ/mol") -> np.ndarray:
        r"""
        Gibbs excess energy from activity coefficients.

        Notes
        -----
        Excess free energy, :math:`G^E`, is calculated according to:

        .. math::
            \frac{G^E}{RT} = \sum_{i=1}^n x_i \ln{\gamma_i}

        where:
            - :math:`x_i` is mole fraction of molecule :math:`i`
            - :math:`\gamma_i` is activity coefficient of molecule :math:`i`
        """
        ge = self.RT(units) * (self.systems.x * self.ln_activity_coef()).sum(axis=1)
        # where any system contains a pure component, set excess to zero
        return self._set_ref_to_zero(ge, ref=1)

    @cached_property_value(default_units="kJ/mol")
    def h_mix(self, units: str = "kJ/mol") -> np.ndarray:
        r"""
        Enthalpy of mixing. Requires pure component simulations.

        Notes
        -----
        Mixing enthalpy, :math:`\Delta H_{mix}`, is calculated via:

        .. math::
            \Delta H_{mix} = H - \sum_{i} x_i H_i^{pure}

        where:
            - :math:`H` is the enthalpy directly from simulation
            - :math:`H_i^{pure}` is the enthalpy directly from simulation for pure :math:`i`

        See Also
        --------
        :func:`~kbkit.systems.collection.SystemCollection.excess_property` for calculation from simulation properties.
        """
        return self.systems.excess_property(name="enthalpy", units=units).value

    @cached_property_value(default_units="kJ/mol/K")
    def s_ex(self, units: str = "kJ/mol/K") -> np.ndarray:
        r"""Excess entropy from mixing enthalpy and Gibbs excess energy. Requires pure component simulations.

        Notes
        -----
        Excess entropy, :math:`S^E`, is calculated according to:

        .. math::
            S^E = \frac{\Delta H_{mix} - G^E}{T}

        where:
            - :math:`x_i` is mole fraction of molecule :math:`i`
        """
        energy_units = "/".join(units.split("/")[:2])
        se = (self.h_mix(energy_units) - self.g_ex(energy_units)) / self.temperature()
        return self._set_ref_to_zero(se, ref=1)

    @cached_property_value(default_units="kJ/mol")
    def g_id(self, units: str = "kJ/mol") -> np.ndarray:
        r"""
        Ideal free energy calculated from mole fractions.

        Notes
        -----
        Ideal free energy, :math:`G^{id}`, is calculated according to:

        .. math::
            \frac{G^{id}}{RT} = \sum_{i=1}^n x_i \ln{x_i}

        where:
            - :math:`x_i` is mole fraction of molecule :math:`i`
        """
        with np.errstate(divide="ignore", invalid="ignore"):
            gid = self.RT(units) * (self.systems.x * np.log(self.systems.x)).sum(axis=1)
        return self._set_ref_to_zero(gid, ref=1)

    @cached_property_value(default_units="kJ/mol/K")
    def s_mix(self, units: str = "kJ/mol/K") -> np.ndarray:
        r"""Mixing entropy, requires pure component simulations.

        Notes
        -----
        Mixing entropy, :math:`\Delta S_{mix}`, is calculated according to:

        .. math::
            \begin{aligned}
            \Delta S_{mix} &= S^E + S^{id} \\
                           &= S^E - R \sum_{i=1}^n x_i \ln{x_i}
            \end{aligned}
        """
        energy_units = "/".join(units.split("/")[:2])
        return self.s_ex(units) - self.g_id(energy_units) / self.temperature()

    @cached_property_value(default_units="kJ/mol")
    def g_mix(self, units: str = "kJ/mol") -> np.ndarray:
        r"""
        Gibbs mixing free energy calculated from excess and ideal contributions.

        Notes
        -----
        Gibbs mixing free energy, :math:`\Delta G_{mix}`, is calculated according to:

        .. math::
            \begin{aligned}
            \Delta G_{mix} &= G^E + G^{id} \\
                           &= \Delta H_{mix} - T \Delta S_{mix}
            \end{aligned}
        """
        return self.g_ex(units) + self.g_id(units)

    @cached_property_value()
    def s0_ij(self) -> np.ndarray:
        r"""Partial structure factors for pairwise interaction between components.

        Notes
        -----
        Partial structure factor, :math:`\hat{S}_{ij}(0)`, is calculated via:

        .. math::
            \hat{S}_{ij}(0) = A_{ij}^{-1} = \rho x_i x_j G_{ij} + x_i \delta_{i,j}

        where:
            - :math:`G_{ij}` is the KBI value for molecules :math:`i,j` (:meth:`kbi`)
            - :math:`A_{ij}^{-1}` is the inverse A matrix (:meth:`A_inv`)
        """
        return self.A_inv()

    @cached_property_value()
    def s0_x(self) -> np.ndarray:
        r"""Contribution from Bhatia-Thornton concentration-concentration fluctuations to structure factor as q :math:`\rightarrow` 0.

        Notes
        -----
        Structure factor, :math:`\hat{S}_{ij}^{x}(0)`, is a 3D matrix (composition x n-1 x n-1 components) and is calculated via:

        .. math::
            \hat{S}_{ij}^{x}(0) = \hat{S}_{ij}(0) - x_i \sum_{k=1}^n \hat{S}_{kj}(0) - x_j \sum_{k=1}^n \hat{S}_{ki}(0) + x_i x_j \sum_{k=1}^n \sum_{l=1}^n \hat{S}_{kl}(0)

        for `i` and `j` from 1 to n-1.
        """
        xi = self.systems.x[:, :, np.newaxis]
        xj = self.systems.x[:, np.newaxis, :]
        value = (
            self.s0_ij()
            - xi * (self.s0_ij()).sum(axis=2)[:, :, np.newaxis]
            - xj * (self.s0_ij()).sum(axis=1)[:, :, np.newaxis]
            + xi * xj * self.s0_ij().sum(axis=(2, 1))[:, np.newaxis, np.newaxis]
        )
        n = self.systems.n_i - 1
        return value[:, :n, :n]

    @cached_property_value()
    def s0_xp(self) -> np.ndarray:
        r"""Contribution from Bhatia-Thornton number-concentration fluctuations to structure factor as q :math:`\rightarrow` 0.

        Notes
        -----
        Structure factor, :math:`\hat{S}_{i}^{x\rho}(0)`, is a 2D array (composition x n-1 components) and is calculated via:

        .. math::
            \hat{S}_{i}^{x\rho}(0) = \sum_{k=1}^n \hat{S}_{ik}(0)  - x_i \sum_{k=1}^n \sum_{l=1}^n \hat{S}_{kl}(0)

        for i from 1 to n-1.
        """
        n = self.systems.n_i - 1
        value = self.s0_ij().sum(axis=2) - self.systems.x * self.s0_ij().sum(axis=(2, 1))[:, np.newaxis]
        return value[:, :n]

    @cached_property_value()
    def s0_p(self) -> np.ndarray:
        r"""Contribution from number-number fluctuations to structure factor as q :math:`\rightarrow` 0.

        Notes
        -----
        Structure factor, :math:`\hat{S}^{\rho}(0)`, is a 1D vector (composition) and is calculated via:

        .. math::
            \hat{S}^{\rho}(0) = \sum_{k=1}^n \sum_{l=1}^n \hat{S}_{kl}(0)
        """
        return self.s0_ij().sum(axis=(2, 1))

    @cached_property_value()
    def s0_kappa(self) -> np.ndarray:
        r"""Contribution from isothermal compressibility to density-density fluctuations structure factor as q :math:`\rightarrow` 0.

        Notes
        -----
        Structure factor, :math:`\hat{S}^{\kappa_T}(0)`, is calculated via:

        .. math::
            \hat{S}^{\kappa_T}(0) = \frac{RT \kappa_T}{\bar{V}}
        """
        return self.RT("kJ/mol") * self.isothermal_compressibility("1/kPa") / self.v_bar("m^3/mol")

    @cached_property_value()
    def s0_x_e(self) -> np.ndarray:
        r"""Contribution from Bhatia-Thornton concentration-concentration fluctuations to electron density structure factor as q :math:`\rightarrow` 0.

        Notes
        -----
        Structure factor, :math:`\hat{S}^{x,e}(0)`, is a 1D vector (composition) and is calculated via:

        .. math::
            \hat{S}^{x,e}(0) = \sum_{i=1}^{n-1}\sum_{j=1}^{n-1} \left( Z_i - Z_n \right) \left( Z_j - Z_n \right) \hat{S}_{ij}^{x}(0)
        """
        dz_sq = self.z_i_diff[:, np.newaxis] * self.z_i_diff[np.newaxis, :]
        value = dz_sq[np.newaxis, :, :] * self.s0_x()
        return value.sum(axis=(2, 1))

    @cached_property_value()
    def s0_xp_e(self) -> np.ndarray:
        r"""Contribution from Bhatia-Thornton number-concentration fluctuations to electron density structure factor as q :math:`\rightarrow` 0.

        Notes
        -----
        Structure factor, :math:`\hat{S}^{x\rho,e}(0)`, is a 1D vector (composition) and is calculated via:

        .. math::
            \hat{S}^{x\rho,e}(0) = 2 \bar{Z} \sum_{i=1}^{n-1} \left( Z_i - Z_n \right)  \hat{S}_{i}^{x\rho}(0)
        """
        value = self.z_i_diff[np.newaxis, :] * self.s0_xp()
        return 2 * self.z_bar * value.sum(axis=1)

    @cached_property_value()
    def s0_p_e(self) -> np.ndarray:
        r"""Contribution from Bhatia-Thornton number-number fluctuations to electron density structure factor as q :math:`\rightarrow` 0.

        Notes
        -----
        Structure factor, :math:`\hat{S}^{\rho,e}(0)`, is a 1D vector (composition) and is calculated via:

        .. math::
            \hat{S}^{\rho,e}(0) = \bar{Z}^2 \hat{S}^{\rho}(0)
        """
        return self.z_bar**2 * self.s0_p()

    @cached_property_value()
    def s0_kappa_e(self) -> np.ndarray:
        r"""Contribution from isothermal compressibility to Bhatia-Thornton number-number fluctuations electron density structure factor as q :math:`\rightarrow` 0.

        Notes
        -----
        Structure factor, :math:`\hat{S}^{\kappa_T, e}(0)`, is calculated via:

        .. math::
            \hat{S}^{\kappa_T, e}(0) = \bar{Z}^2 \hat{S}^{\kappa_T}(0)
        """
        return self.z_bar**2 * self.s0_kappa()

    @cached_property_value()
    def s0_e(self) -> np.ndarray:
        r"""Electron density structure factor as q :math:`\rightarrow` 0 for the entire mixture.

        Notes
        -----
        Structure factor, :math:`\hat{S}^{e}(0)`, can be calculated via partial or from Bhatia-Thornton structure factors (both are equivalent):

        .. math::
            \begin{aligned}
            \hat{S}^{e}(0) &= \sum_{i=1}^n \sum_{j=1}^n Z_i Z_j \hat{S}_{ij}(0) \\
                           &= \hat{S}^{x,e}(0) + \hat{S}^{x\rho,e}(0) + \hat{S}^{\rho,e}(0)
            \end{aligned}

        """
        ne_sq = self.z_i[:, np.newaxis] * self.z_i[np.newaxis, :]
        return (ne_sq * self.s0_ij()).sum(axis=(2, 1))

    def _calculate_i0_from_s0e(self, s0_e: np.ndarray) -> np.ndarray:
        r"""Calculates x-ray scattering intensity from electron density contribution of structure factor."""
        re = float(self.ureg("re").to("cm").magnitude)
        N_A = float(self.ureg("N_A").to("1/mol").magnitude)
        return re**2 * (1 / self.v_bar(units="cm^3/mol")) * N_A * s0_e

    @cached_property_value(default_units="1/cm")
    def i0_x(self, units: str = "1/cm") -> np.ndarray:
        r"""Contribution from concentration-concentration fluctuations to x-ray intensity as q :math:`\rightarrow` 0.

        Notes
        -----
        X-ray intensity, :math:`I^{x}(0)`, is calculated via:

        .. math::
            I^{x}(0) = r_e^2 \rho N_A \hat{S}^{x,e}(0)
        """
        return self._calculate_i0_from_s0e(self.s0_x_e())

    @cached_property_value(default_units="1/cm")
    def i0_xp(self, units: str = "1/cm") -> np.ndarray:
        r"""Contribution from number-concentration fluctuations to x-ray intensity as q :math:`\rightarrow` 0.

        Notes
        -----
        X-ray intensity, :math:`I^{x\rho}(0)`, is calculated via:

        .. math::
            I^{x\rho}(0) = r_e^2 \rho N_A \hat{S}^{x\rho,e}(0)
        """
        return self._calculate_i0_from_s0e(self.s0_xp_e())

    @cached_property_value(default_units="1/cm")
    def i0_p(self, units: str = "1/cm") -> np.ndarray:
        r"""Contribution from number-concentration fluctuations to x-ray intensity as q :math:`\rightarrow` 0.

        Notes
        -----
        X-ray intensity, :math:`I^{\rho}(0)`, is calculated via:

        .. math::
            I^{\rho}(0) = r_e^2 \rho N_A \hat{S}^{\rho,e}(0)
        """
        return self._calculate_i0_from_s0e(self.s0_p_e())

    @cached_property_value(default_units="1/cm")
    def i0_kappa(self, units: str = "1/cm") -> np.ndarray:
        r"""Contribution from isothermal compressibility to density-density fluctuations x-ray intensity as q :math:`\rightarrow` 0.

        Notes
        -----
        X-ray intensity, :math:`I^{\kappa_T}(0)`, is calculated via:

        .. math::
            I^{\kappa_T}(0) = r_e^2 \rho N_A \hat{S}^{\kappa_T,e}(0)
        """
        return self._calculate_i0_from_s0e(self.s0_kappa_e())

    @cached_property_value(default_units="1/cm")
    def i0(self, units: str = "1/cm") -> np.ndarray:
        r"""X-ray intensity as q :math:`\rightarrow` 0 for entire mixture.

        Notes
        -----
        X-ray intensity, :math:`I(0)`, is calculated via:

        .. math::
            \begin{aligned}
            I(0) &= r_e^2 \rho N_A \hat{S}^e \\
                 &= I^x(0) + I^{x\rho}(0) + I^{\rho}(0)
            \end{aligned}
        """
        return self._calculate_i0_from_s0e(self.s0_e())

    @cached_property
    def results(self) -> dict[str, PropertyResult]:
        """dict: Container for :class:`~kbkit.schema.property_result.PropertyResult` objects for KBI and KBI-derived quantities."""
        props = {}
        for attr in dir(self):
            if attr.startswith("_") or attr in ("Q_", "ureg", "results", "plotter"):
                continue

            val = getattr(self, attr)
            try:
                val = val()
            except TypeError:
                continue

            if attr in self._cache:
                props[attr] =self._cache[attr]

        # manually add desired props
        return props

    def plotter(self, molecule_map: dict[str, str] | None = None) -> ThermoPlotter:
        """
        Create a ThermoPlotter for visualizing KBI and KBI-derived properties as a function of composition.

        Returns
        -------
        ThermoPlotter
            Plotter instance for computing KBI-derived thermodynamic properties.
        """
        return ThermoPlotter(self, molecule_map=molecule_map)
