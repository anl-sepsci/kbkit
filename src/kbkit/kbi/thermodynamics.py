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
            \bar{V} = \sum_i x_i \bar{V}_i^{pure}

        Returns
        -------
        np.ndarray
        """
        # first try for pure components & if not enough, then fall back to KBI values
        if self.systems.has_all_required_pures():
            return self.systems.ideal_property(name="molar_volume", units=units).value
        # otherwise use kbi values
        return (self.systems.x * self.molar_volume(units=units)).sum(axis=1)

    @property
    def z_i(self) -> np.ndarray:
        """np.ndarray: Electrons present in the system mapped to ``molecules``."""
        return self.systems.pure_property(name="electron_count").value

    @property
    def z_bar(self) -> np.ndarray:
        r"""Electrons as a function of composition.

        .. math::
            \bar{Z} = \sum_i x_i Z_i

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
            - :math:`Z_n`: Last element in :meth:`Z_i`

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
    def B(self) -> np.ndarray:
        r"""Calculates the fluctuation matrix, **B**.

        The matrix **B** represents particle number fluctuations within a fixed volume.
        In Kirkwood-Buff theory, it serves as the thermodynamic bridge between KBI (**G**) and the Helmholtz Hessian (**A**).

        Returns
        -------
        np.ndarray
            A 3D matrix with shape ``(n_sys, n_i, n_i)``,
            where ``n_sys`` is the number of systems and ``n_i`` is the number of unique components.


        .. math::
            B_{ij} = (A^{-1})_{ij} = \frac{\langle \Delta N_i \Delta N_j \rangle}{V} = x_i \delta_{ij} + \rho x_i x_j G_{ij}

        where:
            - :math:`\rho`: Mixture number density.
            - :math:`x_i`: Mole fraction of species :math:`i`.
            - :math:`G_{ij}`: Kirkwood-Buff integral (KBI) for the pair :math:`i,j`.
            - :math:`\delta_{ij}`: Kronecker delta (:math:`\delta_{ij}=1` if :math:`i=j`, else `0`).


        .. note::
            This matrix describes the system in the **grand canonical** (:math:`\mu VT`) limit.
            It is the direct mathematical inverse of the Helmholtz Hessian at constant volume (:math:`B = A^{-1}`).
        """
        return (
            self._x_3d * self.delta_ij[np.newaxis, :]
            + self.rho("molecule/nm^3")[:, np.newaxis, np.newaxis] * self._x_3d_sq * self.kbi("nm^3/molecule")
        )

    @cached_property_value()
    def A(self) -> np.ndarray:
        r"""Calculates the Helmholtz Hessian matrix, **A**.

        The Helmholtz Hessian consists of the second partial derivatives of the Helmholtz free energy with respect to particle numbers.
        It is computed via the inverse of the fluctuation matrix, **B**.

        Returns
        -------
        np.ndarray
            A 3D matrix with shape ``(n_sys, n_i, n_i)``,
            where ``n_sys`` is the number of systems and ``n_i`` is the number of unique components.


        .. math::
            A_{ij} = (B^{-1})_{ij} = \frac{1}{RT} \left( \frac{\partial \mu_i}{\partial N_j} \right)_{T,V,N_{k \neq j}}

        where:
            - :math:`\mu_i`: Chemical potential of species :math:`i`.
            - :math:`N_j`: Particle number (moles) of species :math:`j`.


        .. note::
            This matrix corresponds to the **constant volume (canonical)** ensemble.
            Unlike the Gibbs Hessian, this is an $n \times n$ matrix and is directly related to particle number fluctuations.
        """
        try:
            return np.array([np.linalg.inv(block) for block in self.B()])
        except np.linalg.LinAlgError as e:
            raise ValueError("One or more B blocks are singular and cannot be inverted.") from e

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
            - :math:`\mathbf{A}_{mn}`: Helmholtz Hessian matrix for molecules :math:`m,n`.
            - :math:`x_m`: Mole fraction of molecule :math:`m`.
        """
        value = self._x_3d_sq * self.A()
        return value.sum(axis=(2, 1))

    @cached_property_value(default_units="kJ/mol")
    def M(self, units: str = "kJ/mol") -> np.ndarray:
        r"""Calculates the full curvature matrix, **M**.

        The matrix **M** represents the second derivatives of the Gibbs free energy with respect to particle numbers.
        It is the unconstrained :math:`n \times n` Hessian that accounts for volume fluctuations by applying the constant pressure correction to the Helmholtz Hessian.

        Returns
        -------
        np.ndarray
            A 3D matrix with shape ``(n_sys, n_i, n_i)``,
            where ``n_sys`` is the number of systems and ``n_i`` is the number of unique components.


        .. math::
            \begin{aligned}
            M_{ij} &= \left(\frac{\partial \mu_i}{\partial N_j}\right)_{T,P,N_{k \neq j}} \\
            &= \left(\frac{\partial \mu_i}{\partial N_j}\right)_{T,V,N_{k \neq j}} - \frac{\bar{V}_i \bar{V}_j}{\bar{V} \kappa_T} \\
            &= RT \left[ A_{ij} - \frac{\left(\sum_{k=1}^n x_k A_{ik}\right) \left(\sum_{k=1}^n x_k A_{jk}\right)}{\sum_{m=1}^n\sum_{n=1}^n x_m x_n A_{mn}} \right] \\
            \end{aligned}

        where:
            - :math:`A_{ij}`: Elements of the Helmholtz Hessian.
            - :math:`x_k`: Mole fraction of molecule :math:`k`.
            - :math:`\bar{V}_i`: Partial molar volume of molecule :math:`i`.
            - :math:`\bar{V}`: Molar volume of the mixture.
            - :math:`\kappa_T`: Isothermal compressibility of the mixture.


        .. note::
            This matrix corresponds to the **constant pressure (isobaric-isothermal)** ensemble.
            Due to the Gibbs-Duhem relation, this matrix is mathematically singular (:math:`\det(M) = 0`).
        """
        upper = (self._x_3d * self.A()).sum(axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            term2 = (upper[:, :, np.newaxis] * upper[:, np.newaxis, :]) / self._l()[:, np.newaxis, np.newaxis]

        return self.RT(units)[:, np.newaxis, np.newaxis] * (self.A() - term2)

    @cached_property_value(default_units="1/kPa")
    def isothermal_compressibility(self, units: str = "1/kPa") -> np.ndarray:
        r"""
        Isothermal compressibility, :math:`\kappa_T`, of mixture.

        Returns
        -------
        np.ndarray
            A 1D array with shape ``(n_sys)``,
            where ``n_sys`` is the number of systems.


        .. math::
            \kappa_T RT = \frac{1}{\rho \sum_{j=1}^n \sum_{k=1}^n x_j x_k A_{jk}}

        where:
            - :math:`\rho`: Mixture number density.
            - :math:`A_{ij}`: Element of Helmholtz Hessian matrix for molecules :math:`i,j`.
        """
        value = 1 / (self.rho(units="mol/m^3") * self.RT("kJ/mol") * self._l())
        return self.Q_(value, "1/kPa").to(units).magnitude

    @cached_property_value(default_units="cm^3/mol")
    def molar_volume(self, units: str = "cm^3/mol") -> np.ndarray:
        r"""
        Partial molar volume of individual components.

        Parameters
        ----------
        units: str, optional
            Desired output units. Defaults to "cm^3/mol".

        Returns
        -------
        np.ndarray
            A 2D array with shape ``(n_sys, n_i)``,
            where ``n_sys`` is the number of systems and ``n_i`` is the number of unique components.


        .. math::
            \bar{V}_i = \frac{\sum_{j=1}^n x_j A_{ij}}{\rho \sum_{j=1}^n \sum_{k=1}^n x_j x_k A_{jk}}

        where:
            - :math:`\rho`: Mixture number density.
            - :math:`A_{ij}`: Element of Helmholtz Hessian matrix for molecules :math:`i,j`.
        """
        xj_Aij = self.systems.x[:,np.newaxis,:] * self.A()
        rho_units = "/".join(units.split("/")[::-1])
        return xj_Aij.sum(axis=2) / (self._l() * self.rho(units=rho_units))[:,np.newaxis]

    def _subtract_nth_elements(self, matrix: np.ndarray) -> np.ndarray:
        """Set up matrices for multicomponent analysis."""
        n = self.systems.n_i - 1
        mat_ij = matrix[:, :n, :n]
        mat_in = matrix[:, :n, n][:, :, np.newaxis]
        mat_jn = matrix[:, n, :n][:, np.newaxis, :]
        mat_nn = matrix[:, n, n][:, np.newaxis, np.newaxis]
        return np.asarray(mat_ij - mat_in - mat_jn + mat_nn)

    @cached_property_value(default_units="kJ/mol")
    def H(self, units: str = "kJ/mol") -> np.ndarray:
        r"""Calculates the Hessian matrix of Gibbs mixing free energy, **H**.

        Returns
        -------
        np.ndarray
            A 3D matrix with shape ``(n_sys, n_i-1, n_i-1)``,
            where ``n_sys`` is the number of systems and ``n_i`` is the number of unique components.


        .. math::
            H_{ij} = \left( \frac{\partial \mu_i}{\partial x_j} \right)_{T,P} = M_{ij} - M_{in} - M_{jn} + M_{nn}


        where:
            - :math:`M_{ij}`: Element of the curvature matrix **M** for molecules :math:`i,j`


        .. note::
            This matrix is defined in the (n-1) x (n-1) composition space.
            It represents the stability of the system at **constant pressure**.
            A state is considered stable only if this matrix is positive definite.
        """
        return self._subtract_nth_elements(self.M(units))

    @cached_property_value(default_units="kJ/mol")
    def det_H(self, units: str = "kJ/mol") -> np.ndarray:
        r"""Calculates the determinant of the Gibbs Hessian, **H**.

        The determinant of the reduced Hessian matrix is the primary indicator of thermodynamic stability in a multicomponent mixture.

        Returns
        -------
        np.ndarray
            A 1D array of shape ``(n_sys)``


        .. math::
            \mathcal{D} = \det(\mathbf{H})


        .. note::
            A system is thermodynamically stable if and only if :math:`\det(H) > 0`.
            The condition :math:`\det(H) = 0` defines the **spinodal line**, beyond which the single-phase mixture becomes spontaneously unstable to infinitesimal fluctuations.
        """
        with np.errstate(divide="ignore", invalid="ignore"):  # avoids zeros in np.ndarray
            return np.asarray([np.linalg.det(block) for block in self.H(units)])

    def _set_ref_to_zero(self, array: np.ndarray, ref: float = 1) -> np.ndarray:
        """Set value of array to zero where value is pure component."""
        if array.ndim == 1:
            array[np.array(np.where(self.systems.x == ref))[0, :]] = 0
        else:
            array[np.where(self.systems.x == ref)] = 0
        return array

    @cached_property_value(default_units="kJ/mol")
    def dmu_dxi(self, units: str = "kJ/mol") -> np.ndarray:
        r"""Calculates the chemical potential derivatives with respect to mole fraction.

        This property returns the diagonal elements of the chemical potential derivative matrix, representing the response of each species' chemical potential to its own mole fraction, constrained by the Gibbs-Duhem relation.

        Returns
        -------
        np.ndarray
            A 2D array of shape ``(n_sys, n_i)``, where ``n_sys`` is the number
            of systems and ``n_i`` is the number of unique components.


        .. math::
            \left(\frac{\partial \mu_i}{\partial x_i}\right)_{T,P} = \sum_j \left(\frac{\partial \mu_i}{\partial n_i}\right)_{T,P} \delta_{ij}

        .. note::
            For the first :math:`n-1` components, the derivatives are transformed from the particle-number basis (**M**).
            The :math:`n^{th}` component is calculated via the **Gibbs-Duhem equation**:

            .. math::
                \frac{\partial \mu_n}{\partial x_n} = \frac{1}{x_n} \sum_{j=1}^{n-1} x_j \frac{\partial \mu_j}{\partial x_j}

            This ensures that the composition derivatives are thermodynamically consistent across the entire mixture.
        """
        n = self.systems.n_i - 1
        M = self.M(units)

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
    def Gamma(self) -> np.ndarray:
        r"""Calculates the thermodynamic factors, :math:`\Gamma_i`.

        The thermodynamic factor scales the composition dependence of the chemical potential.

        Returns
        -------
        np.ndarray
            A 2D array of shape ``(n_sys, n_i)``.


        .. math::
            \Gamma_i = \frac{x_i}{RT} \left( \frac{\partial \mu_i}{\partial x_i} \right)_{T,P}
        """
        return (self.systems.x * self.dmu_dxi()) / self.RT()[:,np.newaxis]

    @cached_property_value()
    def dlngamma_dxi(self) -> np.ndarray:
        r"""
        Derivative of natural logarithm of the activity coefficient of molecule :math:`i` with respect to its own mole fraction.

        This represents the deviation from ideality in the chemical potential gradient.

        Returns
        -------
        np.ndarray
            A 3D matrix with shape ``(n_sys, n_i, n_i)``


        .. math::
            \frac{\partial \ln{\gamma_i}}{\partial x_i} = \frac{1}{R T}\left(\frac{\partial \mu_i}{\partial x_i}\right)_{T,P} - \frac{1}{x_i}

        where:
            - :math:`\mu_i`: Chemical potential of molecule :math:`i`
            - :math:`\gamma_i`: Activity coefficient of molecule :math:`i`
            - :math:`x_i`: Mole fraction of molecule :math:`i`
        """
        with np.errstate(divide="ignore", invalid="ignore"):
            return (1 / self.RT("kJ/mol"))[:, np.newaxis] * self.dmu_dxi("kJ/mol") - 1 / self.systems.x

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
    def lngamma(self) -> np.ndarray:
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
            *  :math:`\ln{\gamma_i}(x_i)`: Natural logarithm of the activity coefficient of molecule `i` at mole fraction :math:`x_i`.
            *  :math:`a`: Index of summation.
            *  :math:`a_0`: Starting value for index of summation.
            *  :math:`N`: Number of data points to sum over.
            *  :math:`x_i`: Mole fraction of component :math:`i`.
            *  :math:`\left(\frac{\partial \ln{\gamma_i}}{\partial x_i}\right)_{a}`: Derivative of the natural logarithm of the activity coefficient of component `i` with respect to its mole fraction, evaluated at point `a`.

        The integration starts at a reference state where :math:`x_i = a_0` and :math:`\ln{\gamma_i}(a_0) = 0`.
        """
        # now for the calculation
        dlng_dxs = self.dlngamma_dxi()

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

    @cached_property
    def activity_metadata(self) -> ActivityMetadata:
        """ActivityMetadata: Container for results from activity coefficient integration."""
        if not self._activity_coef_meta:
            self.lngamma()
        return ActivityMetadata(self._activity_coef_meta)

    @cached_property_value(default_units="kJ/mol")
    def g_ex(self, units: str = "kJ/mol") -> np.ndarray:
        r"""
        Gibbs excess energy from activity coefficients.

        .. math::
            \frac{G^E}{RT} = \sum_{i=1}^n x_i \ln{\gamma_i}

        where:
            - :math:`x_i`: Mole fraction of molecule :math:`i`.
            - :math:`\gamma_i`: Activity coefficient of molecule :math:`i`.
        """
        ge = self.RT(units) * (self.systems.x * self.lngamma()).sum(axis=1)
        # where any system contains a pure component, set excess to zero
        return self._set_ref_to_zero(ge, ref=1)

    @cached_property_value(default_units="kJ/mol")
    def h_mix(self, units: str = "kJ/mol") -> np.ndarray:
        r"""
        Enthalpy of mixing. Requires pure component simulations.

        .. math::
            \Delta H_{mix} = H - \sum_{i} x_i H_i^{pure}

        where:
            - :math:`H`: Enthalpy directly from simulation.
            - :math:`H_i^{pure}`: Enthalpy directly from simulation for pure :math:`i`.
        """
        return self.systems.excess_property(name="enthalpy", units=units).value

    @cached_property_value(default_units="kJ/mol/K")
    def s_ex(self, units: str = "kJ/mol/K") -> np.ndarray:
        r"""Excess entropy from mixing enthalpy and Gibbs excess energy. Requires pure component simulations.

        .. math::
            S^E = \frac{\Delta H_{mix} - G^E}{T}

        where:
            - :math:`x_i`: Mole fraction of molecule :math:`i`.
            - :math:`\Delta H_{mix}`: Enthalpy of mixing.
            - :math:`G^E`: Excess Gibbs energy.
        """
        energy_units = "/".join(units.split("/")[:2])
        se = (self.h_mix(energy_units) - self.g_ex(energy_units)) / self.temperature()
        return self._set_ref_to_zero(se, ref=1)

    @cached_property_value(default_units="kJ/mol")
    def g_id(self, units: str = "kJ/mol") -> np.ndarray:
        r"""
        Ideal free energy calculated from mole fractions.

        .. math::
            \frac{G^{id}}{RT} = \sum_{i=1}^n x_i \ln{x_i}

        where:
            - :math:`x_i` is mole fraction of molecule :math:`i`.
        """
        with np.errstate(divide="ignore", invalid="ignore"):
            gid = self.RT(units) * (self.systems.x * np.log(self.systems.x)).sum(axis=1)
        return self._set_ref_to_zero(gid, ref=1)

    @cached_property_value(default_units="kJ/mol/K")
    def s_mix(self, units: str = "kJ/mol/K") -> np.ndarray:
        r"""Mixing entropy, requires pure component simulations.

        .. math::
            \begin{aligned}
            \Delta S_{mix} &= S^E + S^{id} \\
                           &= S^E - R \sum_{i=1}^n x_i \ln{x_i}
            \end{aligned}

        where:
            - :math:`x_i`: Mole fraction of molecule :math:`i`.
            - :math:`S^E`: Excess entropy.
            - :math:`S^{id}`: Ideal entropy.
        """
        energy_units = "/".join(units.split("/")[:2])
        return self.s_ex(units) - self.g_id(energy_units) / self.temperature()

    @cached_property_value(default_units="kJ/mol")
    def g_mix(self, units: str = "kJ/mol") -> np.ndarray:
        r"""
        Gibbs mixing free energy calculated from excess and ideal contributions.

        .. math::
            \begin{aligned}
            \Delta G_{mix} &= G^E + G^{id} \\
                           &= \Delta H_{mix} - T \Delta S_{mix}
            \end{aligned}

        where:
            - :math:`x_i`: Mole fraction of molecule :math:`i`.
            - :math:`G^E`: Excess Gibbs energy.
            - :math:`\Delta G_{mix}`: Gibbs free energy of mixing.
            - :math:`\Delta H_{mix}`: Enthalpy of mixing.
            - :math:`\Delta S_{mix}`: Entropy of mixing.
        """
        return self.g_ex(units) + self.g_id(units)

    @cached_property_value()
    def s0_ij(self) -> np.ndarray:
        r"""Partial structure factors for pairwise interaction between components.

        Notes
        -----
        Partial structure factor, :math:`\hat{S}_{ij}(0)`, is calculated via:

        .. math::
            \hat{S}_{ij}(0) = B_{ij} = \rho x_i x_j G_{ij} + x_i \delta_{i,j}

        where:
            - :math:`G_{ij}` is the KBI value for molecules :math:`i,j`.
            - :math:`B` is the fluctuation matrix.

        .. note::
            Note that the normalization used here differs from that of the Ashcroft-Langreth partial structure factors used in some texts.
        """
        return self.B()

    @cached_property_value()
    def s0_x(self) -> np.ndarray:
        r"""Bhatia-Thornton composition-composition structure factor as q :math:`\rightarrow` 0, extended to a multicomponent system.

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
        r"""Bhatia-Thornton composition-density structure factor as q :math:`\rightarrow` 0, extended to a multicomponent system.

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
        r"""Bhatia-Thornton density-density structure factor as q :math:`\rightarrow` 0, extended to a multicomponent system.

        Notes
        -----
        Structure factor, :math:`\hat{S}^{\rho}(0)`, is a 1D vector (composition) and is calculated via:

        .. math::
            \hat{S}^{\rho}(0) = \sum_{k=1}^n \sum_{l=1}^n \hat{S}_{kl}(0)
        """
        return self.s0_ij().sum(axis=(2, 1))

    @cached_property_value()
    def s0_kappa(self) -> np.ndarray:
        r"""Contribution from isothermal compressibility to Bhatia-Thornton density-density structure factor as q :math:`\rightarrow` 0.

        Notes
        -----
        Structure factor, :math:`\hat{S}^{\kappa_T}(0)`, is calculated via:

        .. math::
            \hat{S}^{\kappa_T}(0) = \frac{RT \kappa_T}{\bar{V}}
        """
        return self.RT("kJ/mol") * self.isothermal_compressibility("1/kPa") / self.v_bar("m^3/mol")

    @cached_property_value()
    def s0_x_e(self) -> np.ndarray:
        r"""Contribution from extended Bhatia-Thornton composition-composition structure factors to electron density structure factor as q :math:`\rightarrow` 0.

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
        r"""Contribution from extended Bhatia-Thornton composition-density structure factors to electron density structure factor as q :math:`\rightarrow` 0.

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
        r"""Contribution from extended Bhatia-Thornton density-density structure factors to electron density structure factor as q :math:`\rightarrow` 0.

        Notes
        -----
        Structure factor, :math:`\hat{S}^{\rho,e}(0)`, is a 1D vector (composition) and is calculated via:

        .. math::
            \hat{S}^{\rho,e}(0) = \bar{Z}^2 \hat{S}^{\rho}(0)
        """
        return self.z_bar**2 * self.s0_p()

    @cached_property_value()
    def s0_kappa_e(self) -> np.ndarray:
        r"""Contribution from isothermal compressibility part of Bhatia-Thornton density-density structure factor to electron density structure factor as q :math:`\rightarrow` 0.

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
        r"""Contribution from extended Bhatia-Thornton composition-composition structure factors to x-ray intensity as q :math:`\rightarrow` 0.

        Notes
        -----
        X-ray intensity, :math:`I^{x}(0)`, is calculated via:

        .. math::
            I^{x}(0) = r_e^2 \rho N_A \hat{S}^{x,e}(0)
        """
        return self._calculate_i0_from_s0e(self.s0_x_e())

    @cached_property_value(default_units="1/cm")
    def i0_xp(self, units: str = "1/cm") -> np.ndarray:
        r"""Contribution from extended Bhatia-Thornton composition-density structure factors to x-ray intensity as q :math:`\rightarrow` 0.

        Notes
        -----
        X-ray intensity, :math:`I^{x\rho}(0)`, is calculated via:

        .. math::
            I^{x\rho}(0) = r_e^2 \rho N_A \hat{S}^{x\rho,e}(0)
        """
        return self._calculate_i0_from_s0e(self.s0_xp_e())

    @cached_property_value(default_units="1/cm")
    def i0_p(self, units: str = "1/cm") -> np.ndarray:
        r"""Contribution from extended Bhatia-Thornton density-density structure factors to x-ray intensity as q :math:`\rightarrow` 0.

        Notes
        -----
        X-ray intensity, :math:`I^{\rho}(0)`, is calculated via:

        .. math::
            I^{\rho}(0) = r_e^2 \rho N_A \hat{S}^{\rho,e}(0)
        """
        return self._calculate_i0_from_s0e(self.s0_p_e())

    @cached_property_value(default_units="1/cm")
    def i0_kappa(self, units: str = "1/cm") -> np.ndarray:
        r"""Contribution from isothermal compressibility part of Bhatia-Thornton density-density structure factor to x-ray intensity as q :math:`\rightarrow` 0.

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
