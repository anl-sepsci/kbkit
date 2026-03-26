r"""
Computes Kirkwood-Buff integrals (KBIs) from radial distribution function (RDF) and properties or a :class:`~kbkit.systems.properties.SystemProperties` object.

The following RDF and finite-volume correction options are implemented, following the procedure outlined by `Simon (2022) <https://doi.org/10.1063/5.0106162>`_:
    * Corrects RDF for molecule excess/depletion to accurately recover bulk density. [`Ganguly and van der Vegt (2013) <https://doi.org/10.1021/ct301017q>`_]
    * Uses the analytically correct form for hyperspheres to calculate finite-volume KBI [`Krüger et al. (2013) <https://doi.org/10.1021/jz301992u>`_]
    * KBI is extrapolated to the thermodynamic limit [`Dawass, Krüger, et al. (2020) <https://doi.org/10.3390/nano10040771>`_]
"""

import os
import warnings
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import cumulative_trapezoid

from kbkit.config.mplstyle import load_mplstyle
from kbkit.io.rdf import RdfParser

if TYPE_CHECKING:
    from kbkit.systems.properties import SystemProperties

# load plotting style
load_mplstyle()


class KBIConvergenceError(Exception):
    """Raised when KBI fails to converge according to one or more metrics."""


class LinearityError(KBIConvergenceError):
    """Raised when R² linearity metric is not satisfied."""


class KBIntegrator:
    """
    Class to compute the Kirkwood-Buff Integrals (KBI) from RDF data.

    Parameters
    ----------
    r: np.ndarray
        Radial distance array (nm).
    g: np.ndarray
        Radial distribution function values.
    volume: float
        Averaged simulation box volume (nm³).
    molecule_count: dict[str, int]
        Dictionary mapping molecule names to their counts.
    rdf_molecules: list[str], optional
        The two molecules in this RDF pair. If None, inferred from molecule_count.
    """

    def __init__(
        self,
        r: np.ndarray,
        g: np.ndarray,
        volume: float,
        molecule_count: dict[str, int],
        rdf_molecules: list[str],
    ) -> None:
        self.r = r
        self.g = g
        self.box_volume = volume
        self.molecule_count = molecule_count
        self.rdf_molecules = rdf_molecules

    @classmethod
    def from_rdf_parser(
        cls,
        rdf: RdfParser,
        volume: float,
        molecule_count: dict[str, int],
    ) -> "KBIntegrator":
        """
        Create KBIntegrator from an RdfParser object.

        Parameters
        ----------
        rdf: RdfParser
            Parsed RDF data with r, g attributes.
        volume: float
            Averaged simulation box volume (nm³).
        molecule_count: dict[str, int]
            Dictionary mapping molecule names to their counts.

        Returns
        -------
        KBIntegrator
            Initialized integrator with molecules extracted from rdf.fname.
        """
        # Extract molecule pair from filename
        rdf_molecules = RdfParser.extract_molecules(text=rdf.fname, mol_list=list(molecule_count.keys()))

        return cls(
            r=rdf.r,
            g=rdf.g,
            volume=volume,
            molecule_count=molecule_count,
            rdf_molecules=rdf_molecules,
        )

    @classmethod
    def from_system_properties(
        cls,
        rdf: RdfParser,
        system_properties: "SystemProperties",
    ) -> "KBIntegrator":
        """
        Create KBIntegrator from RdfParser and SystemProperties.

        Automatically extracts volume and molecule_count from system_properties.

        Parameters
        ----------
        rdf: RdfParser
            Parsed RDF data.
        system_properties: SystemProperties
            System properties containing volume and topology.

        Returns
        -------
        KBIntegrator
            Initialized integrator.
        """
        volume = system_properties.get("volume", units="nm^3", avg=True)
        if not isinstance(volume, float):
            raise TypeError(f"Expected float for volume, got {type(volume)}")

        molecule_count = system_properties.topology.molecule_count

        # Delegate to from_rdf_parser for molecule extraction
        return cls.from_rdf_parser(
            rdf=rdf,
            volume=volume,
            molecule_count=molecule_count,
        )

    @property
    def _mol_j(self) -> str:
        """Returns second molecule in `rdf_molecules` as default options."""
        return self.rdf_molecules[1]

    def kronecker_delta(self) -> int:
        """Return the Kronecker delta between molecules in RDF, i.e., determines if molecules :math:`i,j` are the same (returns True)."""
        return int(self.rdf_molecules[0] == self.rdf_molecules[1])

    @property
    def n_j(self) -> int:
        r"""Number of molecule :math:`j` in the system.

        Returns
        -------
        int
            Number of molecules :math:`j` in the system.
        """
        mol_j = self._mol_j

        # Validate molecule to be used in RDF integration for coordination number calculation.
        if len(mol_j) == 0:
            raise ValueError(f"Molecule '{mol_j}' cannot be empty str!")
        elif mol_j not in self.rdf_molecules:
            raise ValueError(f"Molecule '{mol_j}' not in rdf molecules '{self.rdf_molecules}'.")

        # compute molecule number
        return self.molecule_count[mol_j]

    def g_vdv(self) -> np.ndarray:
        r"""
        Compute the corrected radial distribution function, accounting for finite-size effects in the simulation box, based on the approach by `Ganguly and Van der Vegt (2013) <https://doi.org/10.1021/ct301017q>`_.

        Returns
        -------
        np.ndarray
            Factor for correcting RDF so density follows the bulk density of the system.

        Notes
        -----
        The correction is calculated as,

        .. math::
           g^{vdV}(r) = g(r) \frac{N_j f(r)}{N_j f(r) - \Delta N_j(r) - \delta_{ij}}

        .. math::
            f(r) = 1 - \frac{\frac{4}{3} \pi r^3}{\langle V \rangle}

        .. math::
            \rho_j = \frac{N_j}{\langle V \rangle}

        .. math::
            \Delta N_j(r) = \rho_j \int_0^{r_{max}} 4 \pi r^2 \bigl(g(r) - 1 \bigr) dr


        where:
         - :math:`r` is the distance
         - :math:`\langle V \rangle` is the box volume
         - :math:`N_j` is the number of particles of type \( j \)
         - :math:`g(r)` is the radial distribution function directly from simulation
         - :math:`\delta_{ij}` is a kronecker delta


        .. note::
            The cumulative integral :math:`\Delta N_j(r)` is approximated numerically using the trapezoidal rule.
        """
        # raise error if `box_vol` is zero
        if self.box_volume == 0:
            raise ZeroDivisionError("Simulation box volume cannot be zero!")
        elif not self.box_volume:
            raise ValueError("Simulation box volume required for Ganguly correction!")

        # calculate the reduced volume
        vr = 1 - ((4 / 3) * np.pi * self.r**3 / self.box_volume)

        # get the number density for Molecule :math:`j`
        rho_j = self.n_j / self.box_volume

        # function to integrate over
        f = 4.0 * np.pi * self.r**2 * rho_j * (self.g - 1)
        try:
            Delta_Nj = cumulative_trapezoid(f, x=self.r, dx=self.r[1] - self.r[0])
            Delta_Nj = np.append(Delta_Nj, Delta_Nj[-1])
        except IndexError as e:
            raise IndexError(f"RDF file is too short; {len(self.r)} lines detected!") from e

        # correct g(r) with GV correction
        vdv_f = self.n_j * vr / (self.n_j * vr - Delta_Nj - self.kronecker_delta())
        return np.asarray(self.g * vdv_f)  # make sure that an array is returned

    def hypersphere_weight(self) -> np.ndarray:
        r"""
        Correct KBI for finite volumes with an exact analytically correct form for hyperspheres, based on the method described by `Krüger et al. (2013) <https://doi.org/10.1021/jz301992u>`_.

        Returns
        -------
        np.ndarray
           Correction factor for finite volumes.

        Notes
        -----
        The correction factor, :math:`w(r)`, is defined as:

        .. math::
            w(r) = 1 - \frac{3}{2} \left( \frac{r}{r_{max}} \right) + \frac{1}{2} \left( \frac{r}{r_{max}} \right)^3

        where:
            - :math:`r` is the radial distance
            - :math:`r_{max}` is the maximum value in the RDF, typically half the length of the simulation box
        """
        return np.asarray(1 - (3 / 2) * (self.r / self.r.max()) + (1 / 2) * (self.r / self.r.max()) ** 3)

    def h_vdv(self) -> np.ndarray:
        r"""
        Calculate correlation function h(r) from van der Vegt corrected g:math:`^{vdV}`(r).

        Returns
        -------
        np.ndarray
            Correlation function h(r) as a numpy array.

        Notes
        -----
        The correlation function is defined as:

        .. math::
            h(r) = g^{vdV}(r) - 1
        """
        return self.g_vdv() - 1

    def running_kbi(self) -> np.ndarray:
        r"""
        Compute KBI as a function of radial distance between molecules :math:`i` and :math:`j`, i.e., running KBI (RKBI).

        Returns
        -------
        np.ndarray
            KBI values as a numpy array corresponding to distances :math:`r` from the RDF.

        Notes
        -----
        The KBI is computed using the formula:

        .. math::
            G_{ij}^R = \int_0^R 4 \pi r^2 w(r) h(r) dr

        where:
            - :math:`h(r)` is the correlation function
            - :math:`w(r)` is the finite volume correction factor
            - :math:`r` is the radial distance

        .. note::
            The integration is performed using the trapezoidal rule.
        """
        integrand = 4 * np.pi * self.r**2 * self.h_vdv() * self.hypersphere_weight()
        rkbi_arr = cumulative_trapezoid(integrand, self.r, initial=0)
        return np.asarray(rkbi_arr)

    @staticmethod
    def _find_max_linear_range(
        x: np.ndarray,
        y: np.ndarray,
        min_x_range: float,
        r2_threshold: float = 0.999,
    ) -> dict:
        """
        Find the maximum x-range where linear regression R² stays above threshold.

        Implements convergence metric:
        1. Linearity: R² >= r2_threshold

        Parameters
        ----------
        x : np.ndarray
            Array of x values (must be strictly increasing).
        y : np.ndarray
            Array of y values corresponding to x.
        min_x_range : float
            Minimum x-range to consider (must be > 0).
        r2_threshold : float, optional
            Desired minimum R² value (default: 0.999).

        Returns
        -------
        dict
            - 'x_start'   : Start of the optimal x-range.
            - 'x_end'     : End of the optimal x-range.
            - 'x_range'   : Total x-range (x_end - x_start).
            - 'r2'        : R² value for the optimal range.
            - 'slope'     : Slope of the linear fit (G_∞).
            - 'intercept' : Intercept of the linear fit.
            - 'x_fit'     : x values used in the fit.
            - 'y_fit'     : y values used in the fit.

        Raises
        ------
        LinearityError
            If no range meets the R² threshold.
        """
        # --- Input validation ---
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)

        if x.ndim != 1 or y.ndim != 1:
            raise ValueError("x and y must be 1D arrays.")
        if len(x) != len(y):
            raise ValueError("x and y must have the same length.")
        MAGIC_THREE = 3
        if len(x) < MAGIC_THREE:
            raise ValueError("At least 3 data points are required.")
        if not np.all(np.diff(x) > 0):
            raise ValueError("x must be strictly increasing (sorted).")
        if min_x_range <= 0:
            raise ValueError("min_x_range must be positive.")
        if not (0 < r2_threshold <= 1):
            raise ValueError("r2_threshold must be in the range (0, 1].")

        n = len(x)

        # --- Precompute cumulative sums for O(1) regression stats ---
        # We need: sum(x), sum(y), sum(x^2), sum(xy), sum(y^2), count
        cum_x = np.concatenate(([0.0], np.cumsum(x)))
        cum_y = np.concatenate(([0.0], np.cumsum(y)))
        cum_x2 = np.concatenate(([0.0], np.cumsum(x * x)))
        cum_xy = np.concatenate(([0.0], np.cumsum(x * y)))
        cum_y2 = np.concatenate(([0.0], np.cumsum(y * y)))

        def compute_regression_fast(i: int, j: int) -> tuple[float, float, float] | None:
            """
            Compute slope, intercept, and R² for x[i:j+1] in O(1).

            Returns None if fewer than 3 points or degenerate case.
            """
            count = j - i + 1
            MAGIC_THREE = 3
            if count < MAGIC_THREE:
                return None

            sx = cum_x[j + 1] - cum_x[i]
            sy = cum_y[j + 1] - cum_y[i]
            sx2 = cum_x2[j + 1] - cum_x2[i]
            sxy = cum_xy[j + 1] - cum_xy[i]
            sy2 = cum_y2[j + 1] - cum_y2[i]

            ss_xx = sx2 - (sx * sx) / count
            ss_yy = sy2 - (sy * sy) / count
            ss_xy = sxy - (sx * sy) / count

            if ss_xx <= 0 or ss_yy <= 0:
                return None

            slope = ss_xy / ss_xx
            intercept = (sy - slope * sx) / count
            r2 = (ss_xy**2) / (ss_xx * ss_yy)

            return slope, intercept, r2

        # --- Find all valid windows and track slope consistency ---
        valid_windows = []

        i_start = 0
        while i_start < n - 1:
            # Find minimum end index using searchsorted
            i_min_end = int(np.searchsorted(x, x[i_start] + min_x_range, side="left"))

            if i_min_end >= n:
                break

            # Check if minimum range meets R² threshold
            result = compute_regression_fast(i_start, i_min_end)
            if result is None or result[2] < r2_threshold:
                i_start += 1
                continue

            # Expand end index as far as R² allows
            i_end = i_min_end
            while i_end + 1 < n:
                result = compute_regression_fast(i_start, i_end + 1)
                if result is None or result[2] < r2_threshold:
                    break
                i_end += 1

            slope, intercept, r2 = compute_regression_fast(i_start, i_end)
            valid_windows.append(
                {
                    "i_start": i_start,
                    "i_end": i_end,
                    "x_start": x[i_start],
                    "x_end": x[i_end],
                    "x_range": x[i_end] - x[i_start],
                    "slope": slope,
                    "intercept": intercept,
                    "r2": r2,
                }
            )

            i_start += 1

        # --- Check: Linearity Metric ---
        if not valid_windows:
            raise LinearityError(
                f"No x-range of at least {min_x_range} nm found with R² >= {r2_threshold}. "
                "The running KBI does not exhibit linear scaling with 1/R. "
                "Consider: (1) running longer simulation, (2) reducing r2_threshold, "
                "(3) checking RDF convergence."
            )

        # Select best window (largest range)
        best_window = max(valid_windows, key=lambda w: w["x_range"])

        # --- Extract final results ---
        i_start = best_window["i_start"]
        i_end = best_window["i_end"]

        return {
            "x_start": best_window["x_start"],
            "x_end": best_window["x_end"],
            "x_range": best_window["x_range"],
            "r2": best_window["r2"],
            "slope": best_window["slope"],
            "intercept": best_window["intercept"],
            "x_fit": x[i_start : i_end + 1],
            "y_fit": y[i_start : i_end + 1],
        }

    def fit_running_kbi(
        self, min_r_range: float = 0.5, r2_threshold: float = 0.999, raise_on_convergence_error: bool = True
    ) -> dict:
        r"""
        Fit linear regression to running KBI for extrapolation to thermodynamic limit.

        .. math::
            R G_{ij}^R = R G_{ij}^\infty + F_{ij}^\infty

        where:
            * :math:`G_{ij}^\infty` is KBI in the thermodynamic limit.
            * :math:`F_{ij}^\infty` is a finite-size surface offset.

        Implements convergence checks from `Dawass et al. (2020) <https://doi.org/10.3390/nano10040771>`_.

        Parameters
        ----------
        min_r_range : float
            Minimum r-range to consider for KBI convergence (must be > 0).
        r2_threshold : float, optional
            Desired minimum R² value (default: 0.999).
        raise_on_convergence_error : bool, optional
            If True, raises KBIConvergenceError when convergence checks fail.
            If False, returns NaN and prints warning. Default: True.

        Returns
        -------
        dict
            Fit results including convergence metrics.
        """
        r = self.r
        y = r * self.running_kbi()  # G_R * R vs R

        # Perform linear fit with all convergence checks
        try:
            result = self._find_max_linear_range(
                x=r,
                y=y,
                min_x_range=min_r_range,
                r2_threshold=r2_threshold,
            )
            return result

        except KBIConvergenceError as e:
            if raise_on_convergence_error:
                raise
            else:
                warnings.warn(f"KBI convergence failed: {e}", RuntimeWarning, stacklevel=2)
                return {}

    def kbi(
        self, min_r_range: float = 0.5, r2_threshold: float = 0.999, raise_on_convergence_error: bool = True
    ) -> float:
        """
        Compute KBI with comprehensive convergence checking.

        Parameters
        ----------
        min_r_range : float
            Minimum r-range to consider for KBI convergence (must be > 0).
        r2_threshold : float, optional
            Desired minimum R² value (default: 0.999).
        raise_on_convergence_error : bool, optional
            If True, raises KBIConvergenceError when convergence checks fail.
            If False, returns NaN and prints warning. Default: True.

        Returns
        -------
        float
            Converged KBI value.
        """
        result = self.fit_running_kbi(
            min_r_range=min_r_range, r2_threshold=r2_threshold, raise_on_convergence_error=raise_on_convergence_error
        )
        return float(result.get("slope", np.nan))

    def plot_integrand(self, save_dir: str | None = None) -> None:
        """
        Plot integrand for running KBI calculation. Includes demonstrating the effect of KBI corrections on the integrand.

        Parameters
        ----------
        save_dir: str, optional
            Directory to save the plot. If not provided, the plot will be displayed but not saved
        """
        A = 4 * np.pi * self.r**2
        integrand_uncorr = A * self.h_vdv()
        integrand_k_anal = A * self.h_vdv() * self.hypersphere_weight()

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(self.r, integrand_uncorr, c="skyblue", label=r"vdV")
        ax.plot(self.r, integrand_k_anal, c="crimson", lw=1.2, zorder=3, label=r"vdV + Kr$\ddot{u}$ger")
        ax.set_xlabel(r"$R$ [$nm$]")
        ax.set_ylabel(r"$4 \pi r^2 \ w(r) \ [g(r) - 1]$")
        ax.legend(fontsize=12)

        if save_dir is not None:
            mols = "_".join(self.rdf_molecules)
            fig.savefig(os.path.join(save_dir, f"kbi_integrand_{mols}.pdf"), dpi=100)
        plt.show()

    def plot_extrapolation(
        self, min_r_range: float = 0.5, r2_threshold: float = 0.999, save_dir: str | None = None
    ) -> None:
        """Plot RDF and the running KBI fit to thermodynamic limit.

        Parameters
        ----------
        min_r_range : float
            Minimum r-range to consider for KBI convergence (must be > 0).
        r2_threshold : float, optional
            Desired minimum R² value (default: 0.999).
        save_dir : str, optional
            Directory to save the plot. If not provided, the plot will be displayed but not saved.
        """
        label = "-".join(self.rdf_molecules)

        fig, ax = plt.subplots(1, 3, figsize=(12, 3.6), sharex=True)
        ax[0].plot(self.r, self.g, c="skyblue", label=label)
        ax[0].set_xlabel(r"$r$ [$nm$]")
        ax[0].set_ylabel(r"$g(r)$")
        ax[0].legend()

        ax[1].plot(self.r, self.running_kbi(), c="skyblue")
        ax[1].set_xlabel(r"$R$ [$nm$]")
        ax[1].set_ylabel(r"$G_{{ij}}^R$ [$nm^3$]")

        ax[2].plot(self.r, self.r * self.running_kbi(), c="skyblue")
        result = self.fit_running_kbi(min_r_range=min_r_range, r2_threshold=r2_threshold)
        ax[2].plot(
            result["x_fit"],
            result["slope"] * result["x_fit"] + result["intercept"],
            "k--",
            lw=3,
            label=rf"Linear fit (R$^2$={result['r2']:.6f})",
        )
        ax[2].legend(fontsize=11)
        ax[2].set_xlabel(r"$R$ [$nm$]")
        ax[2].set_ylabel(r"$R \ G_{{ij}}^R$ [$nm^4$]")

        if save_dir is not None:
            mols = "_".join(self.rdf_molecules)
            fig.savefig(os.path.join(save_dir, f"kbi_extrapolation_{mols}.pdf"), dpi=100)
        plt.show()
