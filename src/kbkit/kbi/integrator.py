r"""
Computes Kirkwood-Buff integrals (KBIs) from radial distribution function (RDF) and properties or a :class:`~kbkit.systems.properties.SystemProperties` object.

By default (``weight_type='geometric'``), RDF and finite-volume correction options are implemented, following the procedure outlined by `Simon (2022) <https://doi.org/10.1063/5.0106162>`_:
    * Corrects RDF for molecule excess/depletion to accurately recover bulk density. [`Ganguly and van der Vegt (2013) <https://doi.org/10.1021/ct301017q>`_]
    * Uses the analytically correct form for hyperspheres to calculate finite-volume KBI [`Krüger et al. (2013) <https://doi.org/10.1021/jz301992u>`_]
    * KBI is extrapolated to the thermodynamic limit [`Dawass, Krüger, et al. (2020) <https://doi.org/10.3390/nano10040771>`_]

Various KBI correction methods demonstrated by `Krüger and Vlugt (2018) <https://doi.org/10.1103/PhysRevE.97.051301>`_ are accessible with the following ``weight_type`` parameters:
    * ``weight_type='none'``: uses the uncorrected RDF, :math:`g^{NpT}(r)` with :math:`u_0(r)` weight function.
    * ``weight_type='u0'``: uses the van der Vegt corrected RDF, :math:`g^{vdV}(r)` with :math:`u_0(r)` weight function.
    * ``weight_type='u1'``: uses the van der Vegt corrected RDF, :math:`g^{vdV}(r)` with :math:`u_1(r)` weight function.
    * ``weight_type='u2'``: uses the van der Vegt corrected RDF, :math:`g^{vdV}(r)` with :math:`u_2(r)` weight function.
    * ``weight_type='geometric'``: uses the van der Vegt corrected RDF, :math:`g^{vdV}(r)` with :math:`w(r)` weight function.

.. note::
    Only for ``weight_type='geometric'`` is linear extrapolation performed; the other weight functions were developed as an estimate of :math:`G^\infty`
"""

import warnings
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import matplotlib.pyplot as plt
import numpy as np
import scipy

from kbkit.config.mplstyle import load_mplstyle
from kbkit.io.rdf import RdfParser
from kbkit.kbi.exceptions import KBIConvergenceError, LinearityError
from kbkit.visualization.format import style_axes, style_legend

if TYPE_CHECKING:
    from kbkit.systems.properties import SystemProperties

# load plotting style
load_mplstyle()


class KBIntegrator:
    """
    Class to compute the Kirkwood-Buff Integrals (KBI) from RDF data.

    Parameters
    ----------
    r: np.ndarray
        Radial distance array (nm).
    gr: np.ndarray
        Radial distribution function values.
    box_volume: float
        Averaged simulation box volume (nm³).
    delta: int
        Kronecker delta for RDF molecules.
    weight_type: str, optional
        Type of weight function for finite-volume corrections. Options: ('none','u0','u1','u2','geometric')
    raise_on_convergence_error : bool, optional
        Only applied for ``weight_type='geometric'``, for linear extrapolation to thermodynamic limit.
        If True, raises KBIConvergenceError when convergence checks fail.
        If False, returns NaN and prints warning. Default: True.
    force: bool, optional
        Only applied for ``weight_type='geometric'``. If KBIConvergenceError is raised, prints warning and returns KBI for ``weight_type='u2'``.
    """

    def __init__(
        self,
        r: np.ndarray,
        gr: np.ndarray,
        n_ref: int,
        box_volume: float,
        delta: int,
        weight_type: Literal["none", "u0", "u1", "u2", "geometric"] = "geometric",
        raise_on_convergence_error: bool = True,
        force: bool = False,
    ) -> None:
        self.r = np.asarray(r)
        self.gr = np.asarray(gr)
        self.n_ref = int(n_ref)
        self.box_volume = float(box_volume)
        self.delta = int(delta)
        self.weight_type = weight_type.lower()
        self.rho_ref = self.n_ref / self.box_volume
        self.raise_on_convergence_error = raise_on_convergence_error
        self.force = force

    @classmethod
    def from_rdf(
        cls,
        rdf_file: str | Path,
        system_properties: "SystemProperties",
        weight_type: Literal["none", "u0", "u1", "u2", "geometric"] = "geometric",
        raise_on_convergence_error: bool = True,
        force: bool = False,
    ) -> "KBIntegrator":
        """
        Create KBI object from RdfParser and SystemProperties.

        Automatically extracts volume and molecule_count from system_properties.

        Parameters
        ----------
        rdf_file: str | Path
            File containing rdf data. Supported filetypes: ('.xvg','.txt','.csv','.xlsx')
        system_properties: SystemProperties
            System properties containing volume and topology.
        weight_type: str, optional
            Type of weight function for finite-volume corrections. Options: ('none','u0','u1','u2','geometric')
        raise_on_convergence_error : bool, optional
            Only applied for ``weight_type='geometric'``, for linear extrapolation to thermodynamic limit.
            If True, raises KBIConvergenceError when convergence checks fail.
            If False, returns NaN and prints warning. Default: True.
        force: bool, optional
            Only applied for ``weight_type='geometric'``. If KBIConvergenceError is raised, prints warning and returns KBI for ``weight_type='u2'``.

        Returns
        -------
        KBI
            Initialized integrator.
        """
        rdf = RdfParser(rdf_file)
        molecule_count = system_properties.get("molecule_count")
        rdf_mols = rdf.extract_molecules(text=Path(rdf_file).name, mol_list=molecule_count.keys())

        return cls(
            r=rdf.r,
            gr=rdf.gr,
            n_ref=molecule_count[rdf_mols[1]],
            box_volume=system_properties.get("volume", units="nm^3"),
            delta=int(rdf_mols[0] == rdf_mols[1]),
            weight_type=weight_type,
            raise_on_convergence_error=raise_on_convergence_error,
            force=force,
        )

    @cached_property
    def gr_vdv(self) -> np.ndarray:
        r"""
        Compute the corrected radial distribution function, accounting for excess/depletion of molecule :math:`i` around molecule :math:`j` due to a finite number of molecules, based on the approach by `Ganguly and Van der Vegt (2013) <https://doi.org/10.1021/ct301017q>`_.

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
        sphere_volume = (4 / 3) * np.pi * self.r**3
        Vr = 1 - (sphere_volume / self.box_volume)
        delta_n_ref = scipy.integrate.cumulative_trapezoid(
            4 * np.pi * self.rho_ref * self.r**2 * (self.gr - 1), x=self.r, initial=0
        )
        vdv_correction = self.n_ref * Vr / (self.n_ref * Vr - delta_n_ref - self.delta)
        return self.gr * vdv_correction

    def compute_hr(self, weight_type: str) -> np.ndarray:
        r"""Total correlation function, :math:`h(r) = g(r) - 1`.

        If ``weight_type='none'``,

        .. math::
            h(r) = g^{NpT}(r) - 1,

        where :math:`g^{NpT}(r)` is the uncorrected radial distribution function.

        For any other ``weight_type`` specified, the `Ganguly and Van der Vegt (2013) <https://doi.org/10.1021/ct301017q>`_ correction is applied,

        .. math::
            h(r) = g^{vdV}(r) - 1.
        """
        return self.gr - 1 if weight_type == "none" else self.gr_vdv - 1

    @cached_property
    def hr(self) -> np.ndarray:
        r"""np.ndarray: Total correlation function, :math:`h(r) = g(r) - 1` based on the ``weight_type`` intitalized in class."""
        return self.compute_hr(self.weight_type)

    def geometric_weight(self, r: np.ndarray) -> np.ndarray:
        r"""
        Geometric weighting function for 3D hypersphere.

        Correct KBI for finite volumes with an exact analytically correct form for hyperspheres, based on the method described by `Krüger et al. (2013) <https://doi.org/10.1021/jz301992u>`_.

        Parameters
        ----------
        r: np.ndarray
            Radial distance for weight function.

        Returns
        -------
        np.ndarray
           Geometric weight function for hyperspheres.

        Notes
        -----
        The geometric weight function, :math:`w(r)`, is defined as:

        .. math::
            w(r) = 4 \pi r^2 \left( 1 - \frac{3}{2}x + \frac{1}{2}x^3 \right),

        where :math:`x` is the dimensionless distance, :math:`x = r/L`.
        """
        return 4 * np.pi * r**2 * (1 - (3 / 2) * (r / r.max()) + (1 / 2) * (r / r.max()) ** 3)

    def u0_weight(self, r: np.ndarray) -> np.ndarray:
        r"""
        Simple truncation of :math:`r = L`.

        This weighting function is well-known to converge badly, it lacks any finite-volume corrections [`Krüger and Vlugt (2018) <https://doi.org/10.1103/PhysRevE.97.051301>`_].

        Parameters
        ----------
        r: np.ndarray
            Radial distance for weight function.

        Returns
        -------
        np.ndarray
           Weight function.

        Notes
        -----
        The weight function, :math:`u_0(r)`, is defined as:

        .. math::
            u_0(r) = 4 \pi r**2

        where:
            - :math:`r` is the radial distance
        """
        return 4 * np.pi * r**2

    def u1_weight(self, r: np.ndarray) -> np.ndarray:
        r"""
        First order Taylor series expansion in :math:`1/R` of the exact finite-volume integral of the sphere [`Krüger and Vlugt (2018) <https://doi.org/10.1103/PhysRevE.97.051301>`_].

        Parameters
        ----------
        r: np.ndarray
            Radial distance for weight function.

        Returns
        -------
        np.ndarray
            Weight function.

        Notes
        -----
        The weight function, :math:`u_1(r)`, is defined as:

        .. math::
            u_1(r) = 4 \pi r**2 \left( 1 - x^3 \right)
        """
        return 4 * np.pi * r**2 * (1 - (r / r.max()) ** 3)

    def u2_weight(self, r: np.ndarray) -> np.ndarray:
        r"""
        Approximation of KBI in the thermodynamic limit using the exact geometric weighting function for hyperspheres and approximation of the surface term [`Krüger and Vlugt (2018) <https://doi.org/10.1103/PhysRevE.97.051301>`_].

        Parameters
        ----------
        r: np.ndarray
            Radial distance for weight function.

        Returns
        -------
        np.ndarray
            Weight function.

        Notes
        -----
        The weight function, :math:`u_2(r)`, is defined as:

        .. math::
            \begin{aligned}
            u_2(r) &= w(r) \left( 1 + \frac{3}{2}x + \frac{9}{4}x^2 \right)
            &= 4 \pi r^2 \left( 1 - \frac{23}{8}x^3 + \frac{3}{4}x^4 + \frac{9}{8}x^5  \right)
            \end{aligned}
        """
        return self.geometric_weight(r=r) * (1 + (3 / 2) * (r / r.max()) + (9 / 4) * (r / r.max()) ** 2)

    @property
    def _weight_mapped(self) -> dict:
        return {
            "none": self.u0_weight,
            "u0": self.u0_weight,
            "u1": self.u1_weight,
            "u2": self.u2_weight,
            "geometric": self.geometric_weight,
        }

    def compute_running_kbi(self, weight_type: str):
        r"""
        Compute KBI as a function of radial distance between molecules, also referred to as the `running KBI`.

        We follow the different correction schemes dicussed in `Krüger and Vlugt (2018) <https://doi.org/10.1103/PhysRevE.97.051301>`_, where various ``weight_type`` implement different corrections.

        Parameters
        ----------
        weight_type: str
            Type of weight function for finite-volume corrections. Options: ('none','u0','u1','u2','geometric')

        Returns
        -------
        np.ndarray
            KBI values as a numpy array corresponding to distances :math:`r` from the RDF.

        Notes
        -----
        The KBI is computed using the formula, where :math:`L \equiv 6V/A` is the linear dimension of the system.

        .. math::
            G^V(L) = \int_0^L h(r) w(r) dr = G^\infty + \frac{1}{L}F^\infty + \mathcal{O}(L^{-2})

        With :math:`F^\infty` being the surface term, related to surface effects of the subvolume and :math:`h(r)` is the total correlation function.
        While the above equation provides an exact relation to :math:`G^\infty` it requires a linear region for extrapolation to the thermodynamic limit, which can be a disadvantage if a linear region is not identified.
        Krüger and Vlugt proposed a method for direct estimation of :math:`G^\infty`, where the accuracy is depending upon weight functions of the form, :math:`u_k(r)`:

        .. math::
            G^\infty \approx G_k(L) = \int_0^L h(r) u_k(r) dr.

        .. note::
            * The integration is performed using the trapezoidal rule.
        """
        weight_fn = self._weight_mapped[weight_type]
        hr = self.compute_hr(weight_type)
        rkbi = np.zeros(self.r.size)
        for i in range(1, rkbi.size):
            rkbi[i] = scipy.integrate.trapezoid(weight_fn(r=self.r[: i + 1]) * (hr[: i + 1]), x=self.r[: i + 1])
        return rkbi

    @cached_property
    def running_kbi_map(self) -> dict[str, np.ndarray]:
        """dict[str, np.ndarray]: Compute the running KBI for each ``weight_type``."""
        return {
            "none": self.compute_running_kbi(weight_type="none"),
            "u0": self.compute_running_kbi(weight_type="u0"),
            "u1": self.compute_running_kbi(weight_type="u1"),
            "u2": self.compute_running_kbi(weight_type="u2"),
            "geometric": self.compute_running_kbi(weight_type="geometric"),
        }

    @property
    def rkbi(self) -> np.ndarray:
        """np.ndarray: Running KBI for class initialized ``weight_type``."""
        return self.running_kbi_map[self.weight_type]

    def compute_geometric_extrapolation(
        self,
        positions: tuple[float, float] | None = None,
        maximize_r2: bool = False,
    ) -> dict:
        r"""
        Extrapolate KBI to the thermodynamic limit for ``weight_type='geometric'`` using linear regression [`Dawass, Krüger, et al. (2020) <https://doi.org/10.3390/nano10040771>`_].

        Parameters
        ----------
        position: tuple[float,float], optional
            Range of `r` values to include for linear fit. Values outside this range will be excluded.
        maximize_r2: bool, optional
            Search through valid positions to find the `r` values that maximize :math:`R^2` with a `r` range greater than 1.0 nm, and that include the last 10% of data (this ensures some range in the beginning is not selected).
        update: bool, optional
            Update the ``key='geometric'`` value in `kbi_map`.

        Returns
        -------
        dict
            Dictionary containing results from linear extrapolation.

        Notes
        -----
        Linear extrapolation of KBI is only performed for the ``weight_type='geometric'`` and provides the exact value in the thermodynamic limit.
        The relation to finite-volume KBI is given through:

        .. math::
            L G^V(L) = L G^\infty + F^\infty,

        where convergence is met if a linear region greater than 1.0 is identified and :math:`R^2 > 0.99`.
        """
        positions = positions or (min([1.0, self.r.max() - 1]), self.r.max())
        rmin = min(positions)  # if all(positions) else min([1.0, (self.r.max() - 1)])
        rmax = max(positions)  # if all(positions) else self.r.max()

        rkbi = self.running_kbi_map["geometric"]  # this method is only for 1 type of rkbi

        valid_mask = (self.r >= rmin) & (self.r <= rmax)
        valid_indices = np.where(valid_mask)[0]

        if len(valid_indices) == 0:
            raise ValueError(f"No data points with r > {rmin}")

        if not maximize_r2:
            slope, intercept, r_value, p_value, std_error = scipy.stats.linregress(
                self.r[valid_mask], self.r[valid_mask] * rkbi[valid_mask]
            )
            return {
                "G": slope,
                "F": intercept,
                "r2": r_value**2,
                "p_value": p_value,
                "std_error": std_error,
                "index_list": valid_indices.tolist(),
                "r_fit": self.r[valid_mask],
                "r_rkbi_pred": slope * self.r[valid_mask] + intercept,
            }

        # Define the last 10% range
        n_valid = len(valid_indices)
        last_10_percent_start_idx = valid_indices[int(0.9 * n_valid)]

        best_r2 = 0.0
        best_fit = None

        # Sample starting points (e.g., every 5th point or adaptive sampling)
        step = max(1, len(valid_indices) // 50)  # ~50 starting points max

        for i in range(0, len(valid_indices), step):
            start_idx = valid_indices[i]

            # For each start, try a few strategic end points in the last 10%
            end_candidates = valid_indices[valid_indices >= last_10_percent_start_idx][:: max(1, n_valid // 20)]

            for end_idx in end_candidates:
                if start_idx >= end_idx:
                    continue

                idx_range = valid_indices[(valid_indices >= start_idx) & (valid_indices <= end_idx)]

                # Check if r range > 1.0
                r_range = self.r[idx_range[-1]] - self.r[idx_range[0]]
                if r_range <= 1.0:
                    continue

                if len(idx_range) < 3:
                    continue

                slope, intercept, r_val, p_val, std_err = scipy.stats.linregress(
                    self.r[idx_range], self.r[idx_range] * rkbi[idx_range]
                )

                r2 = r_val**2
                if np.abs(r2) > best_r2:
                    best_r2 = np.abs(r2)
                    best_fit = {
                        "G": slope,
                        "F": intercept,
                        "r2": r2,
                        "p_value": p_val,
                        "std_error": std_err,
                        "index_list": idx_range.tolist(),
                        "r_fit": self.r[idx_range],
                        "r_rkbi_pred": slope * self.r[idx_range] + intercept,
                    }

        if best_fit is None:
            raise LinearityError("Could not find valid regression window with range > 1.0.")

        if best_r2 < 0.99:
            raise LinearityError("Could not find valid regression window with R^2 > 0.99.")

        return best_fit

    @cached_property
    def geometric_extrapolation_result(self) -> dict:
        """dict: Returns the result of the linear extrapolation using ``maximize_r2=True``."""
        return self.compute_geometric_extrapolation(maximize_r2=True)

    def compute_kbi(
        self,
        weight_type: str,
        raise_on_convergence_error: bool = True,
        force: bool = False,
    ) -> float:
        r"""Get KBI value in the thermodynamic limit according to ``weight_type``.

        Returns
        -------
        float
            KBI in the thermodynamic limit

        Notes
        -----
        For ``weight_type='geometric'``; KBI is compute via extrapolation to the thermodynamic limit. (see also: :meth:`compute_geometric_extrapolation`)

        .. math::
            L G^V(L) = L G^\\infty + F^\\infty

        For all other type of ``weight_type``, KBI is approximated by averaging the value of :math:`G$_\text{k}$(L)` over the last 0.1 nm.

        .. math::
            G^\\infty \approx G_k(L)
        """
        # get mask for rkbi tail average
        mask = (self.r > (self.r.max() - 0.1)) & (self.r < self.r.max())
        if weight_type.lower() == "geometric":
            try:
                return self.geometric_extrapolation_result["G"]
            except KBIConvergenceError as e:
                if force:
                    warnings.warn(f"KBI convergence failed: {e}", RuntimeWarning, stacklevel=2)
                    print("Falling back on KBI for `weight_type='u2'`.")
                    return float(np.nanmean(self.running_kbi_map["u2"][mask]))
                if raise_on_convergence_error:
                    raise
                else:
                    warnings.warn(f"KBI convergence failed: {e}", RuntimeWarning, stacklevel=2)
                    return np.nan

        # for all other types; just extract last values in running kbi
        return float(np.nanmean(self.running_kbi_map[weight_type.lower()][mask]))

    @cached_property
    def kbi(self) -> float:
        """float: KBI value in the thermodynamic limit for class initialized: ``weight_type``, ``raise_on_convergence_error``, ``force``."""
        return self.compute_kbi(
            weight_type=self.weight_type, raise_on_convergence_error=self.raise_on_convergence_error, force=self.force
        )

    def plotKBI(
        self, axhandle: plt.axes, weight_type: str, **kwargs
    ) -> None:
        """Add KBI as a function of radial distance to axes.

        Parameters
        ----------
        axhandle: plt.axes
            Axes to add plot.
        weight_type: str
            Type of weight function for finite-volume corrections. Options: ('none','u0','u1','u2','geometric')
        """
        axhandle.plot(self.r, self.running_kbi_map[weight_type], **kwargs)

    def plotLKBI(self, axhandle: plt.axes, **kwargs) -> None:
        """Add scaled KBI as a function of radial distance to axes, for plotting geometric linear extrapolation.

        Parameters
        ----------
        axhandle: plt.axes
            Axes to add plot.
        """
        axhandle.plot(self.r, self.r * self.running_kbi_map["geometric"], **kwargs)

    def plotKBIFit(self, axhandle: plt.axes, result: dict | None = None, **kwargs) -> None:
        """Add fit results from  geometric linear extrapolation to axes.

        Parameters
        ----------
        axhandle: plt.axes
            Axes to add plot.
        result: dict, optional
            Geometric extrapolation result.
        """
        result = result or self.geometric_extrapolation_result
        rfit = result["r_fit"]
        r_rkbi_pred = result["r_rkbi_pred"]
        axhandle.plot(rfit, r_rkbi_pred, **kwargs)

    def plotKBIValue(
        self, axhandle: plt.axes, weight_type: str, **kwargs
    ) -> None:
        """Add horizontal line for KBI in the thermodynamic limit for a given ``weight_type``.

        Parameters
        ----------
        axhandle: plt.axes
            Axes to add plot.
        weight_type: str
            Type of weight function for finite-volume corrections. Options: ('none','u0','u1','u2','geometric')
        """
        kbi = self.compute_kbi(weight_type, raise_on_convergence_error=False)
        if all(x not in kwargs for x in ("ls", "linestyle")):
            kwargs.update({"ls": (0, (10, 5, 2, 5))})
        axhandle.axhline(kbi, **kwargs)

    @property
    def _colors_by_weight(self) -> dict:
        return {
            "none": "gold",
            "u0": "cyan",
            "u1": "lawngreen",
            "u2": "red",
            "geometric": plt.colormaps["hsv"](np.linspace(0.12, 1, 5))[3],
        }

    def get_colors(self, cmap: str | None = None) -> dict:
        """dict: Get colors mapped to ``weight_type``."""
        if cmap is None:
            return self._colors_by_weight
        else:
            colors_from_cmap = plt.colormaps[cmap](np.linspace(0, 1, 5))
            return {weight: colors_from_cmap[i] for i, weight in enumerate(self._colors_by_weight)}

    def plotKBICompare(
        self,
        weight_types: list[Literal["none", "u0", "u1", "u2", "geometric"]],
        cmap: str | None = None,
        addKBIValue: bool = True,
        filepath: str | None = None,
    ) -> None:
        """Create figure for comparing KBI's from various correction methods.

        Parameters
        ----------
        weight_types: list[str]
            Weight functions to include in plot. Options: ('none','u0','u1','u2','geometric')
        cmap: str, optional
            Matplotlib colormap name.
        addKBIValue: bool, optional
            Add horizontal value for KBI with ``weight_type='geometric'``.
        filepath: str, optional
            Filepath to save figure as.
        """
        colors_by_weight = self.get_colors(cmap)

        fig, ax = plt.subplots()
        style_axes(ax, minorxticks=np.arange(0, self.r.max(), 0.5))
        for weight_type in weight_types:
            label = r"G"
            if weight_type.startswith("u"):
                label += rf"$_{weight_type[1]}^\infty$"
            elif weight_type == "geometric":
                label += r"$^\text{V}$"
            self.plotKBI(ax, weight_type=weight_type, lw=1, color=colors_by_weight[weight_type], label=label)
        if addKBIValue:
            self.plotKBIValue(ax, weight_type="geometric", color="k", lw=0.5, ls="-", label=r"G$^\infty$")

        style_legend(ax)
        ax.set_ylabel(r"G(L) (nm$^3$)", fontsize=16)
        ax.set_xlabel(r"L (nm)", fontsize=16)
        if filepath:
            fig.savefig(filepath, dpi=100)
            plt.close()
        else:
            plt.show()

    def plotKBIExtrap(
        self,
        result: dict | None = None,
        color: str = "turquoise",
        linewidth: float = 1.0,
        fitcolor: str = "k",
        fitstyle: tuple | str = (0, (3, 2)),
        fitwidth: float = 2,
        filepath: str | None = None,
    ) -> None:
        """Create figure for evaluating geometric extrapolation.

        Parameters
        ----------
        result: dict, optional
            Geometric extrapolation result.
        color: str, optional
            Color for KBI value.
        linewidth: float, optional
            Linewidth for plotting KBI value.
        fitcolor: str, optional
            Color for linear fit.
        fitstyle: tuple | str, optional
            Linestyle for linear fit.
        fitwidth: float, optional
            Linewidth for linear fit.
        filepath: str, optional
            Filepath to save figure as.
        """
        result = result or self.geometric_extrapolation_result
        G = result["G"]
        r2 = result["r2"]

        fig, ax = plt.subplots()
        style_axes(ax, minorxticks=np.arange(0, self.r.max(), 0.5))
        self.plotLKBI(ax, c=color, lw=linewidth)
        self.plotKBIFit(ax, result, c=fitcolor, lw=fitwidth, ls=fitstyle, label=rf"G$^\infty$={G:.4f} ($r^2$={r2:.4f})")
        style_legend(ax, linewidth=1.5)
        ax.set_xlabel(r"L")
        ax.set_ylabel(r"LG$^\text{V}$")
        if filepath:
            fig.savefig(filepath, dpi=100)
            plt.close()
        else:
            plt.show()

    def plotKBICompareExtrap(
        self,
        weight_types: list[Literal["none", "u0", "u1", "u2", "geometric"]],
        cmap: str | None = None,
        addKBIValue: bool = True,
        filepath: str | None = None,
    ):
        """Combine KBI comparisons of ``weight_type`` and geometric extrapolation on a single plot.

        Parameters
        ----------
        weight_types: list[str]
            Weight functions to include in plot. Options: ('none','u0','u1','u2','geometric')
        cmap: str, optional
            Matplotlib colormap name.
        addKBIValue: bool, optional
            Add horizontal value for KBI with ``weight_type='geometric'``.
        filepath: str, optional
            Filepath to save figure as.
        """
        colors_by_weight = self.get_colors(cmap)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), width_ratios=[2, 1])
        for weight_type in weight_types:
            label = r"G"
            if weight_type.startswith("u"):
                label += rf"$_{weight_type[1]}^\infty$"
            elif weight_type == "geometric":
                label += r"$^\text{V}$"
            self.plotKBI(ax1, weight_type=weight_type, lw=1, color=colors_by_weight[weight_type], label=label)
        if addKBIValue:
            self.plotKBIValue(ax1, weight_type="geometric", color="k", lw=0.5, ls="-", label=r"G$^\infty$")

        self.plotLKBI(ax2, color=colors_by_weight["geometric"], alpha=0.8, lw=1)
        self.plotKBIFit(
            ax2, color="k", ls=(0, (3, 2)), lw=2.5, label=rf"G$^\infty$={self.compute_kbi('geometric'):.2e}"
        )

        style_axes(ax1, minorxticks=np.arange(0, self.r.max(), 0.5))
        style_axes(ax2, minorxticks=np.arange(0, self.r.max(), 1))
        style_legend(ax1, ncol=2, linewidth=1.5, linelength=1.5)
        style_legend(ax2, ncol=1, linewidth=1.5, linelength=1)
        ax1.set_xlabel("L (nm)")
        ax1.set_ylabel(r"G(L) (nm$^3$)")
        ax2.set_xlabel("L (nm)")
        ax2.set_ylabel(r"LG$^\text{V}$ (nm $\cdot$ nm$^3$)")
        if filepath:
            fig.savefig(filepath, dpi=100)
            plt.close()
        else:
            plt.show()
