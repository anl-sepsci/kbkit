"""
Computes Kirkwood-Buff integrals (KBIs) from RDF data and applies thermodynamic limit corrections.

Relies on RDF parsing and system composition data to produce corrected KBIs for use in thermodynamic models.
"""

import os
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from scipy.integrate import cumulative_trapezoid

from kbkit.core.system_properties import SystemProperties
from kbkit.parsers.rdf_file import RDFParser


class KBIntegrator:
    """
    Class to compute the Kirkwood-Buff Integrals (KBI) from RDF data.

    Parameters
    ----------
    rdf_file : str
        Path to the RDF file containing radial distances and corresponding g(r) values.
    use_fixed_rmin : bool
        Whether to use a fixed minimum distance (rmin) for analysis.
    system_properties : SystemProperties
        SystemProperties object containing information about the system, including topology and box dimensions.

    Attributes
    ----------
    rdf: RDFParser
        RDFParser object for parsing RDF file.
    system_properties: SystemProperties
        SystemProperties object.
    """

    def __init__(self, rdf_file: str | Path, use_fixed_rmin: bool, system_properties: SystemProperties) -> None:
        self.rdf = RDFParser(rdf_file=str(rdf_file), use_fixed_rmin=use_fixed_rmin)
        self.system_properties = system_properties

    def box_vol(self) -> float:
        """Return the volume of the system box in nm^3."""
        vol = self.system_properties.get("volume", units="nm^3")
        if isinstance(vol, tuple):
            vol = vol[0]
        return float(vol)

    def rdf_molecules(self) -> list[str]:
        """Get the molecules corresponding to the RDF file from the system topology.

        Returns
        -------
        list
            List of molecule IDs corresponding to the RDF file.
        """
        # extract molecules from file name and topology information
        rdf_mols = RDFParser.extract_mols(Path(self.rdf.rdf_file).name, self.system_properties.topology.molecules)
        # check length of molecules found --- must be two for rdfs
        N_RDF_MOLS = 2
        if len(rdf_mols) != N_RDF_MOLS:
            raise ValueError("Number of molecules corresponding to ID in .top file is not 2!")
        return rdf_mols

    def kd(self) -> int:
        """Return the Kronecker delta between molecules in RDF, i.e., determines if molecules :math:`i,j` are the same."""
        return int(self.rdf_molecules()[0] == self.rdf_molecules()[1])

    def n_j(self) -> int:
        """Return the number of molecule :math:`j` in the system."""
        return self.system_properties.topology.molecule_count[self.rdf_molecules()[1]]

    def g_gv(self) -> NDArray[np.float64]:
        r"""
        Compute the corrected pair distribution function, accounting for finite-size effects in the simulation box, based on the approach by `Ganguly and Van der Vegt (2013) <https://doi.org/10.1021/ct301017q>`_.

        Returns
        -------
        np.ndarray
            Corrected g(r) values as a numpy array corresponding to distances `r` from the RDF.

        Notes
        -----
        The correction is calculated as

        .. math::
            v_r = 1 - \frac{\frac{4}{3} \pi r^3}{V}

        .. math::
            \rho_j = \frac{N_j}{V}

        .. math::
            \Delta N_j = \int_0^r 4 \pi r^2 \rho_j \bigl(g(r) - 1 \bigr) \, dr

        .. math::
            g_{GV}(r) = g(r) \cdot \frac{N_j v_r}{N_j v_r - \Delta N_j - \delta_{ij}}


        where:
         - :math:`r` is the distance
         - :math:`V` is the box volume
         - :math:`N_j` is the number of particles of type \( j \)
         - :math:`g(r)` is the raw radial distribution function
         - :math:`\delta_{ij}` is a kronecker delta

        .. note::
            The cumulative integral :math:`\Delta N_j` is approximated numerically using the trapezoidal rule.
        """
        # calculate the reduced volume
        vr = 1 - ((4 / 3) * np.pi * self.rdf.r**3 / self.box_vol())

        # get the number density for molecule j
        rho_j = self.n_j() / self.box_vol()

        # function to integrate over
        f = 4.0 * np.pi * self.rdf.r**2 * rho_j * (self.rdf.g - 1)
        Delta_Nj = cumulative_trapezoid(f, x=self.rdf.r, dx=self.rdf.r[1] - self.rdf.r[0])
        Delta_Nj = np.append(Delta_Nj, Delta_Nj[-1])

        # correct g(r) with GV correction
        g_gv = self.rdf.g * self.n_j() * vr / (self.n_j() * vr - Delta_Nj - self.kd())
        return np.asarray(g_gv)  # make sure that an array is returned

    def window(self) -> NDArray[np.float64]:
        r"""
        Apply cubic correction (or window weight) to the radial distribution function, which is useful for ensuring that the integral converges properly at larger distances, based on the method described by `Krüger et al. (2013) <https://doi.org/10.1021/jz301992u>`_.

        Returns
        -------
        np.ndarray
            Windowed weight for the RDF

        Notes
        -----
        The windowed weight is defined as:

        .. math::
            w(r) = 4 \pi r^2 \left(1 - \left(\frac{r}{r_{max}}\right)^3\right)

        where:
            - :math:`r` is the radial distance
            - :math:`r_{max}` is the maximum radial distance in the RDF
        """
        w = 4 * np.pi * self.rdf.r**2 * (1 - (self.rdf.r / self.rdf.rmax) ** 3)
        return np.asarray(w)

    def h(self) -> NDArray[np.float64]:
        r"""
        Calculate correlation function h(r) from the corrected g(r) values.

        Returns
        -------
        np.ndarray
            Correlation function h(r) as a numpy array.

        Notes
        -----
        The correlation function is defined as:

        .. math::
            h(r) = g_{GV}(r) - 1

        """
        return self.g_gv() - 1

    def rkbi(self) -> NDArray[np.float64]:
        r"""
        Compute KBI as a function of radial distance between molecules :math:`i` and :math:`j`.

        Returns
        -------
        np.ndarray
            KBI values as a numpy array corresponding to distances :math:`r` from the RDF.

        Notes
        -----
        The KBI is computed using the formula:

        .. math::
            G_{ij}(r) = \int_0^r h(r) w(r) dr

        where:
            - :math:`h(r)` is the correlation function
            - :math:`w(r)` is the window function
            - :math:`r` is the radial distance

        .. note::
            The integration is performed using the trapezoidal rule.
        """
        rkbi_arr = cumulative_trapezoid(self.window() * self.h(), self.rdf.r, initial=0)
        return np.asarray(rkbi_arr)

    def lambda_ratio(self) -> NDArray[np.float64]:
        r"""
        Calculate length ratio (:math:`\lambda`) of the system based on the radial distances and the box volume.

        Returns
        -------
        np.ndarray
            Length ratio as a numpy array corresponding to distances :math:`r` from the RDF.

        Notes
        -----
        The length ratio is defined as:

        .. math::
            \lambda = \left(\frac{\frac{4}{3} \pi r^3}{V}\right)^{1/3}

        where:
            - :math:`r` is the radial distance
            - :math:`V` is the box volume
        """
        Vr = (4 / 3) * np.pi * self.rdf.r**3 / self.box_vol()
        return Vr ** (1 / 3)

    def fit_kbi_inf(self) -> NDArray[np.float64]:
        r"""
        Fit a linear model to the product of the length ratio and the KBI values for extrapolation to thermodynamic limit.

        Returns
        -------
        tuple
            Tuple containing the slope and intercept of the linear fit, which represents the KBI at infinite distance.


        .. note::
            The KBI at infinite distance is estimated by fitting a linear model to the product of the length ratio and the KBI values, using only the radial distances that are within the specified range (rmin to rmax).
        """
        # get x and y values to fit thermodynamic correction
        lam = self.lambda_ratio()  # characteristic length
        lam_kbi = lam * self.rkbi()  # length x KBI (r)

        # fit linear regression to masked values
        fit_params = np.polyfit(lam[self.rdf.r_mask], lam_kbi[self.rdf.r_mask], 1)
        return fit_params  # return fit

    def integrate(self) -> float:
        """
        Compute KBI in thermodynamic limit.

        Returns
        -------
        float
            KBI in the thermodynamic limit, which is the slope of the linear fit to the product
            of the length ratio and the KBI values.
        """
        return float(self.fit_kbi_inf()[0])

    def plot(self, save_dir: Optional[str] = None) -> None:
        """Plot RDF and the running KBI fit to thermodynamic limit.

        Parameters
        ----------
        save_dir : str, optional
            Directory to save the plot. If not provided, the plot will be displayed but not saved
        """
        # get running kbi
        rkbi = self.rkbi()
        # parameters for thermo-limit extrapolation
        lam = self.lambda_ratio()
        lam_rkbi = lam * rkbi
        # fits to thermo limit
        fit_params = self.fit_kbi_inf()
        lam_fit = lam[self.rdf.r_mask]
        lam_rkbi_fit = np.polyval(fit_params, lam_fit)

        fig, ax = plt.subplots(1, 2, figsize=(9, 4))
        ax[0].plot(self.rdf.r, rkbi)
        ax[0].set_xlabel("r / nm")
        ax[0].set_ylabel("G$_{ij}$ / nm$^3$")
        ax[1].plot(lam, lam_rkbi)
        ax[1].plot(lam_fit, lam_rkbi_fit, ls="--", c="k", label=f"KBI: {fit_params[0]:.2g} nm$^3$")
        ax[1].set_xlabel(r"$\lambda$")
        ax[1].set_ylabel(r"$\lambda$ G$_{ij}$ / nm$^3$")
        fig.suptitle(
            f"KBI Analysis for system: {os.path.basename(self.system_properties.system_path)} {'-'.join(self.rdf_molecules())}"
        )
        if save_dir is not None:
            plt.savefig(os.path.join(save_dir, self.rdf.rdf_file[:-4] + ".png"))
        plt.show()
