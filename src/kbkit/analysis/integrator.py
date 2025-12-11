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

from kbkit.config.mplstyle import load_mplstyle
from kbkit.parsers.rdf_file import RDFParser
from kbkit.systems.properties import SystemProperties

load_mplstyle()


class KBIntegrator:
    """
    Class to compute the Kirkwood-Buff Integrals (KBI) from RDF data.

    Parameters
    ----------
    rdf_file : str
        Path to the RDF file containing radial distances and corresponding g(r) values.
    system_properties : SystemProperties
        SystemProperties object containing information about the system, including topology and box dimensions.
    use_fixed_rmin : bool, optional
        Whether to use a fixed minimum distance (rmin) for analysis (default: False).
    converged_threshold : float, optional
        Value of the slope of RDF tail for the RDF to be considered as converged (default: 0.005).
    correct_rdf_convergence: bool, optional
        Whether to correct RDF for excess/depletion, i.e., Ganguly correction (default: True).
    apply_damping: bool, optional
        Whether to apply damping function to correlation function, i.e., Kruger correction (default: True).
    extrapolate_thermodynamic_limit: bool, optional
        Whether to extrapolate KBI value to the thermodynamic limit (default: True).

    Attributes
    ----------
    rdf: RDFParser
        RDFParser object for parsing RDF file.
    system_properties: SystemProperties
        SystemProperties object.
    """

    def __init__(
        self,
        rdf_file: str | Path,
        system_properties: SystemProperties,
        use_fixed_rmin: bool = False,
        convergence_threshold: float = 0.005,
        correct_rdf_convergence: bool = True,
        apply_damping: bool = True,
        extrapolate_thermodynamic_limit: bool = True
    ) -> None:
        self.rdf = RDFParser(
            rdf_file=rdf_file, use_fixed_rmin=use_fixed_rmin, convergence_threshold=convergence_threshold
        )
        self.system_properties = system_properties
        self.correct_rdf_convergence = correct_rdf_convergence
        self.apply_damping = apply_damping
        self.extrapolate_thermodynamic_limit = extrapolate_thermodynamic_limit

    def box_volume(self) -> float:
        """Return the volume of the system box in nm^3."""
        vol = self.system_properties.get("volume", units="nm^3")
        if isinstance(vol, tuple):
            vol = vol[0]
        return float(vol)

    @property
    def rdf_molecules(self) -> list[str]:
        """Get the molecules corresponding to the RDF file from the system topology.

        Returns
        -------
        list
            List of molecule IDs used in RDF file.
        """
        molecules = RDFParser.extract_molecules(
            text=self.rdf.rdf_file.name, mol_list=self.system_properties.topology.molecules
        )
        MAGIC_TWO = 2
        if len(molecules) != MAGIC_TWO:
            raise ValueError(
                f"Number of molecules detected in RDF calculation is '{len(molecules)}', expected 2. Check that filname is appropriately named."
            )
        return molecules

    @property
    def _mol_j(self) -> str:
        """Returns second molecule in `rdf_molecules` as default options."""
        return self.rdf_molecules[1]

    def kronecker_delta(self) -> int:
        """Return the Kronecker delta between molecules in RDF, i.e., determines if molecules :math:`i,j` are the same (returns True)."""
        return int(self.rdf_molecules[0] == self.rdf_molecules[1])

    def _validate_molecule(self, mol: str) -> None:
        """Validate molecule to be used in RDF integration for coordination number calculation."""
        if len(mol) == 0:
            raise ValueError(f"Molecule '{mol}' cannot be empty str!")
        elif mol not in self.rdf_molecules:
            raise ValueError(f"Molecule '{mol}' not in rdf molecules '{self.rdf_molecules}'.")

    def n_j(self, mol_j: str = "") -> int:
        r"""Number of molecule :math:`j` in the system.

        Parameters
        ----------
        mol_j: str, optional
            Molecule :math:`j` used for g(r) correction (Ganguly). Defaults to second molecule in RDF filename.

        Returns
        -------
        int
            Number of molecules :math:`j` in the system.
        """
        if len(mol_j) == 0:
            mol_j = self._mol_j
        # validate molecule
        self._validate_molecule(mol_j)
        # compute molecule number
        return self.system_properties.topology.molecule_count[mol_j]

    def ganguly_correction_factor(self, mol_j: str = "") -> NDArray[np.float64]:
        r"""
        Compute the corrected radial distribution function, accounting for finite-size effects in the simulation box, based on the approach by `Ganguly and Van der Vegt (2013) <https://doi.org/10.1021/ct301017q>`_.

        Parameters
        ----------
        mol_j: str, optional
            Molecule :math:`j` used for g(r) correction (Ganguly). Defaults to second molecule in RDF filename.

        Returns
        -------
        np.ndarray
            Corrected g(r) values as a numpy array corresponding to distances `r` from the RDF.

        Notes
        -----
        The correction is calculated as

        .. math::
            g^{Ganguly}(r) = g(r) \left[ \frac{N_j f(r)}{N_j f(r) - \Delta N_j(r) - \delta_{ij}} \right]

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
         - :math:`g(r)` is the raw radial distribution function
         - :math:`\delta_{ij}` is a kronecker delta

        .. note::
            The cumulative integral :math:`\Delta N_j(r)` is approximated numerically using the trapezoidal rule.
        """
        # calculate the reduced volume
        vr = 1 - ((4 / 3) * np.pi * self.rdf.r**3 / self.box_volume())

        # get the number density for Molecule :math:`j`
        Nj = self.n_j(mol_j)
        rho_j = Nj / self.box_volume()

        # function to integrate over
        f = 4.0 * np.pi * self.rdf.r**2 * rho_j * (self.rdf.g - 1)
        Delta_Nj = cumulative_trapezoid(f, x=self.rdf.r, dx=self.rdf.r[1] - self.rdf.r[0])
        Delta_Nj = np.append(Delta_Nj, Delta_Nj[-1])

        # correct g(r) with GV correction
        g_gv = (Nj * vr / (Nj * vr - Delta_Nj - self.kronecker_delta()))
        return np.asarray(g_gv)  # make sure that an array is returned

    def kruger_damping_factor(self) -> NDArray[np.float64]:
        r"""
        Damp the radial distribution function, which is useful for ensuring that the integral converges properly at larger distances, based on the method described by `Krüger et al. (2013) <https://doi.org/10.1021/jz301992u>`_.

        Returns
        -------
        np.ndarray
           Damping factor for the RDF

        Notes
        -----
        The damping factor, :math:`\omega(r)`, is defined as:

        .. math::
            \omega(r) = \left[1 - \left(\frac{r}{r_{max}}\right)^3\right]

        where:
            - :math:`r` is the radial distance
            - :math:`r_{max}` is the maximum radial distance of :math:`r`
        """
        return np.asarray(1 - (self.rdf.r / self.rdf.rmax) ** 3)

    def h(self, mol_j: str = "") -> NDArray[np.float64]:
        r"""
        Calculate correlation function h(r) from g(r). 

        If ``correct_rdf_convergence`` is `True`, Ganguly correction factor is applied.

        Parameters
        ----------
        mol_j: str, optional
            Molecule :math:`j` used for g(r) correction (Ganguly). Defaults to second molecule in RDF filename.

        Returns
        -------
        np.ndarray
            Correlation function h(r) as a numpy array.

        Notes
        -----
        The correlation function is defined as:

        .. math::
            h(r) = g(r) - 1

        """
        # apply necessary corrections; here is where ganguly is applied
        if self.correct_rdf_convergence:
            return self.ganguly_correction_factor(mol_j=mol_j) * self.rdf.g - 1
        else:
            return self.rdf.g - 1

    def rkbi(self, mol_j: str = "") -> NDArray[np.float64]:
        r"""
        Compute KBI as a function of radial distance between molecules :math:`i` and :math:`j`, i.e., running KBI (RKBI).

        Takes in the correlation function, and applies Kruger damping function if ``apply_damping`` is set to True.

        Parameters
        ----------
        mol_j: str, optional
            Molecule :math:`j` used for g(r) correction (Ganguly). Defaults to second molecule in RDF filename.

        Returns
        -------
        np.ndarray
            KBI values as a numpy array corresponding to distances :math:`r` from the RDF.

        Notes
        -----
        The KBI is computed using the formula:

        .. math::
            G_{ij}^R = \int_0^R 4 \pi r^2 \omega(r) h(r) dr

        where:
            - :math:`h(r)` is the correlation function
            - :math:`\omega(r)` is the damp factor, this is set to 1 if damping is not desired.
            - :math:`r` is the radial distance

        .. note::
            The integration is performed using the trapezoidal rule.
        """
        integrand = 4 * np.pi * self.rdf.r**2 * self.h(mol_j)
        corrected_integrand = self.kruger_damping_factor() * integrand if self.apply_damping else integrand
        rkbi_arr = cumulative_trapezoid(corrected_integrand, self.rdf.r, initial=0)
        return np.asarray(rkbi_arr)
    
    def _compute_rkbi(self, mol_j: str = "", correct_rdf_convergence: bool = True, apply_damping: bool = True):
        r"""Enables comparison of various running KBIs."""
        g = self.ganguly_correction_factor(mol_j=mol_j) * self.rdf.g if correct_rdf_convergence else self.rdf.g 
        omega = self.kruger_damping_factor() if apply_damping else 1
        integrand = 4 * np.pi * self.rdf.r**2 * omega * (g - 1)
        rkbi_arr = cumulative_trapezoid(integrand, self.rdf.r, initial=0)
        return np.asarray(rkbi_arr)

    def scaled_rkbi(self, mol_j: str = "") -> NDArray[np.float64]:
        r"""Product of R and KBI values from 0 \to R.

        Parameters
        ----------
        mol_j: str, optional
            Molecule :math:`j` used for g(r) correction (Ganguly). Defaults to second molecule in RDF filename.

        Returns
        -------
        np.ndarray
            R x running KBI corresponding to distances :math:`r` from the RDF.
        """
        return self.rdf.r * self.rkbi(mol_j)

    def scaled_rkbi_fit(self, mol_j: str = "") -> NDArray[np.float64]:
        r"""Compute the product of R and KBI values from :math:`0 \rightarrow R` in the range of [:math:`r_{min}`, :math:`r_{max}`].

        Parameters
        ----------
        mol_j: str, optional
            Molecule :math:`j` used for g(r) correction (Ganguly). Defaults to second molecule in RDF filename.

        Returns
        -------
        np.ndarray
            R x running KBI corresponding to distances :math:`r_{min} \rightarrow r_{max}` from the RDF.
        """
        return self.scaled_rkbi(mol_j)[self.rdf.r_mask]

    def fit_limit_params(self, mol_j: str = "") -> NDArray[np.float64]:
        r"""
        Fit a linear regression to the product of R and the running KBI values for extrapolation to thermodynamic limit.

        Parameters
        ----------
        mol_j: str, optional
            Molecule :math:`j` used for g(r) correction (Ganguly). Defaults to second molecule in RDF filename.

        Returns
        -------
        tuple
            Tuple containing the slope and intercept of the linear fit, which represents the KBI and surface term in the thermodynamic limit, respectfully.

        Notes
        -----
        The KBI in thermodynamic limit, :math:`G_{ij}^\infty`, is calculated according to:

        .. math::
            R G_{ij}^R = R G_{ij}^\infty + F_{ij}^\infty

        where :math:`F_{ij}^\infty` is a finite-size surface offset.

        .. note::
            The KBI at infinite distance is estimated by fitting a linear model to the product of r and the KBI values, using only the radial distances that are within the specified range [:math:`r_{min}, r_{max}`].
        """
        # fit linear regression to masked values
        return np.polyfit(self.rdf.r_fit, self.scaled_rkbi_fit(mol_j), 1)
    
    def compute_kbi(self, mol_j: str = "") -> float:
        r"""Compute KBI according the specified corrections.
        
        If ``extrapolate_thermodynamic_limit`` is set to `True`, extrapolate the KBI to the thermodynamic limit with linear regression.
        Otherwise, get the average of the tail of the running KBI.

        
        Parameters
        ----------
        mol_j: str, optional
            Molecule :math:`j` used for g(r) correction (Ganguly). Defaults to second molecule in RDF filename.

        Returns
        -------
        float
            KBI
        """
        if self.extrapolate_thermodynamic_limit:
            return float(self.fit_limit_params(mol_j)[0])
        else:
            return self.rkbi(mol_j=mol_j)[self.rdf.r_mask].mean()
        
    def plot_rkbis(self, mol_j: str = "", save_dir: Optional[str] = None) -> None:
        """Plot various types of running KBIs. Includes raw (no corrections), only Ganguly correction, and Ganguly + Kruger correction.
        
        Parameters
        ----------
        mol_j: str, optional
            Molecule :math:`j` used for g(r) correction (Ganguly). Defaults to second molecule in RDF filename.
        save_dir: str, optional
            Directory to save the plot. If not provided, the plot will be displayed but not saved
        """
        raw_rkbi = self._compute_rkbi(correct_rdf_convergence=False, apply_damping=False)
        g_rkbi = self._compute_rkbi(correct_rdf_convergence=True, apply_damping=False)
        gk_rkbi = self._compute_rkbi(correct_rdf_convergence=True, apply_damping=True)

        fig, ax = plt.subplots(figsize=(5,4.5))
        ax.plot(self.rdf.r, raw_rkbi, c="limegreen", alpha=0.6, lw=3, ls="-", label="no corrections")
        ax.plot(self.rdf.r, g_rkbi, c="tomato", lw=3, ls="-", label="convergence correction")
        ax.plot(self.rdf.r, gk_rkbi, c="skyblue", lw=3, ls="-", label="convergence + damping correction")
        ax.set_xlabel(r"$r$ [$nm$]")
        ax.set_ylabel(r"$\int_0^R 4 \pi r^2 \ \omega (r) \ [g(r) - 1]$")
        ax.legend(
            bbox_to_anchor=(0., 1.02, 1., .102), 
            loc='lower left', 
            mode='expand',
            borderaxespad=0., 
        )

        if save_dir is not None:
            mols = "_".join(self.rdf_molecules)
            fig.savefig(os.path.join(save_dir, f"rkbis_{mols}.pdf"), dpi=100)
        plt.show()


    def plot_integrand(self, mol_j: str = "", save_dir: Optional[str] = None) -> None:
        """
        Plot RDF and integrand for running KBI calculation. Includes demonstrating the effect of damping on the integrand.

        Parameters
        ----------
        mol_j: str, optional
            Molecule :math:`j` used for g(r) correction (Ganguly). Defaults to second molecule in RDF filename.
        save_dir: str, optional
            Directory to save the plot. If not provided, the plot will be displayed but not saved
        """
        A = 4 * np.pi * self.rdf.r**2
        integrand_gv = A * (self.ganguly_correction_factor(mol_j) * self.rdf.g - 1)
        integrand_damp = self.kruger_damping_factor() * integrand_gv

        fig, ax = plt.subplots(1, 2, figsize=(7.5, 3.5), sharex=True)
        ax[0].plot(self.rdf.r, self.rdf.g, c="tomato", label="-".join(self.rdf_molecules))
        ax[0].set_xlabel(r"$r$ [$nm$]")
        ax[0].set_ylabel(r"$g(r)$")
        ax[0].legend()

        ax[1].plot(self.rdf.r, integrand_gv, c="tomato", label="undamped")
        ax[1].plot(self.rdf.r, integrand_damp, c="k", alpha=0.65, ls="--", label="damped")
        ax[1].set_xlabel(r"$R$ [$nm$]")
        ax[1].set_ylabel(r"$4 \pi r^2 \ [g(r) - 1]$")
        ax[1].legend()

        if save_dir is not None:
            mols = "_".join(self.rdf_molecules)
            fig.savefig(os.path.join(save_dir, f"kbi_integrand_{mols}.pdf"), dpi=100)
        plt.show()

    def plot_extrapolation(self, mol_j: str = "", save_dir: Optional[str] = None):
        """Plot RDF and the running KBI fit to thermodynamic limit.

        Parameters
        ----------
        mol_j: str, optional
            Molecule :math:`j` used for g(r) correction (Ganguly). Defaults to second molecule in RDF filename.
        save_dir : str, optional
            Directory to save the plot. If not provided, the plot will be displayed but not saved.
        """
        label = "-".join(self.rdf_molecules)

        fig, ax = plt.subplots(1, 3, figsize=(12, 3.6), sharex=True)
        ax[0].plot(self.rdf.r, self.rdf.g, c="tomato", label=label)
        ax[0].set_xlabel(r"$r$ [$nm$]")
        ax[0].set_ylabel(r"$g(r)$")
        ax[0].legend()

        ax[1].plot(self.rdf.r, self.rkbi(mol_j), c="tomato")
        ax[1].set_xlabel(r"$R$ [$nm$]")
        ax[1].set_ylabel(r"$G_{{ij}}^R$ [$nm^3$]")

        ax[2].plot(self.rdf.r, self.scaled_rkbi(mol_j), c="tomato")
        kbi_inf = self.fit_limit_params(mol_j)[0]
        ax[2].plot(
            self.rdf.r_fit, self.scaled_rkbi_fit(mol_j), c="k", ls="--", lw=3, label=rf"G_{{ij}}^\infty={kbi_inf:.3f}"
        )
        ax[2].set_xlabel(r"$R$ [$nm$]")
        ax[2].set_ylabel(r"$R \ G_{{ij}}^R$ [$nm^4$]")

        if save_dir is not None:
            mols = "_".join(self.rdf_molecules)
            fig.savefig(os.path.join(save_dir, f"kbi_extrapolation_{mols}.pdf"), dpi=100)
        plt.show()
