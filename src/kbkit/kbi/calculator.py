"""
Calculator for Kirkwood-Buff Integrals (KBIs) as a function of composition.

This calculator operates on a :class:`~kbkit.systems.collection.SystemCollection` that contains molecular dynamics properties from structure and energy files.
Additional inputs are key parameters used for the KBI corrections provided in :class:`~kbkit.kbi.integrator.KBIntegrator`.
"""

import itertools
from typing import TYPE_CHECKING

import numpy as np

from kbkit.io.rdf import RdfParser
from kbkit.kbi.integrator import KBIntegrator
from kbkit.schema.kbi_metadata import KBIMetadata
from kbkit.schema.property_result import PropertyResult
from kbkit.visualization.kbi import KBIAnalysisPlotter

if TYPE_CHECKING:
    from kbkit.systems.collection import SystemCollection


class KBICalculator:
    """KBI calculator for system collections.

    Parameters
    ----------
    systems: SystemCollection
        SystemCollection object for set of systems.
    ignore_convergence_errors : bool, optional
        If True, ingnores convergence errors and forces KBI calculations to skip entire systems with non-converged RDFs.
    convergence_thresholds: tuple[float, float], optional
        Thresholds for convergence requirements of RDF tail.
    tail_length: float, optional
        Length of RDF tail (nm) to use for convergence evaluation & KBI corrections. If this is set, no iteration to find maximum length for RDF convergence will be performed.
    correct_rdf_convergence: bool, optional
        Whether to correct RDF for excess/depletion, i.e., Ganguly correction.
    apply_damping: bool, optional
        Whether to apply damping function to correlation function, i.e., Kruger correction.
    extrapolate_thermodynamic_limit: bool, optional
        Whether to extrapolate KBI value to the thermodynamic limit.
    """

    def __init__(
            self,
            systems: "SystemCollection",
            ignore_convergence_errors: bool = False,
            convergence_thresholds: tuple = (1e-3, 1e-2),
            tail_length: float | None = None,
            correct_rdf_convergence: bool = True,
            apply_damping: bool = True,
            extrapolate_thermodynamic_limit: bool = True,
    ) -> None:
        self.systems = systems
        self.ignore_convergence_errors=ignore_convergence_errors
        self.convergence_thresholds=convergence_thresholds
        self.tail_length=tail_length
        self.correct_rdf_convergence=correct_rdf_convergence
        self.apply_damping=apply_damping
        self.extrapolate_thermodynamic_limit=extrapolate_thermodynamic_limit

        self._cache: dict[tuple, PropertyResult] = {}

    def kbi(self, units: str = "nm^3/molecule") -> PropertyResult:
        r"""
        Computes Kirkwood-Buff integrals for molecular systems, if ``charges`` are present in :class:`~kbkit.systems.collection.SystemCollection` return :meth:`electrolyte_kbi` otherwise :meth:`residue_kbi` is returned.

        Parameters
        ----------
        units: str, optional
            Units to compute KBI in, molar volume units.

        Returns
        -------
        PropertyResult
            KBI Matrix with shape (composition x components x components).
        """
        return self.electrolyte_kbi(units) if self.systems.charges else self.residue_kbi(units)


    def residue_kbi(self, units: str = "nm^3/molecule") -> PropertyResult:
        r"""
        Computes Kirkwood-Buff integrals for molecular systems using RDF data.

        Interfaces with RdfParser and KBIntegrator to extract pairwise KBIs and populate metadata.

        Parameters
        ----------
        units: str, optional
            Units to compute KBI in, molar volume units.

        Returns
        -------
        PropertyResult
            KBI Matrix with shape (composition x components x components).

        Notes
        -----
        KBIs between components :math:`i, j` are calculated according to:

        .. math::
            G_{ij}^{\infty} = \int_0^{\infty} 4 \pi r^2 (g_{ij}(r) - 1) dr

        .. notes::
            * If an RDF directory is missing, the corresponding system's values remain NaN, if ignore_convergence_errors is True.
            * Populates `metadata` with integration results for each RDF file.

        See Also
        --------
        :class:`~kbkit.kbi.integrator.KBIntegrator` for a detailed description of KBI calculations and corrections.
        """
        units = units or "nm^3/molecule"

        # first check if cached
        cache_key = ("kbi",)
        if cache_key in self._cache:
            return self._cache[cache_key].to(units)

        # kbis are calculated in nm^3/molecule
        kbis = np.full((len(self.systems), len(self.systems.residue_molecules), len(self.systems.residue_molecules)), fill_value=np.nan)
        kbi_metadata: dict[str, dict[str, KBIMetadata]] = {}

        for s, meta in enumerate(self.systems):
            if not meta.has_rdf():
                continue
            # get all RDF files
            all_files = sorted(meta.rdf_path.iterdir())
            rdf_files = [f for f in all_files if f.suffix in (".xvg", ".txt")]

            for fpath in rdf_files:
                rdf = RdfParser(path=fpath, convergence_thresholds=self.convergence_thresholds, tail_length=self.tail_length)

                integrator = KBIntegrator.from_system_properties(
                    rdf=rdf,
                    system_properties=meta.props,
                    correct_rdf_convergence=self.correct_rdf_convergence,
                    apply_damping=self.apply_damping,
                    extrapolate_thermodynamic_limit=self.extrapolate_thermodynamic_limit,
                )

                mol_i, mol_j = integrator.rdf_molecules
                i, j = [list(self.systems.residue_molecules).index(mol) for mol in integrator.rdf_molecules]

                if rdf.is_converged:
                    kbis[s, i, j] = integrator.compute_kbi(mol_j)
                    kbis[s, j, i] = integrator.compute_kbi(mol_i)

                # override convergence check to skip system if not converged
                else:  # for not converged rdf
                    msg = f"RDF for system '{meta.name}' and pair {integrator.rdf_molecules} did not converge."
                    if self.ignore_convergence_errors:
                        print(f"WARNING: {msg} Skipping this system.")
                        continue
                    else:
                        raise RuntimeError(msg)

                # add values to metadata
                kbi_metadata.setdefault(meta.name, {})[".".join(integrator.rdf_molecules)] = KBIMetadata(
                    mols=tuple(integrator.rdf_molecules),
                    r=rdf.r,
                    g=rdf.g,
                    rkbi=(integrator.rkbi()),
                    scaled_rkbi=(integrator.scaled_rkbi()),
                    r_fit=(rfit := rdf.r_tail),
                    scaled_rkbi_fit=integrator.scaled_rkbi_fit(),
                    scaled_rkbi_est=np.polyval(integrator.fit_limit_params(), rfit),
                    kbi_limit=integrator.compute_kbi(),
                )

        result = PropertyResult(
            name="kbi",
            value=kbis,
            units="nm^3/molecule",
            metadata=kbi_metadata
        )

        self._cache[cache_key] = result
        return result.to(units)

    def electrolyte_kbi(self, units: str = "nm^3/molecule") -> PropertyResult:
        r"""
        Build the transformed KBI matrix for salts + molecules.

        Here we implement the approach to convert distinguishable ions into indistinguishable salts through applying the electroneutrality principle according to the approach by `Ploetz (2025) <https://doi.org/10.1021/acs.jpcb.4c07583>`_.

        Parameters
        ----------
        units: str, optional
            Units to compute KBI in, molar volume units.

        Returns
        -------
        PropertyResult
            KBI Matrix with shape (composition x neutral components x neutral components).

        Notes
        -----
        Electrolyte KBI applys corrections to transform KBI for ions into neutral species accounting for charge neutrality, sing mole fraction weighted combinations of cation and anion contributions.

        Salt-salt KBI, :math:`G_{II'}`, are computed from ion-ion KBIs according to:

        .. math::
            G_{II'} = x_{+}x_{+'}G_{++'} + x_{+}x_{-'}G_{+-'} + x_{+'}x_{-}G_{+'-} + x_{-}x_{-'}G_{--'}

        .. math::
            \begin{aligned}
            x_{+} &= \frac{n_{+}}{n_{+} + n_{-}} \\
            x_{-} &= \frac{n_{-}}{n_{+} + n_{-}} \\
            x_{+'} &= \frac{n_{+'}}{n_{+'} + n_{-'}} \\
            x_{-'} &= \frac{n_{-'}}{n_{+'} + n_{-'}} \\
            \end{aligned}

        Salt-molecule KBI, :math:`G_{iI}`, are computed from molecule-ion KBIs according to:

        .. math::
            G_{iI} = x_{+}G_{i+} + x_{-}G_{i-}

        where:
            - :math:`I` and :math:`I'` represent two salts.
            - :math:`x_{+/-}` represent the mole fraction of the cation/anion between two ions in a given salt.
            - :math:`i` represents a neutral molecule.
        """
        units = units or "nm^3/molecule"

        # first check if cached
        cache_key = ("electrolyte_kbi",)
        if cache_key in self._cache:
            return self._cache[cache_key].to(units)

        kbis = self.residue_kbi(units=units).value
        residues = self.systems.residue_molecules
        new_molecules = self.systems.electrolyte_molecules
        nmol_new = len(new_molecules)
        MAX_SALT = 2

        new_kbis = np.zeros((self.systems.n_sys, nmol_new, nmol_new))

        # Pre-parse species
        parsed = [self._parse_species(m) for m in new_molecules]

        for j, k in itertools.combinations_with_replacement(range(nmol_new), 2):
            sp_j = parsed[j]
            sp_k = parsed[k]

            # Case 1: both are salts
            if len(sp_j) == MAX_SALT and len(sp_k) == MAX_SALT:
                c1, a1 = sp_j
                c2, a2 = sp_k
                try:
                    c1_i = residues.index(c1)
                    a1_i = residues.index(a1)
                    c2_i = residues.index(c2)
                    a2_i = residues.index(a2)
                except ValueError as e:
                    raise ValueError(f"Salt species in new_molecules not found in original molecules: {e}") from e

                xc1, xa1 = self._ion_fraction(c1_i, a1_i)
                xc2, xa2 = self._ion_fraction(c2_i, a2_i)

                value = (
                    xc1 * xc2 * kbis[:, c1_i, c2_i] +
                    xc1 * xa2 * kbis[:, c1_i, a2_i] +
                    xa1 * xc2 * kbis[:, a1_i, c2_i] +
                    xa1 * xa2 * kbis[:, a1_i, a2_i]
                )

            # Case 2: one salt, one molecule/ion
            elif len(sp_j) == MAX_SALT or len(sp_k) == MAX_SALT:
                salt = sp_j if len(sp_j) == MAX_SALT else sp_k
                molec = sp_k[0] if salt is sp_j else sp_j[0]

                c, a = salt
                try:
                    c_i = residues.index(c)
                    a_i = residues.index(a)
                    m_i = residues.index(molec)
                except ValueError as e:
                    raise ValueError(f"Species in new_molecules not found in original molecules: {e}") from e

                xc, xa = self._ion_fraction(c_i, a_i)
                value = xc * kbis[:, m_i, c_i] + xa * kbis[:, m_i, a_i]

            # Case 3: neither is a salt -> direct lookup
            else:
                try:
                    m_j = residues.index(sp_j[0])
                    m_k = residues.index(sp_k[0])
                except ValueError as e:
                    raise ValueError(f"Species in new_molecules not found in original molecules: {e}") from e

                value = kbis[:, m_j, m_k]

            # Assign symmetric entries
            new_kbis[:, j, k] = value
            new_kbis[:, k, j] = value

        result = PropertyResult(
            name="electrolyte_kbi",
            value=new_kbis,
            units=units,
        )

        self._cache[cache_key] = result
        return result


    # --- electrolyte kbi helpers ---

    def _parse_species(self, name: str) -> tuple[str, ...]:
        """
        Parse a species name.

        - If it's a salt, it looks like 'Na.Cl' -> returns ('Na','Cl')
        - If it's a molecule/ion, returns ('Na',)
        """
        parts = name.split('.')
        MAX_SALT = 2
        if len(parts) > MAX_SALT:
            raise ValueError(f"Invalid species name '{name}'. Expected 'Cation.Anion' or single molecule.")
        return tuple(parts)


    def _ion_fraction(self, c_i, a_i):
        """Compute xc, xa for a salt c.a given ion counts N (nsys x nmolecules)."""
        N = self.systems.residue_counts
        Nc = N[:, c_i]
        Na = N[:, a_i]
        denom = Nc + Na

        # Avoid division by zero: if salt absent, xc=xa=0
        denom_safe = np.where(denom == 0, 1.0, denom)
        xc = Nc / denom_safe
        xa = Na / denom_safe

        # Where denom was zero, force xc=xa=0 explicitly
        mask_zero = (denom == 0)
        xc[mask_zero] = 0.0
        xa[mask_zero] = 0.0

        return xc, xa

    def kbi_plotter(self, molecule_map: dict[str, str] | None = None) -> KBIAnalysisPlotter:
        """
        Create a KBIAnalysisPlotter for visualizing RDF integration and KBI convergence.

        Parameters
        ----------
        molecule_map: dict[str, str], optional
            dictionary mapping molecule names to desired molecule labels in figures.

        Returns
        -------
        KBIAnalysisPlotter
            Plotter instance for inspecting KBI process.
        """
        return KBIAnalysisPlotter(kbi=self.kbi(), molecule_map=molecule_map)
