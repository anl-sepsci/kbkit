"""Calculator for Kirkwood-Buff Integrals (KBIs)."""

import numpy as np
from numpy.typing import NDArray

from kbkit.analysis.integrator import KBIntegrator
from kbkit.schema.kbi_metadata import KBIMetadata
from kbkit.systems.state import SystemState
from kbkit.utils.file_resolver import FileResolver


class KBICalculator:
    """
    Computes Kirkwood-Buff integrals for molecular systems using RDF data.

    Interfaces with RDFParser and KBIntegrator to extract pairwise KBIs,
    populate metadata, and apply corrections for electrolyte systems.

    Parameters
    ----------
    state : SystemState
        SystemState object providing molecule indexing, salt pairs, and composition.
    use_fixed_r : bool, optional
        If True, uses a fixed cutoff radius for KBI calculations (default: False).
    ignore_convergence_errors : bool, optional
        If True, ingnores convergence errors and forces KBI calculations to skip entire systems with non-converged RDFs (default: False).
    rdf_convergence_threshold: float, optional
        Value of the slope of RDF tail for the RDF to be considered as converged (default: 0.005).
    correct_rdf_convergence: bool, optional
        Whether to correct RDF for excess/depletion, i.e., Ganguly correction (default: True).
    apply_damping: bool, optional
        Whether to apply damping function to correlation function, i.e., Kruger correction (default: True).
    extrapolate_thermodynamic_limit: bool, optional
        Whether to extrapolate KBI value to the thermodynamic limit (default: True).

    Attributes
    ----------
    kbi_metadata : dict[str, list[KBIMetadata]]
        Dictionary mapping system names to lists of KBI metadata objects.
    """

    def __init__(
        self,
        state: SystemState,
        use_fixed_r: bool = False,
        ignore_convergence_errors: bool = False,
        rdf_convergence_threshold: float = 0.005,
        correct_rdf_convergence: bool = True,
        apply_damping: bool = True,
        extrapolate_thermodynamic_limit: bool = True
    ) -> None:
        self.state = state
        self.use_fixed_r = use_fixed_r
        self.ignore_convergence_errors = ignore_convergence_errors
        self.rdf_convergence_threshold = rdf_convergence_threshold
        self.correct_rdf_convergence = correct_rdf_convergence
        self.apply_damping = apply_damping
        self.extrapolate_thermodynamic_limit = extrapolate_thermodynamic_limit
        self.kbi_metadata: dict[str, list[KBIMetadata]] = {}

    def run(self, apply_electrolyte_correction: bool = True) -> NDArray[np.float64]:
        """
        Runs the full KBI computation workflow.

        This is the main entry point for the calculator. It orchestrates the
        process of computing the raw KBI matrix and, if specified, applies
        the electrolyte correction.

        Parameters
        ----------
        apply_electrolyte_correction: bool, optional
            If True (default), applies corrections
            for salt-salt and salt-other interactions. If False, returns the
            raw, uncorrected KBI matrix.

        Returns
        -------
        np.ndarray
            A 3D numpy array representing the final KBI matrix.
        """
        # calculate kbis for each unique molecule in topology file
        kbis = self.compute_raw_kbi_matrix()
        # check if any electrolytes are present
        electrolyte_chk = True if any("." in x for x in self.state.unique_molecules) else False
        # correct for electrolytes if present
        if apply_electrolyte_correction and electrolyte_chk:
            return self.compute_electrolyte_corrected_kbi_matrix(kbis)
        # return raw kbi matrix if electrolytes are detected but user doesn't want corrected kbi matrix
        else:
            return kbis

    def compute_raw_kbi_matrix(self) -> NDArray[np.float64]:
        r"""
        Compute the raw KBI matrix for all systems.

        Each KBI value :math:`G_{ij}` is computed by integrating the RDF
        between molecule types :math:`i, j`:

        .. math::
            G_{ij} = 4\pi \int_0^\infty (g_{ij}(r) - 1) r^2 \, dr

        Returns
        -------
        np.ndarray
            A 3D matrix of KBIs with shape ``(n_sys, n_mols, n_mols)``, where:
            ``n_sys`` is the number of systems and``n_mols`` is the number of unique molecules.

        Notes
        -----
        * If an RDF directory is missing, the corresponding system's values remain NaN, if ignore_convergence_errors is True.
        * Populates `kbi_metadata` with integration results for each RDF file.

        See Also
        --------
        :class:`KBIntegrator` : For derivation and detailed formulas for RDF integration and finite-size corrections.
        """
        kbis = np.full(
            (self.state.n_sys, len(self.state.top_molecules), len(self.state.top_molecules)), fill_value=np.nan
        )

        # iterate through all systems
        for s, meta in enumerate(self.state.config.registry):
            # if rdf dir not in system, skip
            if not meta.has_rdf():
                continue

            # get all rdf files present
            file_res = FileResolver(filepath=meta.rdf_path)
            rdf_files = file_res.get_all(role="rdf")

            # read all rdf_files
            for filepath in rdf_files:
                # integrate rdf --> kbi calc
                integrator = KBIntegrator(
                    rdf_file=filepath,
                    system_properties=meta.props,
                    use_fixed_rmin=self.use_fixed_r,
                    convergence_threshold=self.rdf_convergence_threshold,
                    correct_rdf_convergence=self.correct_rdf_convergence,
                    apply_damping=self.apply_damping,
                    extrapolate_thermodynamic_limit=self.extrapolate_thermodynamic_limit
                )

                # get molecules present in rdf
                mol_i, mol_j = integrator.rdf_molecules

                # get molecule indices
                i = self.state._get_mol_idx(mol_i, self.state.top_molecules)
                j = self.state._get_mol_idx(mol_j, self.state.top_molecules)

                # if convergence is met, store kbi value
                if integrator.rdf.is_converged:
                    kbis[s, i, j] = integrator.compute_kbi(mol_j=mol_j)
                    kbis[s, j, i] = integrator.compute_kbi(mol_j=mol_i)
                # override convergence check to skip system if not converged
                else:  # for not converged rdf
                    msg = f"RDF for system '{meta.name}' and pair {integrator.rdf_molecules} did not converge."
                    if self.ignore_convergence_errors:
                        print(f"WARNING: {msg} Skipping this system.")
                        continue
                    else:
                        raise RuntimeError(msg)

                # add values to metadata
                self._populate_kbi_metadata(system=meta.name, integrator=integrator)

        return kbis

    def _populate_kbi_metadata(self, system: str, integrator: KBIntegrator) -> None:
        r"""
        Populate KBI metadata dictionary with integration results for a given RDF file.

        Stores both raw and corrected KBI values, including:

        * :math:`r` — radial distances
        * :math:`g(r)` — RDF values
        * :math:`G(r)` — cumulative KBI curve
        * :math:`\lambda(r)` — finite-size correction factor
        * :math:`\lambda(r) \cdot G(r)` — corrected KBI curve
        * :math:`G_{\infty}` — extrapolated KBI at infinite dilution

        Parameters
        ----------
        system : str
            Name of the system being processed.
        integrator : KBIntegrator
            Integrator object containing RDF and KBI data.
        """
        self.kbi_metadata.setdefault(system, []).append(
            KBIMetadata(
                mols=tuple(integrator.rdf_molecules),
                r=integrator.rdf.r,
                g=integrator.rdf.g,
                rkbi=(integrator.rkbi()),
                scaled_rkbi=(integrator.scaled_rkbi()),
                r_fit=(rfit := integrator.rdf.r_fit),
                scaled_rkbi_fit=integrator.scaled_rkbi_fit(),
                scaled_rkbi_est=np.polyval(integrator.fit_limit_params(), rfit),
                kbi_limit=integrator.compute_kbi(),
            )
        )

    def get_metadata(self, system: str, mol_pair: tuple[str, str]) -> KBIMetadata | None:
        """
        Retrieve metadata for a specific system and molecular pair.

        Parameters
        ----------
        system: str
            System name.
        mol_pair: tuple[str, str]
            Molecule pair.

        Returns
        -------
        KBIMetadata or None
            Metadata object if found.
        """
        for meta in self.kbi_metadata.get(system, []):
            if set(meta.mols) == set(mol_pair):
                return meta
        return None

    def compute_electrolyte_corrected_kbi_matrix(self, kbi_matrix) -> NDArray[np.float64]:
        r"""
        Apply electrolyte correction to the input KBI matrix.

        This method modifies the KBI matrix to account for salt-salt and salt-other interactions
        using mole fraction-weighted combinations of cation and anion contributions.

        Returns
        -------
        np.ndarray
            Corrected KBI matrix with additional rows/columns for salt interactions.

        Notes
        -----
        Salt-salt interactions :math:`G_{ss}` are computed as:

        .. math::
            G_{ss} = x_c^2 G_{cc} + x_a^2 G_{aa} + x_c x_a (G_{ca} + G_{ac})

        Salt-other interactions :math:`G_{si}` are computed as:

        .. math::
            G_{si} = x_c G_{ic} + x_a G_{ia}

        where:
            * :math:`x_c = \frac{N_c}{N_c + N_a}` is the mole fraction of the cation
            * :math:`x_a = \frac{N_a}{N_c + N_a}` is the mole fraction of the anion
            * :math:`G_{ij}` are the raw KBIs between molecule types :math:`i` and :math:`j`
        """
        # This method first computes the raw matrix then corrects it
        salt_pairs = self.state.salt_pairs
        top_molecules = self.state.top_molecules
        unique_molecules = self.state.unique_molecules
        nosalt_molecules = self.state._nosalt_molecules
        molecule_counts = self.state.molecule_counts

        # if no salt pairs detected return original matrix
        if len(salt_pairs) == 0:
            return kbi_matrix

        # create new kbi-matrix
        adj = len(salt_pairs) - len(top_molecules)
        kbi_el = np.full((self.state.n_sys, self.state.n_comp + adj, self.state.n_comp + adj), fill_value=np.nan)

        for cat, an in salt_pairs:
            # get index of anion and cation in topology molecules
            cat_idx = top_molecules.index(cat)
            an_idx = top_molecules.index(an)

            # mol fraction of anion/cation in anion-cation pair
            x_cat = molecule_counts[:, cat_idx] / (molecule_counts[:, cat_idx] + molecule_counts[:, an_idx])
            x_an = molecule_counts[:, an_idx] / (molecule_counts[:, cat_idx] + molecule_counts[:, an_idx])

            # for salt-salt interactions add to kbi-matrix
            salt_idx = next(
                (i for i, val in enumerate(unique_molecules) if val in {f"{cat}-{an}", f"{an}-{cat}"}),
                -1,  # default if not found
            )

            if salt_idx == -1:
                raise ValueError(f"Neither f'{cat}-{an}' nor f'{an}-{cat}' found in unique_molecules.")

            # calculate KBI for salt-salt pairs
            kbi_el[salt_idx, salt_idx] = (
                x_cat**2 * kbi_matrix[cat_idx, cat_idx]
                + x_an**2 * kbi_matrix[an_idx, an_idx]
                + x_cat * x_an * (kbi_matrix[cat_idx, an_idx] + kbi_matrix[an_idx, cat_idx])
            )

            # for salt-other interactions
            for m1, mol1 in enumerate(nosalt_molecules):
                m1j = top_molecules.index(mol1)
                for m2, mol2 in enumerate(nosalt_molecules):
                    m2j = top_molecules.index(mol2)
                    kbi_el[m1, m2] = kbi_matrix[m1j, m2j]
                # adjusted KBI for mol-salt interactions
                kbi_el[m1, salt_idx] = x_cat * kbi_matrix[m1, cat_idx] + x_an * kbi_matrix[m1, salt_idx]
                kbi_el[salt_idx, m1] = x_cat * kbi_matrix[cat_idx, m1] + x_an * kbi_matrix[an_idx, m1]

        return kbi_el
