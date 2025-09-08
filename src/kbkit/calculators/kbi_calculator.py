"""Calculator for Kirkwood-Buff Integrals (KBIs)."""

import numpy as np
from numpy.typing import NDArray

from kbkit.analysis.kb_integrator import KBIntegrator
from kbkit.analysis.system_state import SystemState
from kbkit.parsers.rdf_file import RDFParser
from kbkit.schema.kbi_metadata import KBIMetadata


class KBICalculator:
    """
    Computes Kirkwood-Buff integrals for molecular systems using RDF data.

    Interfaces with RDFParser and KBIntegrator to extract pairwise KBIs,
    populate metadata, and apply corrections for electrolyte systems.

    Parameters
    ----------
    state : SystemState
        SystemState object providing molecule indexing, salt pairs, and composition.

    Attributes
    ----------
    kbi_metadata : dict[str, list[KBIMetadata]]
        Dictionary mapping system names to lists of KBI metadata objects.
    """

    def __init__(self, state: SystemState, use_fixed_r: bool) -> None:
        self.state = state
        self.use_fixed_r = use_fixed_r
        self.kbi_metadata: dict[str, list[KBIMetadata]] = {}

    def calculate(self, corrected: bool = True) -> NDArray[np.float64]:
        """
        Public entry point for computing KBIs.

        Parameters
        ----------
        corrected: bool
            Whether to apply electrolyte correction.

        Returns
        -------
        np.ndarray
            KBI matrix
        """
        return self.compute_electrolyte_corrected_kbi_matrix() if corrected else self.compute_raw_kbi_matrix()

    def compute_raw_kbi_matrix(self) -> NDArray[np.float64]:
        r"""
        Compute the raw KBI matrix for all systems.

        Each KBI value :math:`G_{ij}` is computed by integrating the radial distribution function (RDF)
        between molecule types :math:`i` and :math:`j`:

        .. math::
            G_{ij} = 4\pi \int_0^\infty [g_{ij}(r) - 1] r^2 \, dr

        Returns
        -------
        np.ndarray
            A 3D matrix of KBIs with shape ``(n_sys, n_mols, n_mols)``, where:
            - ``n_sys`` is the number of systems
            - ``n_mols`` is the number of unique molecules

        Notes
        -----
        - If an RDF directory is missing, the corresponding system's values remain NaN.
        - Populates `kbi_metadata` with integration results for each RDF file.

        See Also
        --------
        :class:`KBIntegrator` : Performs RDF integration and finite-size corrections.
        :class:`RDFParser` : Extracts molecule pairs from RDF filenames.
        """
        kbis = np.full(
            (self.state.n_sys, len(self.state.top_molecules), len(self.state.top_molecules)), fill_value=np.nan
        )

        # iterate through all systems
        for s, meta in enumerate(self.state.config.registry):
            # if rdf dir not in system, skip
            if not meta.has_rdf():
                continue

            # read all rdf_files
            for filepath in meta.rdf_path.iterdir():
                # if hidden file skip or not readable
                if filepath.name.startswith(".") or not filepath.is_file():
                    continue

                # extract molecule pair from filename
                rdf_mols = RDFParser.extract_mols(filepath.name, self.state.top_molecules)
                i, j = [self.state._get_mol_idx(mol, self.state.top_molecules) for mol in rdf_mols]

                # integrate rdf --> kbi calc
                integrator = KBIntegrator(
                    rdf_file=filepath, use_fixed_rmin=self.use_fixed_r, system_properties=meta.props
                )
                kbi = integrator.integrate()
                kbis[s, i, j] = kbi
                kbis[s, j, i] = kbi

                # add values to metadata
                self._populate_kbi_metadata(system=meta.name, rdf_mols=rdf_mols, integrator=integrator)

        return kbis

    def _populate_kbi_metadata(
        self, system: str, rdf_mols: tuple[str, ...] | list[str], integrator: KBIntegrator
    ) -> None:
        r"""
        Populate KBI metadata dictionary with integration results for a given RDF file.

        Stores both raw and corrected KBI values, including:

        - :math:`r` — radial distances
        - :math:`g(r)` — RDF values
        - :math:`G(r)` — cumulative KBI curve
        - :math:`\lambda(r)` — finite-size correction factor
        - :math:`\lambda(r) \cdot G(r)` — corrected KBI curve
        - :math:`G_{\infty}` — extrapolated KBI at infinite dilution

        Parameters
        ----------
        system : str
            Name of the system being processed.
        rdf_mols : tuple[str, str]
            Pair of molecules associated with the RDF file.
        integrator : KBIntegrator
            Integrator object containing RDF and KBI data.
        """
        self.kbi_metadata.setdefault(system, []).append(
            KBIMetadata(
                mols=tuple(rdf_mols),
                r=integrator.rdf.r,
                g=integrator.rdf.g,
                rkbi=(rkbi := integrator.rkbi()),
                lam=(lam := integrator.lambda_ratio()),
                lam_rkbi=rkbi * lam,
                lam_fit=(lam_fit := lam[integrator.rdf.r_mask]),
                lam_rkbi_fit=np.polyval(integrator.fit_kbi_inf(), lam_fit),
                kbi=integrator.integrate(),
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

    def compute_electrolyte_corrected_kbi_matrix(self) -> NDArray[np.float64]:
        r"""
        Apply electrolyte correction to the input KBI matrix.

        This method modifies the KBI matrix to account for salt-salt and salt-other interactions
        using mole fraction-weighted combinations of cation and anion contributions.

        Parameters
        ----------
        kbi_matrix : np.ndarray
            Raw KBI matrix with shape ``(n_sys, n_comp, n_comp)``.
        salt_pairs : list[tuple[str, str]]
            List of salt component pairs (cation, anion).
        top_molecules : list[str]
            Molecules defined in the topology.
        unique_molecules : list[str]
            Molecules including salt pairs.
        nosalt_molecules : list[str]
            Molecules excluding salt components.
        molecule_counts : np.ndarray
            Molecule counts per system.

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
            - :math:`x_c = \frac{N_c}{N_c + N_a}` is the mole fraction of the cation
            - :math:`x_a = \frac{N_a}{N_c + N_a}` is the mole fraction of the anion
            - :math:`G_{ij}` are the raw KBIs between molecule types :math:`i` and :math:`j`
        """
        # initialize the variables
        kbi_matrix = self.compute_raw_kbi_matrix()
        salt_pairs = self.state.salt_pairs
        top_molecules = self.state.top_molecules
        unique_molecules = self.state.unique_molecules
        nosalt_molecules = self.state._nosalt_molecules
        molecule_counts = self.state.molecule_counts

        # if no salt pairs detected return original matrix
        if len(salt_pairs) == 0:
            return kbi_matrix

        n_sys = kbi_matrix.shape[0]
        n_comp = len(unique_molecules)

        # create new kbi-matrix
        adj = len(salt_pairs) - len(top_molecules)
        kbi_el = np.full((n_sys, n_comp + adj, n_comp + adj), fill_value=np.nan)

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
