"""Calculator for Kirkwood-Buff Integrals."""


from numpy.typing import NDArray
import numpy as np

from kbkit.analysis.analyzer import SystemAnalyzer
from kbkit.analysis.integrator import KBIntegrator
from kbkit.parsers.rdf import RDFParser
from kbkit.schema.config import SystemConfig
from kbkit.schema.kbi_metadata import KBIMetadata


class KBICalculator:

    def __init__(self, config: SystemConfig) -> None:
        self.config = config
        self.analyzer = SystemAnalyzer(self.config)
        self.kbi_metadata: dict[str, KBIMetadata] = {}

    def compute_raw_kbi_matrix(self) -> NDArray[np.float64]:
        r"""
        Get Kirkwood-Buff integral (KBI) matrix, **G**, for all systems and all pairs of molecules.

        Returns
        -------
        np.ndarray
            A 3D matrix of Kirkwood-Buff integrals with shape ``(n_sys, n_mols, n_mols)``,
            where:

            - ``n_sys`` — number of systems
            - ``n_mols`` — number of unique molecules

        If an RDF directory is missing, the corresponding system's values remain NaN.

        See Also
        --------
        :class:`kbkit.parsers.rdf.RDFParser` : Parses RDF files.
        :class:`kbkit.analysis.integrator.KBIntegrator` : Performs the RDF integration to compute KBIs and apply finite-size corrections.
        """      
        kbis = np.full((self.n_sys, len(self.top_molecules), len(self.top_molecules)), fill_value=np.nan)

        # iterate through all systems
        for s, meta in enumerate(self.config.registry):
            # if rdf dir not in system, skip
            if not meta.has_rdf():
                continue

            # read all rdf_files
            for filepath in meta.rdf_path.iter_dir():
                rdf_mols = RDFParser.extract_mols(filepath.name, self.top_molecules)
                i, j = [self._get_mol_idx(mol, self.top_molecules) for mol in rdf_mols]

                # integrate rdf --> kbi calc
                integrator = KBIntegrator(filepath, meta.props)
                kbi = integrator.integrate()
                kbis[s, i, j] = kbi
                kbis[s, j, i] = kbi

                # add values to metadata
                self._populate_kbi_metadata(system=meta.name, rdf_mols=rdf_mols, integrator=integrator)

        return kbis
    
    def _populate_kbi_metadata(self, system: str, rdf_mols: tuple[str, str], integrator: KBIntegrator) -> None:
        """Add KBI integration results to MetaData dictionary."""
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
                kbi=integrator.integrate()
            )
        )

    def get_corrected_kbi_matrix(self) -> NDArray[np.float64]:
        """Correct KBI matrix for electrolytes."""
        return self.apply_electrolyte_correction(
            kbi_matrix=self.compute_raw_kbi_matrix(),
            salt_pairs=self.analyzer.salt_pairs,
            top_molecules=self.analyzer.top_molecules,
            unique_molecules=self.analyzer.unique_molecules,
            nosalt_molecules=self.analyzer.nosalt_molecules,
            molecule_counts=self.analyzer.molecule_counts
        )

    
    @staticmethod
    def apply_electrolyte_correction(
        self, 
        kbi_matrix: NDArray[np.float64],
        salt_pairs: list[tuple[str, str]],
        top_molecules: list[str],
        unique_molecules: list[str],
        nosalt_molecules: list[str],
        molecule_counts: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        r"""
        Apply electrolyte correction to the input KBI matrix.

        This method modifies the KBI matrix to account for salt-salt and salt-other interactions
        by adding additional rows and columns for salt pairs. It calculates the KBI for salt-salt interactions
        based on the mole fractions of the salt components and their interactions with other molecules.

        Parameters
        ----------
        kbi_matrix : np.ndarray
            A 3D matrix representing the original KBI matrix with shape ``(n_sys, n_comp, n_comp)``,
            where ``n_sys`` is the number of systems and ``n_comp`` is the number of unique components.

        Returns
        -------
        np.ndarray
            A 3D matrix representing the modified KBI matrix with additional rows and columns for salt pairs.

        Notes
        -----
        - If no salt pairs are defined, it returns the original KBI matrix.
        - The salt pairs are defined in ``KBThermo.salt_pairs``, which should be a list of tuples containing the names of the salt components.

        This method calculates the KBI matrix (**G**) for systems with salts for salt-salt interactions (:math:`G_{ss}`) and salt-other interactions (:math:`G_{si}`) as follows:

        .. math::
            G_{ss} = x_c^2 G_{cc} + x_a^2 G_{aa} + x_c x_a (G_{ca} + G_{ac})

        .. math::
            G_{si} = x_c G_{ic} + x_a G_{ia}

        .. math::
            x_c = \frac{N_c}{N_c + N_a}

        .. math::
            x_a = \frac{N_a}{N_c + N_a}

        where:
            - :math:`G_{ss}` is the KBI for salt-salt interactions.
            - :math:`G_{si}` is the KBI for salt-other interactions.
            - :math:`x_c` and :math:`x_a` are the mole fractions of the salt components.
            - :math:`N_c` and :math:`N_a` are the counts of the salt components in the system.
            - :math:`G_{cc}`, :math:`G_{aa}`, and :math:`G_{ca}` are the KBIs for the respective pairs of molecules.
        """

        kbi_matrix = self.compute_raw_kbi_matrix()

        # if no salt pairs detected return original matrix
        if len(salt_pairs) == 0:
            return kbi_matrix
        
        n_sys = kbi_matrix.shape[0]
        n_comp = len(unique_molecules)
        
        # create new kbi-matrix
        adj = len(salt_pairs) - len(top_molecules)
        kbi_el = np.full((n_sys, n_comp+adj, n_comp+adj), fill_value=np.nan)

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
                -1 # default if not found
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

    