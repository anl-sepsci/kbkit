"""Calculate thermodynamics from Kirkwood-Buff theory."""

import os
from functools import cached_property, partial
from itertools import product
from pathlib import Path
from typing import Any, Callable

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import cumulative_trapezoid

from kbkit.analysis import KBI, RDF, SystemSet


class KBThermo(SystemSet):
    """Apply Kirkwood-Buff (KB) theory to calculate thermodynamic properties from RDF.

    This class inherits system properties from :class:`SystemSet` and uses them for the calculation of
    Kirkwood-Buff integrals and other thermodynamic properties.

    Parameters
    ----------
    base_path : str, optional
        The base path where the systems are located. Defaults to the current working directory.
    pure_component_path : str, optional
        The path where pure component systems are located. Defaults to a 'pure_components' directory next to the base path.
    base_systems : list, optional
        A list of base systems to include. If not provided, it will automatically detect systems in the base path.
    pure_component_systems : list, optional
        A list of pure component systems to include. If not provided, it will automatically detect systems in the pure component path.
    rdf_dir : str, optional
        The directory name where RDF files are stored. Defaults to 'rdf_files'.
    start_time : int, optional
        The starting time for analysis, used in temperature and enthalpy calculations. Defaults to 0.
    ensemble : str, optional
        The ensemble type for the systems, e.g., 'npt', 'nvt'. Defaults to 'npt'.
    gamma_integration_type : str, optional
        The type of integration to use for gamma calculations. Defaults to 'numerical'.
    gamma_polynomial_degree : int, optional
        The degree of the polynomial for gamma integration. Defaults to 5.
    cation_list : list, optional
        A list of cation names to consider for salt pairs. Defaults to an empty list.
    anion_list : list, optional
        A list of anion names to consider for salt pairs. Defaults to an empty list.
    """

    def __init__(
        self,
        base_path: str,
        pure_component_path: str,
        base_systems: list[str],
        pure_component_systems: list[str],
        rdf_dir: str,
        start_time: int,
        ensemble: str,
        gamma_integration_type: str,
        gamma_polynomial_degree: int,
        cation_list: list[str],
        anion_list: list[str],
    ) -> None:
        # initialize SystemSet
        super().__init__(
            base_path=base_path,
            pure_component_path=pure_component_path,
            base_systems=base_systems,
            pure_component_systems=pure_component_systems,
            rdf_dir=rdf_dir,
            start_time=start_time,
            ensemble=ensemble,
            cation_list=cation_list,
            anion_list=anion_list,
        )

        self.gamma_integration_type = gamma_integration_type
        self.gamma_polynomial_degree = gamma_polynomial_degree

    @property
    def salt_pairs(self) -> list[tuple[str, str]]:
        """list: List of salt pairs as (cation, anion) tuples."""
        return self._salt_pairs

    @salt_pairs.setter
    def salt_pairs(self, pairs: list[tuple[str, str]]) -> None:
        # validates the salt_pairs list
        if not isinstance(pairs, list):
            raise TypeError(f"Expected a list of salt pairs, got {type(pairs).__name__}: {pairs}")
        PAIR = 2
        if not all(isinstance(pair, tuple) and len(pair) == PAIR for pair in pairs):
            raise ValueError("Each salt pair must be a tuple of two elements (cation, anion).")
        # checks molecules in pairs are in top_molecules
        for pair in pairs:
            if not all(mol in self.top_molecules for mol in pair):
                raise ValueError(
                    f"Salt pair {pair} contains molecules not present in top_molecules: {self.top_molecules}"
                )
        self._salt_pairs = pairs

    @cached_property
    def nosalt_molecules(self) -> list[str]:
        """list: Molecules not part of any salt pair."""
        # filter out molecules that are part of salt pairs
        return [mol for mol in self.top_molecules if mol not in [x for pair in self.salt_pairs for x in pair]]

    @cached_property
    def salt_molecules(self) -> list[str]:
        """list: Unique molecules in salt-pairs."""
        return ["-".join(pair) for pair in self.salt_pairs]

    @cached_property
    def unique_molecules(self) -> list[str]:
        """list: Molecules after combining salt pairs as single entries."""
        return self.nosalt_molecules + self.salt_molecules

    @property
    def n_comp(self) -> int:
        """int: Number of unique components (molecules) in the system set."""
        return len(self.unique_molecules)

    def _top_mol_idx(self, mol: str) -> int:
        """
        Index of molecule (`mol`) in `top_molecules` list.

        Parameters
        ----------
        mol: str
            Molecule name to get index of

        Returns
        -------
        int
            Index of mol topology molecules list
        """
        if mol not in self.top_molecules:
            raise ValueError(f"Molecule {mol} not in topology molecules. Topology molecules: {self.top_molecules}")
        return list(self.top_molecules).index(mol)

    def _mol_idx(self, mol: str) -> int:
        """
        Index of molecule (`mol`) from in `unique_molecules` list.

        Parameters
        ----------
        mol: str
            Molecule name to get index of

        Returns
        -------
        int
            Index of mol in unique molecules list
        """
        if mol not in self.unique_molecules:
            raise ValueError(f"Molecule {mol} not in topology molecules. Unique molecules: {self.unique_molecules}")
        return list(self.unique_molecules).index(mol)

    def calculate_kbis(self) -> NDArray[np.float64]:
        r"""
        Get Kirkwood-Buff integral (KBI) matrix, **G**, for all systems and all pairs of molecules.

        Returns
        -------
        np.ndarray
            A 3D matrix of Kirkwood-Buff integrals with shape ``(n_sys, n_mols, n_mols)``,
            where:

            - ``n_sys`` — number of systems
            - ``n_mols`` — number of unique molecules

        Notes
        -----
        For each system, element :math:`G_{ij}` of matrix **G**, is the KBI for a pair of molecules :math:`i, j`
        and computed as:

        .. math::

            G_{ij} = 4 \\pi \\, \\int_0^{\\infty} \\, (g_{ij}(r) - 1) \\, r^2 \\, dr

        where, :math:`g_{ij}(r)` is the RDF for the pair.

        The algorithm:
            1. Iterates through each system.
            2. Checks if the RDF directory exists; skips systems without RDF data.
            3. Reads RDF files for each molecular pair.
            4. Integrates RDF data to compute :math:`G_{ij}`.
            5. Stores results in a symmetric KBI matrix for the system.

        If an RDF directory is missing, the corresponding system's values remain NaN.

        See Also
        --------
        :class:`kbkit.kb.rdf.RDF` : Parses RDF files.
        :class:`kbkit.kb.kbi.KBI` : Performs the RDF integration to compute KBIs and apply finite-size corrections.
        """
        if "_kbis" not in self.__dict__:
            self._kbis = np.full((self.n_sys, len(self.top_molecules), len(self.top_molecules)), fill_value=np.nan)

            # iterate through all systems
            for s, sys in enumerate(self.systems):
                # if rdf dir not in system, skip
                rdf_full_path = os.path.join(self.base_path, sys, self.rdf_dir)
                if not Path(rdf_full_path).exists():
                    continue

                # read all rdf_files
                for rdf_file in os.listdir(rdf_full_path):
                    rdf_file_path = os.path.join(rdf_full_path, rdf_file)
                    rdf_mols = RDF.extract_mols(rdf_file_path, self.top_molecules)
                    i, j = [self._top_mol_idx(mol) for mol in rdf_mols]

                    # integrate rdf --> kbi calc
                    integrator = KBI(rdf_file_path)
                    kbi = integrator.integrate()
                    self._update_kbi_dict(system=sys, rdf_mols=rdf_mols, integrator=integrator)

                    # add to matrix
                    self._kbis[s, i, j] = kbi
                    self._kbis[s, j, i] = kbi

        return self._kbis

    def kbi_dict(self) -> dict[str, dict[str, Any]]:
        r"""
        Get a dictionary of KBI and RDF properties for each system and molecular pair.

        Returns
        -------
        dict[str, dict[str, float or numpy.ndarray]]
            A nested dictionary mapping systems and molecule pairs to RDF and KBI properties.
            Outer keys are systems, inner keys are molecule pairs, and values are either scalars(:class:`float`) or arrays (:class:`np.ndarray`).

        Notes
        -----
        The inner keys are defined as follows:
            - '`r`': Radial distance array from RDF.
            - '`g`': RDF values for the pair.
            - '`rkbi`': KBI value for the pair.
            - '`lambda`': Lambda ratio used in KBI calculation.
            - '`lambda_kbi`': KBI value adjusted by lambda ratio.
            - '`lambda_fit`': Lambda ratio for the fitted RDF.
            - '`lambda_kbi_fit`': KBI value adjusted by fitted lambda ratio.
            - '`kbi_inf`': Infinite dilution KBI value.
        """
        # returns dictionary of kbi / rdf properties by system and pair molecular interaction
        if not hasattr(self, "_kbi_dict"):
            self.kbi_mat()
        return self._kbi_dict

    def _update_kbi_dict(self, system: str, rdf_mols: list[str], integrator: KBI) -> None:
        # add kbi/rdf properties to dictionary; sorted by system / rdf
        if not hasattr(self, "_kbi_dict"):
            self._kbi_dict: dict[str, dict[str, Any]] = {}

        self._kbi_dict.setdefault(system, {}).update(
            {
                "-".join(rdf_mols): {
                    "r": integrator.rdf.r,
                    "g": integrator.rdf.g,
                    "rkbi": (rkbi := integrator.rkbi()),
                    "lambda": (lam := integrator.lambda_ratio()),
                    "lambda_kbi": lam * rkbi,
                    "lambda_fit": lam[integrator.rdf.r_mask],
                    "lambda_kbi_fit": np.polyval(integrator.fit_kbi_inf(), lam[integrator.rdf.r_mask]),
                    "kbi_inf": integrator.integrate(),
                }
            }
        )

    def electrolyte_kbi_correction(self, kbi_matrix: np.ndarray) -> NDArray[np.float64]:
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
        # if no salt pairs detected return original matrix
        if len(self.salt_pairs) == 0:
            return kbi_matrix

        # create new kbi-matrix
        adj = len(self.salt_pairs) - len(self.top_molecules)
        kbi_el = np.full((self.n_sys, self.n_comp + adj, self.n_comp + adj), fill_value=np.nan)

        for c, a in self.salt_pairs:
            # get index of anion and cation in topology molecules
            cj = self._top_mol_idx(c)
            aj = self._top_mol_idx(a)

            # mol fraction of anion/cation in anion-cation pair
            xc = self.molecule_counts[:, cj] / (self.molecule_counts[:, cj] + self.molecule_counts[:, aj])
            xa = self.molecule_counts[:, aj] / (self.molecule_counts[:, cj] + self.molecule_counts[:, aj])

            # for salt-salt interactions add to kbi-matrix
            sj = next(
                (i for i, val in enumerate(self.unique_molecules) if val in {f"{c}-{a}", f"{a}-{c}"}),
                -1,  # default if not found
            )
            if sj == -1:
                raise ValueError(f"Neither f'{c}-{a}' nor f'{a}-{c}' found in unique_molecules.")

            # calculate electrolyte KBI for salt-salt pairs
            kbi_el[sj, sj] = (
                xc**2 * kbi_matrix[cj, cj]
                + xa**2 * kbi_matrix[aj, aj]
                + xc * xa * (kbi_matrix[cj, aj] + kbi_matrix[aj, cj])
            )

            # for salt other interactions
            for m1, mol1 in enumerate(self.nosalt_molecules):
                m1j = self.top_molecules.index(mol1)
                for m2, mol2 in enumerate(self.nosalt_molecules):
                    m2j = self.top_molecules.index(mol2)
                    kbi_el[m1, m2] = kbi_matrix[m1j, m2j]
                # adjusted KBI for mol-salt interactions
                kbi_el[m1, sj] = xc * kbi_matrix[m1, cj] + xa * kbi_matrix[m1, aj]
                kbi_el[sj, m1] = xc * kbi_matrix[cj, m1] + xa * kbi_matrix[aj, m1]

        return kbi_el

    def kbi_mat(self) -> NDArray[np.float64]:
        """
        Get the KBI matrix (**G**) with electrolyte corrections applied if salt pairs are defined.

        Returns
        -------
        np.ndarray
            A 3D matrix representing the KBI matrix with shape ``(n_sys, n_comp, n_comp)``,
            where ``n_sys`` is the number of systems and ``n_comp`` is the number of components,
            including any additional salt pairs if defined.
        """
        if "_kbi_mat" not in self.__dict__:
            kbi_matrix = self.calculate_kbis()
            self._kbi_mat = self.electrolyte_kbi_correction(kbi_matrix=kbi_matrix.copy())
        return self._kbi_mat

    def kd(self) -> NDArray[np.float64]:
        """
        Get the Kronecker delta between pairs of unique molecules.

        Returns
        -------
        np.ndarray
            A 2D array representing the Kronecker deltas with shape ``(n_comp, n_comp)``,
            where ``n_comp`` is the number of unique components.
        """
        return np.eye(self.n_comp)

    def b_mat(self) -> NDArray[np.float64]:
        r"""
        Construct a symmetric matrix **B** for each system based on the number densities and KBIs.

        Returns
        -------
        np.ndarray
            A 3D matrix with shape ``(n_sys, n_comp, n_comp)``,
            where ``n_sys`` is the number of systems and ``n_comp`` is the number of unique components.

        Notes
        -----
        Elements of **B** are calculated for molecules :math:`i,j`, using the formula:

        .. math::
            B_{ij} = \rho_{ij} G_{ij} + \rho_i \delta_{i,j}

        where:
            - :math:`\rho_{ij}` is the pairwise number density of molecules in each system.
            - :math:`G_{ij}` is the KBI for the pair of molecules.
            - :math:`\rho_i` is the number density of molecule :math:`i`.
            - :math:`\delta_{i,j}` is the Kronecker delta for molecules :math:`i,j`.
        """
        if "_b_mat" not in self.__dict__:
            self._b_mat = (
                self.rho_ij(units="molecule/nm^3") * self.kbi_mat()
                + self.rho(units="molecule/nm^3")[:, :, np.newaxis] * self.kd()[np.newaxis, :, :]
            )
        return self._b_mat

    @property
    def _b_inv(self) -> NDArray[np.float64]:
        """np.ndarray: Inverse of the B matrix."""
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.asarray(np.linalg.inv(self.b_mat()))

    @property
    def _b_det(self) -> NDArray[np.float64]:
        """np.ndarray: Determinant of the B matrix."""
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.asarray(np.linalg.det(self.b_mat()))

    def b_cofactors(self) -> NDArray[np.float64]:
        r"""
        Get the cofactors of **B** for each system.

        Returns
        -------
        np.ndarray
            A 3D matrix representing the cofactors of **B** with shape ``(n_sys, n_comp, n_comp)``,

        Notes
        -----
        The cofactors of **B**, :math:`Cof(\mathbf{B})`, are calculated as:

        .. math::
            Cof(\mathbf{B}) = |\mathbf{B}| \cdot \mathbf{B}^{-1}

        where:
            - :math:`|\mathbf{B}|` is the determinant of **B**
            - :math:`\mathbf{B}^{-1}` is the inverse of **B**
        """
        if "_b_cofactors" not in self.__dict__:
            self._b_cofactors = self._b_det[:, np.newaxis, np.newaxis] * self._b_inv
        return self._b_cofactors

    def a_mat(self) -> NDArray[np.float64]:
        r"""
        Construct a symmetric matrix **A** for each system from compositions and **G**.

        Returns
        -------
        np.ndarray
            A 3D matrix with shape ``(n_sys, n_comp, n_comp)``,
            where ``n_sys`` is the number of systems and ``n_comp`` is the number of unique components.

        Notes
        -----
        Elements of **A** are calculated for molecules :math:`i,j`, using the formula:

        .. math::
            A_{ij} = \rho x_i x_j G_{ij} + x_i \delta_{i,j}

        where:
            - :math:`\rho` is the average mixture density.
            - :math:`G_{ij}` is the KBI for the pair of molecules.
            - :math:`x_i` is the mol fraction of molecule :math:`i`.
            - :math:`\delta_{i,j}` is the Kronecker delta for molecules :math:`i,j`.
        """
        if "_a_mat" not in self.__dict__:
            self._a_mat = (1 / self.v_bar(units="nm^3/molecule"))[:, np.newaxis, np.newaxis] * self.mol_fr[
                :, :, np.newaxis
            ] * self.mol_fr[:, np.newaxis, :] * self.kbi_mat() + self.mol_fr[:, :, np.newaxis] * self.kd()[
                np.newaxis, :, :
            ]
        return self._a_mat

    def isothermal_compressability(self, units: str = "kJ/mol") -> NDArray[np.float64]:
        r"""
        Calculate the isothermal compressability, :math:`\kappa`, for each system.

        Parameters
        ----------
        units : str
            Units of energy to report values in. Default is 'kJ/mol'.

        Returns
        -------
        np.ndarray
            Isothermal compressability values for each system, with shape ``(n_sys)``

        Notes
        -----
        Isothermal compressability (:math:`\kappa`) is calculated by:

        .. math::
            \kappa RT = \sum_{j=1}^n V_j A_{ij}^{-1}

        where:
            - :math:`V_j` is the molar volume of molecule :math:`j`
            - :math:`A_{ij}^{-1}` is the inverse of **A** for molecules :math:`i,j`

        """
        R = self.ureg("R").to(units + "/K").magnitude  # gas constant
        kT = (1 / (R * self.temperature())) * (self.molar_volume()[np.newaxis, :] / self.a_mat()[:, 0, :]).sum(
            axis=1
        )  # isothermal compressability
        kT_converted = (
            self.Q_(kT, units=f"{units.split('/')[1]}/{units.split('/')[0]} * nm^3/molecule").to("1/kPa").magnitude
        )
        return np.asarray(kT_converted)

    def dmu_dn(self, units: str = "kJ/mol") -> NDArray[np.float64]:
        r"""
        Compute the derivative of the chemical potential of molecule :math:`i` with respect to the number of molecules of molecule :math:`j`.

        Parameters
        ----------
        units : str
            Units of energy to report values in. Default is 'kJ/mol'.

        Returns
        -------
        np.ndarray
            A 3D matrix of shape ``(n_sys, n_comp, n_comp)``

        Notes
        -----
        Derivative of chemical potential with respect to molecule number (:math:`\frac{\partial \mu_i}{\partial n_j}`) is calculated as follows:

        .. math::
           \frac{\partial \mu_i}{\partial n_j} = \frac{k_bT}{\left<V\right> |\mathbf{B}|}\left(\frac{\sum_{a=1}^n\sum_{b=1}^n \rho_a\rho_b\left|B^{ij}B^{ab}-B^{ai}B^{bj}\right|}{\sum_{a=1}^n\sum_{b=1}^n \rho_a\rho_b B^{ab}}\right)

        where:
            - :math:`\mu_i` is the chemical potential of molecule :math:`i`
            - :math:`n_j` is the molecule number of molecule :math:`j`
            - :math:`k_b` is the Boltmann constant
            - :math:`\left<V\right>` is the ensemble average box volume
            - :math:`B^{ij}` is the element of :math:`Cof(\mathbf{B})` (the cofactors of **B**) for molecules :math:`i,j`

        """
        # get cofactors x number density
        cofactors_rho = self.b_cofactors() * self.rho_ij(units="molecule/nm^3")

        # get denominator of matrix calculation
        b_lower = cofactors_rho.sum(axis=tuple(range(1, cofactors_rho.ndim)))  # sum over dimensions 1:end

        # get numerator of matrix calculation
        B_prod = np.empty((self.n_sys, self.n_comp, self.n_comp, self.n_comp, self.n_comp))
        for a, b, i, j in product(range(self.n_comp), repeat=4):
            B_prod[:, a, b, i, j] = self.rho_ij(units="molecule/nm^3")[:, i, j] * (
                self.b_cofactors()[:, a, b] * self.b_cofactors()[:, i, j]
                - self.b_cofactors()[:, i, a] * self.b_cofactors()[:, j, b]
            )
        b_upper = B_prod.sum(axis=tuple(range(3, B_prod.ndim)))

        # get chemical potential with respect to mol number in target units
        b_frac = b_upper / b_lower[:, np.newaxis, np.newaxis]
        dmu_dn_mat = (
            self.ureg("R").to(units + "/K").magnitude
            * self.temperature()[:, np.newaxis, np.newaxis]
            * b_frac
            / (self.volume() * self._b_det)[:, np.newaxis, np.newaxis]
        )
        return np.asarray(dmu_dn_mat)

    def _matrix_setup(self, matrix: np.ndarray) -> NDArray[np.float64]:
        """Set up matrices for multicomponent analysis."""
        n = self.n_comp - 1
        mat_ij = matrix[:, :n, :n]
        mat_in = matrix[:, :n, n][:, :, np.newaxis]
        mat_jn = matrix[:, n, :n][:, np.newaxis, :]
        mat_nn = matrix[:, n, n][:, np.newaxis, np.newaxis]
        return np.asarray(mat_ij - mat_in - mat_jn + mat_nn)

    def hessian(self, units: str = "kJ/mol") -> NDArray[np.float64]:
        r"""
        Hessian of Gibbs mixing free energy for molecules :math:`i,j`.

        Parameters
        ----------
        units: str
            Units of energy to report values in. Default is 'kJ/mol'.

        Returns
        -------
        np.ndarray
            A 3D matrix of shape ``(n_sys, n_comp-1, n_comp-1)``

        Notes
        -----
        Hessian matrix, **H**, with elements for molecules :math:`i,j` is calculated as follows:

        .. math::
            H_{ij} = M_{ij} - M_{in} - M_{jn} + M_{nn}

        .. math::
            M_{ij} = \frac{RT \Delta_{ij}^{-1}}{\rho x_i x_j}

        .. math::
            \Delta_{ij} = \frac{\delta_{ij}}{\rho x_i} + \frac{1}{\rho x_n} + G_{ij} - G_{in} - G_{jn} + G_{nn}

        where:
            - **H** is the Hessian matrix
            - **G** is the KBI matrix
            - :math:`x_i` is mol fraction of molecule :math:`i`
            - :math:`\rho` is the density of each system

        """
        G = self.kbi_mat()  # Cache this to avoid repeated calls

        # difference between ij interactions with each other and last component
        delta_G = self._matrix_setup(G)

        with np.errstate(divide="ignore", invalid="ignore"):
            # get Delta matrix for Hessian calc
            Delta_ij = (
                self.kd()[np.newaxis, :] * self.v_bar()[:, np.newaxis, np.newaxis] / self.mol_fr[:, np.newaxis]
                + (self.v_bar() / (self.mol_fr[:, self.n_comp - 1]))[:, np.newaxis, np.newaxis]
                + delta_G
            )
            Delta_ij_inv = np.linalg.inv(Delta_ij)
            R = self.ureg("R").to(units + "/K").magnitude  # gas constant

            # get M matrix for hessian calculation
            M_ij = (
                Delta_ij_inv
                * R
                * self.temperature()[:, np.newaxis, np.newaxis]
                * self.v_bar()[:, np.newaxis, np.newaxis]
                / (self.mol_fr[:, :, np.newaxis] * self.mol_fr[:, np.newaxis, :])
            )

        return self._matrix_setup(M_ij)

    def det_hessian(self, units: str = "kJ/mol") -> NDArray[np.float64]:
        r"""
        Compute the determinant, :math:`|\mathbf{H}|`, of the Hessian matrix.

        Parameters
        ----------
        units: str
            Units of energy to report values in. Default is 'kJ/mol'.

        Returns
        -------
        np.ndarray
            A 1D array of shape ``(n_sys)``
        """
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.asarray(np.linalg.det(self.hessian(units)))

    def s0_mat(self) -> NDArray[np.float64]:
        r"""
        Structure factor as q :math:`\rightarrow` 0 for composition-composition fluctuations.

        Parameters
        ----------
        energy_units: str
            Units of energy to report values in. Default is 'kJ/mol'.
        vol_units: str
            Units of volume for scattering intensity calculations. Default is nm^3/molecule.

        Returns
        -------
        np.ndarray
            A 3D matrix of shape ``(n_sys, n_comp-1, n_comp-1)``

        Notes
        -----
        The structure factor, :math:`S_{ij}(0)`, is calculated as follows:

        .. math::
            S_{ij}(0)  = RT H_{ij}^{-1}

        where:
            - :math:`H_{ij}` is the Hessian of molecules :math:`i,j`
        """
        units = "kJ/mol"  # the units don't matter here because the energy units will cancel out
        R = self.ureg("R").to(units + "/K").magnitude  # gas constant
        S0 = R * self.temperature()[:, np.newaxis, np.newaxis] / self.hessian(units)
        return np.asarray(S0)

    def drho_elec_dx(self, units: str = "cm^3/molecule") -> NDArray[np.float64]:
        r"""
        Electron density contrast for a mixture.

        Parameters
        ----------
        units: str
            Units of energy to report values in. Default is 'kJ/mol'.

        Returns
        -------
        np.ndarray
            A 2D array with shape ``(n_comp-1, n_comp-1)``

        Notes
        -----
        The electron density contrast, :math:`\frac{\partial \rho^e}{\partial x_i}`, is calculated according to:

        .. math::
            \frac{\partial \rho^e}{\partial x_i} = \rho \left( Z_i - Z_n \right) - \overline{Z} \rho \left( \frac{V_i - V_n}{\overline{V}} \right)

        where:
            - :math:`Z_i` is the number of electrons in molecule :math:`i`
            - :math:`V_i` is the molar volume of molecule :math:`i`
            - :math:`\overline{V}` is the molar volume of each system
        """
        # calculate electron density contrast
        return (1 / self.v_bar(units))[:, np.newaxis] * (
            self.delta_n_elec()[np.newaxis, :]
            - self.n_elec_bar()[:, np.newaxis] * self.delta_v(units)[np.newaxis, :] / self.v_bar(units)[:, np.newaxis]
        )

    def i0(self, units: str = "1/cm") -> NDArray[np.float64]:
        r"""
        Small angle x-ray scattering (SAXS) intensity as q :math:`\rightarrow` 0.

        Parameters
        ----------
        units: str
            Units of inverse length to report values in. Default is '1/cm'.

        Returns
        -------
        np.ndarray
            A 1D array with shape ``(n_sys)``

        Notes
        -----
        SAXS intensity, :math:`I_0`, is calculated via:

        .. math::
            I_0 = \frac{r_e^2}{\rho} \sum_{i=1}^{n-1} \sum_{j=1}^{n-1} \left(\frac{\partial \rho^e}{\partial x_i}\right) \left(\frac{\partial \rho^e}{\partial x_j}\right) S_{ij}(0)

        where:
            - :math:`r_e` is the electron radius
            - :math:`\rho` is density of system
            - :math:`\frac{\partial \rho^e}{\partial x_i}` is electron density contrast for molecule :math:`i`
            - :math:`S_{ij}(0)` is structure factor for molecules :math:`i,j`

        See Also
        --------
        :meth:`s0_mat`: Structure factor calculation
        :meth:`drho_elec_dx`: Electron density constrast calculation
        """
        # get the electron radius in desired units
        re_units = units.split("/")[1] if "/" in units else "cm"
        re = self.Q_(2.81794092e-13, units="cm").to(re_units).magnitude  # electron radius
        vol_units = f"{units.split('/')[1]}^3/molecule"
        # calculate squared of electron density constrast combinations
        drho_dx2 = (
            self.drho_elec_dx(units=vol_units)[:, :, np.newaxis] * self.drho_elec_dx(units=vol_units)[:, np.newaxis, :]
        )
        # calculate saxs intensity
        i0_mat = re**2 * self.v_bar(vol_units)[:, np.newaxis, np.newaxis] * drho_dx2 * self.s0_mat()
        i0_1d = np.nansum(i0_mat, axis=tuple(range(1, i0_mat.ndim)))  # sum of 1:last_dim
        return np.asarray(i0_1d)

    def dmu_dxs(self, units: str = "kJ/mol") -> NDArray[np.float64]:
        r"""
        Compute the derivative of the chemical potential of molecule :math:`i` with respect to mol fraction of molecule :math:`j`.

        Parameters
        ----------
        units : str
            Units of energy to report values in. Default is 'kJ/mol'.

        Returns
        -------
        np.ndarray
            A 3D matrix of shape ``(n_sys, n_comp, n_comp)``

        Notes
        -----
        Derivative of chemical potential with respect to mol fraction (:math:`\frac{\partial \mu_i}{\partial x_j}`) is calculated as follows:

        .. math::
           \frac{\partial \mu_i}{\partial x_j} = n_T \left( \frac{\partial \mu_i}{\partial n_j} - \frac{\partial \mu_i}{\partial n_n} \right)

        where:
            - :math:`\mu_i` is the chemical potential of molecule :math:`i`
            - :math:`n_j` is the molecule number of molecule :math:`j`
            - :math:`x_j` is the mol fraction of molecule :math:`j`
            - :math:`n_T` is the total number of molecules in system
        """
        # convert to mol fraction
        dmu = self.dmu_dn(units)  # Cache this to avoid repeated calls
        n = self.n_comp - 1

        # chemical potential deriv / mol frac for all molecules until n-1
        dmu_dxs = self.total_molecules[:, np.newaxis, np.newaxis] * (dmu[:, :n, :n] - dmu[:, :n, -1][:, :, np.newaxis])

        # now get the derivative for each component
        dmui_dxi = np.full_like(self.mol_fr, fill_value=np.nan)
        dmui_dxi[:, :-1] = np.diagonal(dmu_dxs, axis1=1, axis2=2)

        # calculate chemical potential deriv for last component
        sum_xi_dmui = (self.mol_fr[:, :-1] * dmui_dxi[:, :-1]).sum(axis=1)
        dmui_dxi[:, -1] = sum_xi_dmui / self.mol_fr[:, -1]
        return np.asarray(dmui_dxi)

    def dlngammas_dxs(self) -> NDArray[np.float64]:
        r"""
        Compute the derivative of natural logarithm of the activity coefficient of molecule :math:`i` with respect to its mol fraction.

        Returns
        -------
        np.ndarray
            A 3D matrix with shape ``(n_sys, n_comp, n_comp)``

        Notes
        -----
        Activity coefficient derivatives, :math:`\frac{\partial \gamma_i}{\partial x_i}` are calculated as follows:

        .. math::
            \frac{\partial \ln{\gamma_i}}{\partial x_i} = \frac{1}{k_b T}\left(\frac{\partial \mu_i}{\partial x_i}\right) - \frac{1}{x_i}

        where:
            - :math:`\mu_i` is the chemical potential of molecule :math:`i`
            - :math:`\gamma_i` is the activity coefficient of molecule :math:`i`
            - :math:`x_i` is the mol fraction of molecule :math:`i`
            - :math:`k_b` is the Boltzmann constant
        """
        if "_dlngammas_dxs" not in self.__dict__:
            # convert zeros to nan to avoid, ZeroDivisionError
            nan_z = self.mol_fr.copy()
            nan_z[nan_z == 0] = np.nan

            # calculate activity derivs
            R = self.ureg("R").to("kJ/mol/K").magnitude
            self._dlngammas_dxs = (1 / (R * self.temperature()))[:, np.newaxis] * self.dmu_dxs("kJ/mol") - 1 / nan_z

        return np.asarray(self._dlngammas_dxs)

    def _get_ref_state_dict(self, mol: str) -> dict[str, object]:
        """Get reference state parameters for each molecule."""
        # get max mol fr at each composition
        z0 = self.mol_fr.copy()
        z0[np.isnan(z0)] = 0
        comp_max = z0.max(axis=1)
        # get mol index
        i = self._mol_idx(mol=mol)
        # get mask for max mol frac at each composition
        is_max = z0[:, i] == comp_max

        # create dict for ref. state values
        # if mol is max at any composition; it cannot be a 'solute'
        if np.any(is_max):
            ref_state_dict = {
                "ref_state": "pure_component",
                "x_initial": 1.0,
                "sorted_idx_val": -1,
                "weight_fn": partial(self._weight_fn, exp_mult=1),
            }
        # if solute, use inf. dil. ref state
        else:
            ref_state_dict = {
                "ref_state": "inf_dilution",
                "x_initial": 0.0,
                "sorted_idx_val": 1,
                "weight_fn": partial(self._weight_fn, exp_mult=-1),
            }
        return ref_state_dict

    def _weight_fn(self, x: NDArray[np.float64], exp_mult: float) -> NDArray[np.float64]:
        try:
            return 100 ** (exp_mult * np.log10(x))
        except ValueError as ve:
            raise ValueError(f"Cannot take log of negative value. Details: {ve}.") from ve

    def ref_state(self, mol: str) -> str:
        r"""
        Get reference state for a molecule.

        Parameters
        ----------
        mol: str
            Molecule name in ``KBThermo.unique_molecules`` names list.

        Returns
        -------
        str
            Either '`pure_component`' or '`inf_dilution`'. Molecule is considered as '`pure_component`' if for any system it is the major component in the system.
        """
        value = self._get_ref_state_dict(mol)["ref_state"]
        if isinstance(value, str):
            return str(value)
        raise TypeError(f"ref_state value must be string, got {type(value)}")

    def _x_initial(self, mol: str) -> float:
        value = self._get_ref_state_dict(mol)["x_initial"]
        if isinstance(value, (float, int)):
            return float(value)
        raise TypeError(f"x_initial value must be numeric, got {type(value)}")

    def _sort_idx_val(self, mol: str) -> int:
        value = self._get_ref_state_dict(mol)["sorted_idx_val"]
        if isinstance(value, int):
            return value
        raise TypeError(f"sorted_idx_val must be int, got {type(value)}")

    def _weights(self, mol: str, x: NDArray[np.float64]) -> NDArray[np.float64]:
        w = self._get_ref_state_dict(mol)["weight_fn"]
        if callable(w):
            return w(x)  # type: ignore
        raise TypeError(f"Expected a callable for weight_fn, got {type(w)}")

    def integrate_dlngammas(
        self, integration_type: str = "numerical", polynomial_degree: int = 5
    ) -> NDArray[np.float64]:
        r"""
        Integrate the derivative of activity coefficients.

        Parameters
        ----------
        integration_type: str
            This determines how the integration will be performed. Options include: numerical, polynomial.
        polynomial_degree: int
            For the '`polynomial`' integration, this specifies the degree of polynomial to fit the derivatives to.

        Returns
        -------
        np.ndarray
            A 2D array with shape ``(n_sys, n_comp)``

        Notes
        -----
        Numerical integration of activity coefficient derivatives occurs through:

        .. math::
            \ln{\gamma_i}(x_i) = \int_{a_0}^{x_i} \left(\frac{\partial \ln{\gamma_i}}{\partial x_i}\right) dx_i \approx \sum_{a=a_0}^{x_i} \frac{\Delta x}{2} \left[\left(\frac{\partial \ln{\gamma_i}}{\partial x_i}\right)_{a} + \left(\frac{\partial \ln{\gamma_i}}{\partial x_i}\right)_{a \pm \Delta x}\right]

        where:
            - :math:`\gamma_i` is the activity coefficient of molecule :math:`i`
            - :math:`x_i` is the mol fraction of molecule :math:`i`
            - :math:`\Delta x` is the step size in :math:`x` between points

        .. note::
            The integral is approximated by a summation using the trapezoidal rule, where the upper limit of summation is :math:`x_i` and the initial condition (or reference state) is :math:`a_0`. Note that the term :math:`a \pm \Delta x` behaves differently based on the value of :math:`a_0`: if :math:`a_0 = 1` (pure component reference state), it becomes :math:`a - \Delta x`, and if :math:`a_0 = 0` (infinite dilution reference state), it becomes :math:`a + \Delta x`.


        Analytical integration of activity coefficient derivatives thorough polynomial fitting occurs by fitting an n-order polynomial function to :math:`\frac{\partial \ln{\gamma_i}}{\partial x_i}`.

        .. note::
            This method takes a set of mole fractions (`xi`) and the corresponding derivatives of :math:`\ln{\gamma}`, fits a polynomial of a specified degree to the derivative data, integrates the polynomial to reconstruct :math:`\ln{\gamma}`, and evaluates :math:`\ln{\gamma}` at the given mol fractions. The integration constant is chosen such that :math:`\ln{\gamma}` obeys boundary conditions of reference state.

        """
        integration_type = integration_type.lower()

        ln_gammas = np.full_like(self.mol_fr, fill_value=np.nan)
        for i, mol in enumerate(self.unique_molecules):
            # get x & dlng for molecule
            xi0 = self.mol_fr[:, i]
            dlng0 = self.dlngammas_dxs()[:, i]
            lng_i = np.full(len(xi0), fill_value=np.nan)

            # filter nan
            nan_mask = (~np.isnan(xi0)) & (~np.isnan(dlng0))
            xi, dlng = xi0[nan_mask], dlng0[nan_mask]

            # if len of True values == 0; no valid mols dln gamma/dxs is found.
            if sum(nan_mask) == 0:
                raise ValueError(f"No real values found for molecule {mol} in dlngammas_dxs.")

            # search for x-initial
            x_initial_found = np.any(np.isclose(xi, self._x_initial(mol)))
            if not x_initial_found:
                xi = np.append(xi, self._x_initial(mol))
                dlng = np.append(dlng, 0)

            # sort by mol fr.
            sorted_idxs = np.argsort(xi)[:: self._sort_idx_val(mol)]
            xi, dlng = xi[sorted_idxs], dlng[sorted_idxs]

            # integrate
            if integration_type == "polynomial":
                lng = self._polynomial_integration(xi, dlng, mol, polynomial_degree)
            elif integration_type == "numerical":
                lng = self._numerical_integration(xi, dlng, mol)
            else:
                raise ValueError(
                    f"Integration type not recognized. Must be `polynomial` or `numerical`, {integration_type} was provided."
                )

            # now prepare data for saving
            inverse_permutation = np.argsort(sorted_idxs)
            lng = lng[inverse_permutation]

            # remove ref. state if added
            if not x_initial_found:
                x_initial_idx = np.where(lng == 0)[0][0]
                lng = np.delete(lng, x_initial_idx)

            try:
                lng_i[nan_mask] = lng  # this makes sure that shape of lng is same as xi
                ln_gammas[:, i] = lng_i
            except ValueError as ve:
                if len(lng) != ln_gammas.shape[0]:
                    raise ValueError(
                        f"Length mismatch between lngammas: {len(lng)} and lngammas matrix: {ln_gammas.shape[0]}. Details: {ve}."
                    ) from ve

        return ln_gammas

    def _polynomial_integration(
        self, xi: np.ndarray, dlng: np.ndarray, mol: str, polynomial_degree: int = 5
    ) -> NDArray[np.float64]:
        # use polynomial to integrate dlng_dxs.
        try:
            dlng_fit = np.poly1d(np.polyfit(xi, dlng, polynomial_degree, w=self._weights(mol, xi)))
        except ValueError as ve:
            if polynomial_degree > len(xi):
                raise ValueError(
                    f"Not enough data points for polynomial fit. Required degree < number points. Details: {ve}."
                ) from ve
            elif len(xi) != len(dlng):
                raise ValueError(
                    f"Length mismatch! Shapes of xi {(len(xi))} and dlng {(len(xi))} do not match. Details: {ve}."
                ) from ve

        # integrate polynomial function to get ln gammas
        lng_fn = dlng_fit.integ(k=0)
        yint = 0 - lng_fn(1)  # adjust for lng=0 at x=1.
        lng_fn = dlng_fit.integ(k=yint)

        # check if _lngamma_fn has been initialized
        if "_lngamma_fn_dict" not in self.__dict__:
            self._lngamma_fn_dict = {}
        if "_dlngamma_fn_dict" not in self.__dict__:
            self._dlngamma_fn_dict = {}

        # add func. to dict
        self._lngamma_fn_dict[mol] = lng_fn
        self._dlngamma_fn_dict[mol] = dlng_fit

        # evalutate lng at xi
        lng = lng_fn(xi)
        return lng

    def _numerical_integration(self, xi: np.ndarray, dlng: np.ndarray, mol: str) -> NDArray[np.float64]:
        # using numerical integration via trapezoid method
        try:
            return np.asarray(cumulative_trapezoid(dlng, xi, initial=0))
        except Exception as e:
            raise Exception(f"Could not perform numerical integration for {mol}. Details: {e}.") from e

    def lngamma_fn(self, mol: str) -> Callable[..., Any]:
        r"""
        Get the integrated polynomial function used to calculate activity coefficients (if integration type is polynomial).

        Parameters
        ----------
        mol: str
            Molecule ID for a molecule in ``KBThermo.unique_molecules``

        Returns
        -------
        np.poly1d
            Polynomial function representing :math:`\ln{\gamma}` of mol
        """
        # retrieve function for ln gamma of mol
        if "_lngamma_fn_dict" not in self.__dict__:
            self.integrate_dlngammas(integration_type="polynomial")
        return self._lngamma_fn_dict[mol]

    def dlngamma_fn(self, mol: str) -> Callable[..., Any]:
        r"""
        Get the polynomial function used to fit activity coefficient derivatives (if integration type is polynomial).

        Parameters
        ----------
        mol: str
            Molecule ID for a molecule in ``KBThermo.unique_molecules``

        Returns
        -------
        np.poly1d
            Polynomial function representing :math:`\frac{\partial \ln{\gamma}}{\partial x}` of mol
        """
        # retrieve function for dln gamma of mol
        if "_dlngamma_fn_dict" not in self.__dict__:
            self.integrate_dlngammas(integration_type="polynomial")
        return self._dlngamma_fn_dict[mol]

    def lngammas(self) -> NDArray[np.float64]:
        r"""
        Results of integrated activity coefficient derivatives according to instance attribute ``gamma_integration_type``.

        Returns
        -------
        np.ndarray
            Activity coefficients as a function of system compositions points according to specificied integration type.

        See Also
        --------
        :meth:`integrate_dlngammas` : Integration of activity coefficient derivatives.

        """
        if "_lngammas" not in self.__dict__:
            self._lngammas = self.integrate_dlngammas(
                integration_type=self.gamma_integration_type,
                polynomial_degree=self.gamma_polynomial_degree,
            )
        return self._lngammas

    def ge(self, units: str = "kJ/mol") -> NDArray[np.float64]:
        r"""
        Gibbs excess free energy calculated from activity coefficients.

        Parameters
        ----------
        units : str
            Units of energy to report values in. Default is 'kJ/mol'.

        Returns
        -------
        np.ndarray
            A 1D array of shape ``(n_sys)``

        Notes
        -----
        Excess free energy, :math:`G^E`, is calculated according to:

        .. math::
            \frac{G^E}{RT} = \sum_{i=1}^n x_i \ln{\gamma_i}

        where:
            - :math:`x_i` is mol fraction of molecule :math:`i`
            - :math:`\gamma_i` is activity coefficient of molecule :math:`i`
        """
        R = self.ureg("R").to(units + "/K").magnitude
        _GE = R * self.temperature(units="K") * (self.mol_fr * self.lngammas()).sum(axis=1)
        return np.asarray(_GE)

    def gid(self, units: str = "kJ/mol") -> NDArray[np.float64]:
        r"""
        Ideal free energy calculated from mol fractions.

        Parameters
        ----------
        units : str
            Units of energy to report values in. Default is 'kJ/mol'.

        Returns
        -------
        np.ndarray
            A 1D array of shape ``(n_sys)``

        Notes
        -----
        Ideal free energy, :math:`G^{id}`, is calculated according to:

        .. math::
            \frac{G^{id}}{RT} = \sum_{i=1}^n x_i \ln{x_i}

        where:
            - :math:`x_i` is mol fraction of molecule :math:`i`
        """
        R = self.ureg("R").to(units + "/K").magnitude
        with np.errstate(divide="ignore", invalid="ignore"):
            _GID = R * self.temperature(units="K") * (self.mol_fr * np.log(self.mol_fr)).sum(axis=1)
        return np.asarray(_GID)

    def gm(self, units: str = "kJ/mol") -> NDArray[np.float64]:
        r"""
        Gibbs mixing free energy calculated from excess and ideal contributions.

        Parameters
        ----------
        units : str
            Units of energy to report values in. Default is 'kJ/mol'.

        Returns
        -------
        np.ndarray
            A 1D array of shape ``(n_sys)``

        Notes
        -----
        Gibbs mixing free energy, :math:`\Delta G_{mix}`, is calculated according to:

        .. math::
            \Delta G_{mix} = G^E + G^{id}
        """
        return self.ge(units) + self.gid(units)

    def hmix(self, units: str = "kJ/mol") -> NDArray[np.float64]:
        r"""
        Mixing enthalpy (excess enthalpy) for each system in specified units.

        Parameters
        ----------
        units : str, optional
            Units for enthalpy. Default is 'kJ/mol'.

        Returns
        -------
        np.ndarray
            A 1D array of mixing enthalpies for each system in specified units.

        Notes
        -----
        This is calculated as the difference between the total enthalpy and the ideal mixing enthalpy.

        .. math::
            \Delta H_{mix} = H_{total} - \sum_{i=1}^n x_i H_i^{pure}

        where:
            - :math:`H_{total}` is the total enthalpy of the system
            - :math:`x_i` is the mol fraction of molecule :math:`i`
            - :math:`H_i^{pure}` is the pure component enthalpy of molecule :math:`i`

        .. note::
            The ideal mixing enthalpy is calculated as a linear combination of pure component enthalpies
            weighted by their mol fractions, thus requiring the pure component enthalpies to be defined under the same conditions as the systems.
        """
        return np.fromiter(self._system_mixing_enthalpy(units=units).values(), dtype=float)

    def se(self, units: str = "kJ/mol") -> NDArray[np.float64]:
        r"""
        Excess entropy determined from Gibbs relation between enthlapy and free energy.

        Parameters
        ----------
        units : str
            Units of energy to report values in. Default is 'kJ/mol'.

        Returns
        -------
        np.ndarray
            A 1D array of shape ``(n_sys)``

        Notes
        -----
        Excess entropy, :math:`S^{E}`, is calculated according to:

        .. math::
            S^E = \frac{\Delta H_{mix} - G^E}{T}
        """
        return (self.hmix(units) - self.ge(units)) / self.temperature(units="K")

    def _property_map(self, energy_units: str = "kJ/mol") -> dict[str, np.ndarray]:
        # returns a dictionary of key properties from analysis
        return {
            "mol_fr": self.mol_fr,
            "kbi": self.kbi_mat(),
            "isotherm_comp": self.isothermal_compressability(units=energy_units),
            "det_hessian": self.det_hessian(units=energy_units),
            "dmu": self.dmu_dxs(units=energy_units),
            "dlngamma": self.dlngammas_dxs(),
            "lngamma": self.lngammas(),
            "ge": self.ge(units=energy_units),
            "gid": self.gid(units=energy_units),
            "gm": self.gm(units=energy_units),
            "se": self.se(units=energy_units),
            "hmix": self.hmix(units=energy_units),
            "i0": self.i0(units="1/cm"),
        }
