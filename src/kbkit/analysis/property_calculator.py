"""
Calculator for molecular dynamic properties and Kirkwood-Buff Integrals (KBIs) as a function of composition.

This calculator operates on a :class:`~kbkit.core.system_collection.SystemCollection` that contains molecular dynamics properties from structure (.gro) and energy (.edr) files.
Additional inputs to :func:`~kbkit.analysis.property_calculator.PropertyCalculator.kbi` are key parameters used for the KBI corrections provided in :class:`~kbkit.analysis.kb_integrator.KBIntegrator`.
"""

from typing import TYPE_CHECKING, Literal

import numpy as np

from kbkit.analysis.kb_integrator import KBIntegrator
from kbkit.parsers.rdf_file import RDFParser
from kbkit.schema.kbi_metadata import KBIMetadata
from kbkit.schema.property_result import PropertyResult
from kbkit.utils.decorators import cached_property_result

if TYPE_CHECKING:
    from kbkit.core.system_collection import SystemCollection


class PropertyCalculator:
    """Thermodynamic property calculator for system collections."""

    def __init__(self, systems: "SystemCollection") -> None:
        self.systems = systems
        self._cache: dict[tuple, PropertyResult] = {}
        self._kbi_metadata: dict[str, dict[str, KBIMetadata]] = {}

    def _get_from_cache(self, key: tuple, target_units: str) -> PropertyResult:
        """Check cache and return converted result if found."""
        if key in self._cache:
            return self._cache[key].to(target_units)
        return None

    def has_all_required_pures(self) -> bool:
        """Check that collection has required pure components for excess properties calculation."""
        if not self.systems.pures:
            return False

        # check that all molecules in mixtures have pure references
        mixture_mols = set(self.systems.molecules)
        pure_mols = set()
        for pure in self.systems.pures:
            pure_mols.update(pure.props.topology.molecules)

        missing = mixture_mols - pure_mols
        if missing:
            return False

        return True

    @cached_property_result()
    def simulated_property(self, name: str, units: str | None = None, avg: bool = True) -> PropertyResult:
        """
        Extract raw values directly from MD simulation (EDR files).

        Returns
        -------
        PropertyResult
            Values as simulated in the MD engine.
        """
        units = units or self.systems.get_units(name)
        return self.systems.get(name, units=units, avg=avg)

    @cached_property_result()
    def pure_property(self, name: str, units: str | None = None, avg: bool = True) -> PropertyResult:
        """
        Extract pure component properties.

        Parameters
        ----------
        name : str
            Property name (e.g., 'Density', 'Volume').
        units : str, optional
            Target units for conversion.
        avg : bool, default True
            Return time-averaged values.

        Returns
        -------
        PropertyResult
            Pure component property values with metadata.
        """
        units = units or self.systems.get_units(name)

        pure_dict = self._build_pure_lookup(name, units, avg)
        return np.array([pure_dict[mol] for mol in self.systems.molecules])

    @cached_property_result()
    def ideal_property(
        self,
        name: str,
        mixing_rule: Literal["linear", "volume_weighted"] = "linear",
        units: str | None = None,
        avg: bool = True,
    ) -> PropertyResult:
        r"""
        Calculate ideal mixing property using specified mixing rule.

        For extensive properties (Volume, Enthalpy):
        .. math::
            Ideal = \Sigma(x_i * V_i^pure) [linear]

        For intensive properties (Density):
        .. math::
            Ideal = 1 / \Sigma(x_i / \rho_i^pure)  [volume-weighted]

        Parameters
        ----------
        name : str
            Property name.
        mixing_rule : {"linear", "volume_weighted"}, default "linear"
            Mixing rule to apply.
        units : str, optional
            Target units.
        avg : bool, default True
            Use time-averaged values.

        Returns
        -------
        PropertyResult
            Ideal property values for each mixture composition.

        Notes
        -----
        For a given property, :math:`P`, the ideal property, :math:`\bar{P}`, is calculated according to:

        .. math::
            \bar{P} = \sum_{i} x_i P_i

        where:
            - :math:`x_i` is the mole fraction of molecule `i`
            - :math:`P_i` is the property for pure component `i`
        """
        units = units or self.systems.get_units(name)

        # 1. Get pure component properties in the correct order (matching self.systems.molecules)
        pure_res = self.pure_property(name=name, units=units, avg=avg)

        # 2. Get mixture compositions (Shape: [N_systems, N_molecules])
        compositions = self.systems.x

        # 3. Vectorized Math
        if "lin" in mixing_rule.lower():
            # Matrix multiplication: [N_systems x N_mols] @ [N_mols] -> [N_systems]
            ideal_values = compositions @ pure_res.value

        elif "vol" in mixing_rule.lower():
            # 1 / sum(x_i / P_i)
            ideal_values = 1.0 / (compositions @ (1.0 / pure_res.value))

        else:
            raise ValueError(f"Unknown mixing rule: {mixing_rule}")

        return ideal_values

    @cached_property_result()
    def excess_property(
        self,
        name: str,
        mixing_rule: Literal["linear", "volume_weighted"] = "linear",
        units: str | None = None,
        avg: bool = True,
    ) -> PropertyResult:
        r"""
        Calculate excess property: Excess = Real - Ideal.

        Parameters
        ----------
        name : str
            Property name.
        mixing_rule : {"linear", "volume_weighted"}, default "linear"
            Mixing rule for ideal calculation.
        units : str, optional
            Target units.
        avg : bool, default True
            Use time-averaged values.

        Returns
        -------
        PropertyResult
            Excess property values.

        Notes
        -----
        For a given property, :math:`P`, the excess property, :math:`P^{EX}`, is calculated according to:

        .. math::
            \begin{aligned}
            P^{EX} &= P - \bar{P} \\
                   &= P - \sum_{i} x_i P_i
            \end{aligned}

        where:
            - :math:`x_i` is the mole fraction of molecule `i`
            - :math:`P` is the property directly from simulation
        """
        units = units or self.systems.get_units(name)

        # Logic: Excess = Simulated - Ideal
        sim_res = self.simulated_property(name=name, units=units, avg=avg)  # No units passed = base units
        ideal_res = self.ideal_property(name=name, units=units, mixing_rule=mixing_rule, avg=avg)

        # Subtract in base units
        return sim_res.value - ideal_res.value

    def _build_pure_lookup(self, name: str, units: str | None = None, avg: bool = True) -> dict[str, float]:
        r"""
        Build a lookup dictionary mapping molecule names to pure property values.

        Parameters
        ----------
        name : str
            Property name.
        units : str, optional
            Target units.
        avg : bool, default True
            Use time-averaged values.

        Returns
        -------
        dict[str, float]
            Mapping of molecule name to pure property value.
        """
        pure_lookup = {}
        for pure_sys in self.systems.pures:
            # Get the molecule name (should be single component)
            mol_counts = pure_sys.props.topology.molecule_count
            if len(mol_counts) != 1:
                raise ValueError(f"Pure system {pure_sys.name} contains multiple molecules: {mol_counts}")

            mol_name = next(iter(mol_counts.keys()))
            pure_value = pure_sys.props.get(name, units=units, avg=avg)
            if isinstance(pure_value, dict):
                pure_value = pure_value[mol_name]
            pure_lookup[mol_name] = pure_value

        return pure_lookup

    def kbi(
        self,
        units: str = "nm^3/molecule",
        ignore_convergence_errors: bool = False,
        convergence_thresholds: tuple = (1e-3, 1e-2),
        tail_length: float | None = None,
        correct_rdf_convergence: bool = True,
        apply_damping: bool = True,
        extrapolate_thermodynamic_limit: bool = True,
    ) -> PropertyResult:
        r"""
        Computes Kirkwood-Buff integrals for molecular systems using RDF data.

        Interfaces with RDFParser and KBIntegrator to extract pairwise KBIs and populate metadata.

        Parameters
        ----------
        units: str, optional
            Units to compute KBI in, molar volume units.
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

        Returns
        -------
        PropertyResult
            KBI Matrix with shape (composition x components x components).

        See Also
        --------
        `KBIntegrator` for a detailed description of KBI calculations and corrections.
        """
        kbi_res = self._compute_kbi(
            units=units,
            ignore_convergence_errors=ignore_convergence_errors,
            convergence_thresholds=convergence_thresholds,
            tail_length=tail_length,
            correct_rdf_convergence=correct_rdf_convergence,
            apply_damping=apply_damping,
            extrapolate_thermodynamic_limit=extrapolate_thermodynamic_limit,
        )
        # add kbi metdata to PropertyResult object
        kbi_res.metadata = self._kbi_metadata
        return kbi_res

    @cached_property_result(default_units="nm^3/molecule")
    def _compute_kbi(
        self,
        units: str = "nm^3/molecule",
        ignore_convergence_errors: bool = False,
        convergence_thresholds: tuple = (1e-3, 1e-2),
        tail_length: float | None = None,
        correct_rdf_convergence: bool = True,
        apply_damping: bool = True,
        extrapolate_thermodynamic_limit: bool = True,
    ) -> PropertyResult:
        """Workhouse of KBI Calculation, so that metadata can be assigned in the actual function."""
        units = units or "nm^3/molecule"
        kbis = np.full((len(self.systems), len(self.systems.molecules), len(self.systems.molecules)), fill_value=np.nan)

        for s, meta in enumerate(self.systems):
            if not meta.has_rdf():
                continue
            # get all RDF files
            all_files = sorted(meta.rdf_path.iterdir())
            rdf_files = [f for f in all_files if f.suffix in (".xvg", ".txt")]

            for fpath in rdf_files:
                rdf = RDFParser(path=fpath, convergence_thresholds=convergence_thresholds, tail_length=tail_length)

                integrator = KBIntegrator.from_system_properties(
                    rdf=rdf,
                    system_properties=meta.props,
                    correct_rdf_convergence=correct_rdf_convergence,
                    apply_damping=apply_damping,
                    extrapolate_thermodynamic_limit=extrapolate_thermodynamic_limit,
                )

                mol_i, mol_j = integrator.rdf_molecules
                i, j = [list(self.systems.molecules).index(mol) for mol in integrator.rdf_molecules]

                if rdf.is_converged:
                    kbis[s, i, j] = integrator.compute_kbi(mol_j)
                    kbis[s, j, i] = integrator.compute_kbi(mol_i)

                # override convergence check to skip system if not converged
                else:  # for not converged rdf
                    msg = f"RDF for system '{meta.name}' and pair {integrator.rdf_molecules} did not converge."
                    if ignore_convergence_errors:
                        print(f"WARNING: {msg} Skipping this system.")
                        continue
                    else:
                        raise RuntimeError(msg)

                # add values to metadata
                self._kbi_metadata.setdefault(meta.name, {})[".".join(integrator.rdf_molecules)] = KBIMetadata(
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

        return kbis
