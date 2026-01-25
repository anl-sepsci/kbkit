"""
Calculator for Kirkwood-Buff Integrals (KBIs) as a function of composition.

This calculator operates on a :class:`~kbkit.systems.collection.SystemCollection` that contains molecular dynamics properties from structure (.gro) and energy (.edr) files.
Additional inputs are key parameters used for the KBI corrections provided in :class:`~kbkit.kbi.integrator.KBIntegrator`.
"""

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
        kbis = np.full((len(self.systems), len(self.systems.molecules), len(self.systems.molecules)), fill_value=np.nan)
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
                i, j = [list(self.systems.molecules).index(mol) for mol in integrator.rdf_molecules]

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
