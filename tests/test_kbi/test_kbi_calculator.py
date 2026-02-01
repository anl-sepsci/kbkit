"""
Unit tests for KBICalculator - targeting >95 % branch + line coverage.

Mocking strategy
----------------
* ``SystemCollection`` - thin dataclass-style mock with the attributes the
  calculator actually reads: ``charges``, ``residue_molecules``,
  ``electrolyte_molecules``, ``residue_counts``, ``n_sys``, and iteration
  (``__iter__`` / ``__len__``).
* ``RdfParser`` - mocked at the module level so every instantiation returns a
  controllable stub (``is_converged``, ``r``, ``g``, ``r_tail``).
* ``KBIntegrator`` - mocked via ``from_system_properties``; the returned
  instance exposes ``rdf_molecules``, ``compute_kbi``, ``rkbi``,
  ``scaled_rkbi``, ``scaled_rkbi_fit``, ``fit_limit_params``.
* ``PropertyResult`` - real-ish wrapper; we mock only ``.to()`` to return
  *self* so unit-conversion round-trips don't need a real converter.
* ``KBIAnalysisPlotter`` - mocked to verify it is constructed with the right
  arguments.
"""
from __future__ import annotations

import warnings

# Suppress NumPy/SciPy compatibility warning (harmless with NumPy 2.x + SciPy 1.16+)
warnings.filterwarnings('ignore', message='numpy.ndarray size changed', category=RuntimeWarning)


from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Paths for patching - must match the *import* location inside the module
# ---------------------------------------------------------------------------
MOD = "kbkit.kbi.calculator"  # adjust if the module lives elsewhere
_PATCH_RDF      = f"{MOD}.RdfParser"
_PATCH_INTEG    = f"{MOD}.KBIntegrator"
_PATCH_PLOTTER  = f"{MOD}.KBIAnalysisPlotter"


# ===========================================================================
# Helper factories
# ===========================================================================


def _make_system_meta(
    name: str = "sys0",
    has_rdf: bool = True,
    rdf_files: list[str] | None = None,
    props: object | None = None,
):
    """Return a single system-meta stub (one element yielded by SystemCollection)."""
    if rdf_files is None:
        rdf_files = ["AB.xvg"]

    meta = MagicMock()
    meta.name = name
    meta.has_rdf.return_value = has_rdf
    meta.props = props or MagicMock()

    # Build a fake rdf_path directory listing.
    # These must support < comparison because the source calls sorted() on
    # the result of iterdir().  A plain MagicMock does not implement __lt__,
    # so we use a thin wrapper that delegates ordering to the filename string.
    class _SortablePath:
        def __init__(self, name: str):
            self._name = name
            self.suffix = Path(name).suffix

        def __lt__(self, other):
            return self._name < other._name

        def __eq__(self, other):
            return self._name == other._name

        def __hash__(self):
            return hash(self._name)

        def __repr__(self):
            return f"_SortablePath({self._name!r})"

    meta.rdf_path = MagicMock()
    path_objects = [_SortablePath(fname) for fname in rdf_files]
    meta.rdf_path.iterdir.return_value = path_objects

    return meta


def _make_systems(
    n_sys: int = 2,
    residue_molecules: list[str] | None = None,
    electrolyte_molecules: list[str] | None = None,
    charges: list | None = None,
    residue_counts: np.ndarray | None = None,
    metas: list | None = None,
):
    """Return a mock SystemCollection."""
    if residue_molecules is None:
        residue_molecules = ["A", "B"]
    if electrolyte_molecules is None:
        electrolyte_molecules = residue_molecules

    systems = MagicMock()
    systems.charges = charges  # None / falsy  → residue path; truthy → electrolyte
    systems.residue_molecules = residue_molecules
    systems.electrolyte_molecules = electrolyte_molecules
    systems.n_sys = n_sys

    if residue_counts is None:
        residue_counts = np.ones((n_sys, len(residue_molecules)), dtype=float)
    systems.residue_counts = residue_counts

    if metas is None:
        metas = [_make_system_meta(name=f"sys{i}") for i in range(n_sys)]

    systems.__iter__ = MagicMock(return_value=iter(metas))
    systems.__len__ = MagicMock(return_value=n_sys)

    return systems


def _stub_integrator(rdf_molecules=("A", "B"), kbi_val=1.5):
    """Return a mock KBIntegrator instance with sensible defaults."""
    integ = MagicMock()
    integ.rdf_molecules = rdf_molecules
    integ.compute_kbi.return_value = kbi_val
    integ.rkbi.return_value = np.array([0.1, 0.2])
    integ.scaled_rkbi.return_value = np.array([0.05, 0.1])
    integ.scaled_rkbi_fit.return_value = np.array([0.08])
    integ.fit_limit_params.return_value = np.array([0.0, kbi_val])
    return integ


def _stub_rdf(converged: bool = True):
    """Return a mock RdfParser instance."""
    rdf = MagicMock()
    rdf.is_converged = converged
    rdf.r = np.linspace(0, 5, 50)
    rdf.g = np.ones(50)
    rdf.r_tail = 4.5
    return rdf


# ===========================================================================
# Import the class under test (after helpers are defined so patches work)
# ===========================================================================
from kbkit.kbi.calculator import KBICalculator

# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture
def two_component_systems():
    """2-system, 2-component collection with no charges (residue path)."""
    metas = [
        _make_system_meta(name="sys0", rdf_files=["AB.xvg"]),
        _make_system_meta(name="sys1", rdf_files=["AB.xvg"]),
    ]
    return _make_systems(n_sys=2, residue_molecules=["A", "B"], metas=metas)


@pytest.fixture
def electrolyte_systems():
    """
    3-component residue system (Na, Cl, Water) mapped to
    2 electrolyte-level species: NaCl (salt) and Water (molecule).
    Charges present → electrolyte path.
    """
    residues = ["Na", "Cl", "Water"]
    electrolytes = ["Na.Cl", "Water"]
    counts = np.array([
        [10, 10, 100],   # sys0
        [5,  5,  200],   # sys1
    ], dtype=float)
    metas = [
        _make_system_meta(name="sys0", rdf_files=["NaCl.xvg", "NaW.xvg", "ClW.xvg"]),
        _make_system_meta(name="sys1", rdf_files=["NaCl.xvg", "NaW.xvg", "ClW.xvg"]),
    ]
    return _make_systems(
        n_sys=2,
        residue_molecules=residues,
        electrolyte_molecules=electrolytes,
        charges=[1, -1, 0],  # truthy → triggers electrolyte path
        residue_counts=counts,
        metas=metas,
    )


# ===========================================================================
# 1. __init__ - parameter storage
# ===========================================================================


class TestInit:
    def test_defaults_stored(self, two_component_systems):
        calc = KBICalculator(two_component_systems)
        assert calc.systems is two_component_systems
        assert calc.ignore_convergence_errors is False
        assert calc.convergence_thresholds == (1e-3, 1e-2)
        assert calc.tail_length is None
        assert calc.correct_rdf_convergence is True
        assert calc.apply_damping is True
        assert calc.extrapolate_thermodynamic_limit is True
        assert calc._cache == {}

    def test_custom_params_stored(self, two_component_systems):
        calc = KBICalculator(
            two_component_systems,
            ignore_convergence_errors=True,
            convergence_thresholds=(1e-4, 5e-2),
            tail_length=3.0,
            correct_rdf_convergence=False,
            apply_damping=False,
            extrapolate_thermodynamic_limit=False,
        )
        assert calc.ignore_convergence_errors is True
        assert calc.convergence_thresholds == (1e-4, 5e-2)
        assert calc.tail_length == 3.0
        assert calc.correct_rdf_convergence is False
        assert calc.apply_damping is False
        assert calc.extrapolate_thermodynamic_limit is False


# ===========================================================================
# 2. kbi() - routing logic
# ===========================================================================


class TestKbiRouting:
    @patch.object(KBICalculator, "residue_kbi")
    def test_routes_to_residue_when_no_charges(self, mock_res, two_component_systems):
        two_component_systems.charges = None  # falsy
        calc = KBICalculator(two_component_systems)
        calc.kbi(units="nm^3/molecule")
        mock_res.assert_called_once_with("nm^3/molecule")

    @patch.object(KBICalculator, "electrolyte_kbi")
    def test_routes_to_electrolyte_when_charges_present(self, mock_elec, electrolyte_systems):
        calc = KBICalculator(electrolyte_systems)
        calc.kbi(units="nm^3/molecule")
        mock_elec.assert_called_once_with("nm^3/molecule")


# ===========================================================================
# 3. residue_kbi() - full branch coverage
# ===========================================================================


class TestResidueKbi:
    # --- happy path: converged RDFs, symmetric assignment ----------------

    @patch(_PATCH_INTEG)
    @patch(_PATCH_RDF)
    def test_happy_path_converged(self, MockRdf, MockInteg, two_component_systems):
        rdf_stub = _stub_rdf(converged=True)
        MockRdf.return_value = rdf_stub

        integ_stub = _stub_integrator(rdf_molecules=("A", "B"), kbi_val=2.0)
        MockInteg.from_system_properties.return_value = integ_stub

        calc = KBICalculator(two_component_systems)
        result = calc.residue_kbi()

        # shape: (n_sys=2, n_mol=2, n_mol=2)
        assert result.value.shape == (2, 2, 2)
        # symmetric: [s, 0, 1] and [s, 1, 0] should both be filled (not NaN)
        for s in range(2):
            assert not np.isnan(result.value[s, 0, 1])
            assert not np.isnan(result.value[s, 1, 0])

        # Integrator was called with correct flags
        MockInteg.from_system_properties.assert_called()
        _, kwargs = MockInteg.from_system_properties.call_args
        assert kwargs["correct_rdf_convergence"] is True
        assert kwargs["apply_damping"] is True
        assert kwargs["extrapolate_thermodynamic_limit"] is True

    # --- system with no RDF directory is skipped (values stay NaN) --------

    @patch(_PATCH_INTEG)
    @patch(_PATCH_RDF)
    def test_system_without_rdf_stays_nan(self, MockRdf, MockInteg):
        metas = [
            _make_system_meta(name="sys0", has_rdf=False),   # skipped
            _make_system_meta(name="sys1", rdf_files=["AB.xvg"]),
        ]
        systems = _make_systems(n_sys=2, residue_molecules=["A", "B"], metas=metas)

        rdf_stub = _stub_rdf(converged=True)
        MockRdf.return_value = rdf_stub
        integ_stub = _stub_integrator(rdf_molecules=("A", "B"), kbi_val=3.0)
        MockInteg.from_system_properties.return_value = integ_stub

        calc = KBICalculator(systems)
        result = calc.residue_kbi()

        # sys0 should remain all-NaN
        assert np.all(np.isnan(result.value[0]))
        # sys1 should have values
        assert not np.isnan(result.value[1, 0, 1])

    # --- non-converged RDF, ignore_convergence_errors=True → warning + skip

    @patch(_PATCH_INTEG)
    @patch(_PATCH_RDF)
    def test_non_converged_ignored(self, MockRdf, MockInteg, two_component_systems, capsys):
        rdf_stub = _stub_rdf(converged=False)
        MockRdf.return_value = rdf_stub
        integ_stub = _stub_integrator(rdf_molecules=("A", "B"))
        MockInteg.from_system_properties.return_value = integ_stub

        calc = KBICalculator(two_component_systems, ignore_convergence_errors=True)
        result = calc.residue_kbi()

        captured = capsys.readouterr()
        assert "WARNING" in captured.out
        assert "did not converge" in captured.out
        # All values remain NaN because every RDF was non-converged
        assert np.all(np.isnan(result.value))

    # --- non-converged RDF, ignore_convergence_errors=False → RuntimeError

    @patch(_PATCH_INTEG)
    @patch(_PATCH_RDF)
    def test_non_converged_raises(self, MockRdf, MockInteg, two_component_systems):
        rdf_stub = _stub_rdf(converged=False)
        MockRdf.return_value = rdf_stub
        integ_stub = _stub_integrator(rdf_molecules=("A", "B"))
        MockInteg.from_system_properties.return_value = integ_stub

        calc = KBICalculator(two_component_systems, ignore_convergence_errors=False)
        with pytest.raises(RuntimeError, match="did not converge"):
            calc.residue_kbi()

    # --- cache hit returns immediately without re-computing ---------------

    @patch(_PATCH_INTEG)
    @patch(_PATCH_RDF)
    def test_cache_hit(self, MockRdf, MockInteg, two_component_systems):
        rdf_stub = _stub_rdf(converged=True)
        MockRdf.return_value = rdf_stub
        integ_stub = _stub_integrator(rdf_molecules=("A", "B"), kbi_val=5.0)
        MockInteg.from_system_properties.return_value = integ_stub

        calc = KBICalculator(two_component_systems)

        # First call populates cache
        calc.residue_kbi()
        call_count_after_first = MockRdf.call_count

        # Reset iterator for a potential second pass (won't happen due to cache)
        two_component_systems.__iter__ = MagicMock(
            return_value=iter([
                _make_system_meta(name="sys0", rdf_files=["AB.xvg"]),
                _make_system_meta(name="sys1", rdf_files=["AB.xvg"]),
            ])
        )

        # Second call - should hit cache
        calc.residue_kbi()
        # RdfParser should NOT have been called again
        assert MockRdf.call_count == call_count_after_first

    # --- units default when None is passed --------------------------------

    @patch(_PATCH_INTEG)
    @patch(_PATCH_RDF)
    def test_units_default_none(self, MockRdf, MockInteg, two_component_systems):
        rdf_stub = _stub_rdf(converged=True)
        MockRdf.return_value = rdf_stub
        integ_stub = _stub_integrator(rdf_molecules=("A", "B"))
        MockInteg.from_system_properties.return_value = integ_stub

        calc = KBICalculator(two_component_systems)
        # Should not raise even with None
        result = calc.residue_kbi(units=None)
        assert result is not None

    # --- metadata is populated correctly ----------------------------------

    @patch(_PATCH_INTEG)
    @patch(_PATCH_RDF)
    def test_metadata_populated(self, MockRdf, MockInteg, two_component_systems):
        rdf_stub = _stub_rdf(converged=True)
        MockRdf.return_value = rdf_stub
        integ_stub = _stub_integrator(rdf_molecules=("A", "B"), kbi_val=1.0)
        MockInteg.from_system_properties.return_value = integ_stub

        calc = KBICalculator(two_component_systems)
        result = calc.residue_kbi()

        # metadata should have entries for each system
        assert "sys0" in result.metadata
        assert "sys1" in result.metadata
        # each system should have an entry keyed by "A.B"
        assert "A.B" in result.metadata["sys0"]
        assert "A.B" in result.metadata["sys1"]


# ===========================================================================
# 4. electrolyte_kbi() - salt transformation cases
# ===========================================================================


class TestElectrolyteKbi:
    """
    We inject a *pre-built* residue KBI array via the cache so that
    electrolyte_kbi() skips the RDF/integration machinery entirely and we can
    focus purely on the salt-transformation algebra.
    """

    def _seed_residue_cache(self, calc, kbis: np.ndarray):
        """Manually seed the residue_kbi cache so electrolyte_kbi uses it."""
        from kbkit.schema.property_result import PropertyResult

        result = MagicMock()
        result.value = kbis
        result.to = MagicMock(return_value=result)
        calc._cache[("kbi",)] = result

    # --- Case 1: salt ↔ salt (both species are "X.Y") --------------------

    def test_salt_salt(self, electrolyte_systems):
        # residue order: Na=0, Cl=1, Water=2
        # electrolyte order: Na.Cl=0, Water=1
        # We set known ion-ion KBI values
        kbis = np.zeros((2, 3, 3))
        kbis[:, 0, 0] = 1.0   # Na-Na
        kbis[:, 0, 1] = 2.0   # Na-Cl
        kbis[:, 1, 0] = 2.0   # Cl-Na
        kbis[:, 1, 1] = 3.0   # Cl-Cl

        calc = KBICalculator(electrolyte_systems)
        self._seed_residue_cache(calc, kbis)

        result = calc.electrolyte_kbi()

        # xc = xNa = 10/(10+10) = 0.5, xa = xCl = 0.5  (same salt both sides)
        # G_salt_salt = 0.5*0.5*1 + 0.5*0.5*2 + 0.5*0.5*2 + 0.5*0.5*3 = 2.0
        expected = 0.25 * (1.0 + 2.0 + 2.0 + 3.0)  # = 2.0
        np.testing.assert_allclose(result.value[:, 0, 0], expected)
        # Symmetric
        np.testing.assert_allclose(result.value[:, 0, 0], result.value[:, 0, 0])

    # --- Case 2: salt ↔ molecule ------------------------------------------

    def test_salt_molecule(self, electrolyte_systems):
        kbis = np.zeros((2, 3, 3))
        kbis[:, 2, 0] = 4.0   # Water-Na
        kbis[:, 0, 2] = 4.0
        kbis[:, 2, 1] = 6.0   # Water-Cl
        kbis[:, 1, 2] = 6.0

        calc = KBICalculator(electrolyte_systems)
        self._seed_residue_cache(calc, kbis)

        result = calc.electrolyte_kbi()

        # G_Water_NaCl = xNa * G_Water_Na + xCl * G_Water_Cl
        #              = 0.5 * 4.0 + 0.5 * 6.0 = 5.0
        np.testing.assert_allclose(result.value[:, 1, 0], 5.0)
        # Symmetric
        np.testing.assert_allclose(result.value[:, 0, 1], result.value[:, 1, 0])

    # --- Case 3: molecule ↔ molecule (direct look-up) --------------------

    def test_molecule_molecule(self):
        # 2 neutral molecules only, no salt
        residues = ["Water", "Ethanol"]
        electrolytes = ["Water", "Ethanol"]
        systems = _make_systems(
            n_sys=1,
            residue_molecules=residues,
            electrolyte_molecules=electrolytes,
            charges=[0, 0],  # truthy (non-empty list) → electrolyte path
            residue_counts=np.array([[50, 30]], dtype=float),
            metas=[_make_system_meta(name="sys0")],
        )

        kbis = np.array([[[1.0, 2.0], [2.0, 3.0]]])  # shape (1, 2, 2)

        calc = KBICalculator(systems)
        # seed residue cache
        result_mock = MagicMock()
        result_mock.value = kbis
        result_mock.to = MagicMock(return_value=result_mock)
        calc._cache[("kbi",)] = result_mock

        result = calc.electrolyte_kbi()
        np.testing.assert_array_equal(result.value[0], kbis[0])

    # --- cache hit on electrolyte_kbi -------------------------------------

    def test_electrolyte_cache_hit(self, electrolyte_systems):
        calc = KBICalculator(electrolyte_systems)

        sentinel = MagicMock()
        sentinel.to = MagicMock(return_value=sentinel)
        calc._cache[("electrolyte_kbi",)] = sentinel

        result = calc.electrolyte_kbi(units="nm^3/molecule")
        sentinel.to.assert_called_once_with("nm^3/molecule")
        assert result is sentinel

    # --- ValueError: salt species not in residue list ----------------------

    def test_salt_salt_species_not_found(self):
        """Both electrolyte species are salts but one ion is missing from residues."""
        residues = ["Na", "Cl"]  # missing "K"
        electrolytes = ["Na.Cl", "K.Cl"]  # K not in residues
        systems = _make_systems(
            n_sys=1,
            residue_molecules=residues,
            electrolyte_molecules=electrolytes,
            charges=[1, -1],
            residue_counts=np.array([[5, 5]], dtype=float),
            metas=[_make_system_meta(name="sys0")],
        )

        kbis = np.zeros((1, 2, 2))
        calc = KBICalculator(systems)
        result_mock = MagicMock()
        result_mock.value = kbis
        result_mock.to = MagicMock(return_value=result_mock)
        calc._cache[("kbi",)] = result_mock

        with pytest.raises(ValueError, match="not found in original molecules"):
            calc.electrolyte_kbi()

    def test_salt_molecule_species_not_found(self):
        """One salt, one molecule - molecule missing from residues."""
        residues = ["Na", "Cl"]  # missing "Water"
        electrolytes = ["Na.Cl", "Water"]
        systems = _make_systems(
            n_sys=1,
            residue_molecules=residues,
            electrolyte_molecules=electrolytes,
            charges=[1, -1],
            residue_counts=np.array([[5, 5]], dtype=float),
            metas=[_make_system_meta(name="sys0")],
        )

        kbis = np.zeros((1, 2, 2))
        calc = KBICalculator(systems)
        result_mock = MagicMock()
        result_mock.value = kbis
        result_mock.to = MagicMock(return_value=result_mock)
        calc._cache[("kbi",)] = result_mock

        with pytest.raises(ValueError, match="not found in original molecules"):
            calc.electrolyte_kbi()

    def test_molecule_molecule_species_not_found(self):
        """Both are single molecules but one is missing from residues."""
        residues = ["Water"]  # missing "Ethanol"
        electrolytes = ["Water", "Ethanol"]
        systems = _make_systems(
            n_sys=1,
            residue_molecules=residues,
            electrolyte_molecules=electrolytes,
            charges=[0],
            residue_counts=np.array([[100]], dtype=float),
            metas=[_make_system_meta(name="sys0")],
        )

        kbis = np.zeros((1, 1, 1))
        calc = KBICalculator(systems)
        result_mock = MagicMock()
        result_mock.value = kbis
        result_mock.to = MagicMock(return_value=result_mock)
        calc._cache[("kbi",)] = result_mock

        with pytest.raises(ValueError, match="not found in original molecules"):
            calc.electrolyte_kbi()


# ===========================================================================
# 5. _parse_species()
# ===========================================================================


class TestParseSpecies:
    def _calc(self):
        """Minimal calculator - only _parse_species is called."""
        systems = _make_systems(n_sys=1)
        return KBICalculator(systems)

    def test_single_molecule(self):
        assert self._calc()._parse_species("Water") == ("Water",)

    def test_salt_two_parts(self):
        assert self._calc()._parse_species("Na.Cl") == ("Na", "Cl")

    def test_too_many_parts_raises(self):
        with pytest.raises(ValueError, match="Invalid species name"):
            self._calc()._parse_species("A.B.C")


# ===========================================================================
# 6. _ion_fraction()
# ===========================================================================


class TestIonFraction:
    def test_normal_fractions(self):
        counts = np.array([
            [20, 80],   # sys0: xc=0.2  xa=0.8
            [50, 50],   # sys1: xc=0.5  xa=0.5
        ], dtype=float)
        systems = _make_systems(
            n_sys=2,
            residue_molecules=["Cat", "An"],
            residue_counts=counts,
        )
        calc = KBICalculator(systems)
        xc, xa = calc._ion_fraction(0, 1)

        np.testing.assert_allclose(xc, [0.2, 0.5])
        np.testing.assert_allclose(xa, [0.8, 0.5])

    def test_zero_denominator_returns_zero(self):
        """When both ion counts are 0 the salt is absent; fractions must be 0."""
        counts = np.array([
            [0, 0],     # salt absent
            [3, 7],     # normal
        ], dtype=float)
        systems = _make_systems(
            n_sys=2,
            residue_molecules=["Cat", "An"],
            residue_counts=counts,
        )
        calc = KBICalculator(systems)
        xc, xa = calc._ion_fraction(0, 1)

        # sys0: both zero
        assert xc[0] == 0.0
        assert xa[0] == 0.0
        # sys1: normal
        np.testing.assert_allclose(xc[1], 0.3)
        np.testing.assert_allclose(xa[1], 0.7)


# ===========================================================================
# 7. kbi_plotter()
# ===========================================================================


class TestKbiPlotter:
    @patch(_PATCH_PLOTTER)
    @patch.object(KBICalculator, "kbi")
    def test_plotter_constructed_correctly(self, mock_kbi, MockPlotter):
        sentinel_result = MagicMock()
        mock_kbi.return_value = sentinel_result

        systems = _make_systems(n_sys=1)
        calc = KBICalculator(systems)

        mol_map = {"A": "Molecule A"}
        calc.kbi_plotter(molecule_map=mol_map)

        MockPlotter.assert_called_once_with(kbi=sentinel_result, molecule_map=mol_map)

    @patch(_PATCH_PLOTTER)
    @patch.object(KBICalculator, "kbi")
    def test_plotter_default_molecule_map_none(self, mock_kbi, MockPlotter):
        mock_kbi.return_value = MagicMock()

        systems = _make_systems(n_sys=1)
        calc = KBICalculator(systems)
        calc.kbi_plotter()

        MockPlotter.assert_called_once_with(kbi=mock_kbi.return_value, molecule_map=None)


# ===========================================================================
# 8. Integration-style: end-to-end residue_kbi → electrolyte_kbi pipeline
# ===========================================================================


class TestEndToEnd:
    @patch(_PATCH_INTEG)
    @patch(_PATCH_RDF)
    def test_residue_then_electrolyte_uses_cache(self, MockRdf, MockInteg):
        """
        After residue_kbi populates the cache, electrolyte_kbi should read
        from it rather than re-invoking RdfParser.
        """
        residues = ["Na", "Cl", "Water"]
        electrolytes = ["Na.Cl", "Water"]
        counts = np.array([[10, 10, 100]], dtype=float)

        # Build metas with all 3 pair files
        pair_files = ["NaCl.xvg", "NaW.xvg", "ClW.xvg"]
        metas = [_make_system_meta(name="sys0", rdf_files=pair_files)]

        systems = _make_systems(
            n_sys=1,
            residue_molecules=residues,
            electrolyte_molecules=electrolytes,
            charges=[1, -1, 0],
            residue_counts=counts,
            metas=metas,
        )

        # Stub RDF + Integrator - return different KBI per pair
        rdf_stub = _stub_rdf(converged=True)
        MockRdf.return_value = rdf_stub

        pair_kbis = {"Na.Cl": 1.0, "Na.Water": 2.0, "Cl.Water": 3.0}
        call_idx = {"i": 0}
        pairs_order = [("Na", "Cl"), ("Na", "Water"), ("Cl", "Water")]

        def make_integrator(**kwargs):
            idx = call_idx["i"]
            pair = pairs_order[idx % len(pairs_order)]
            call_idx["i"] += 1
            return _stub_integrator(rdf_molecules=pair, kbi_val=pair_kbis[".".join(pair)])

        MockInteg.from_system_properties.side_effect = make_integrator

        calc = KBICalculator(systems)

        # First call builds residue cache
        res_result = calc.residue_kbi()
        rdf_call_count = MockRdf.call_count

        # electrolyte_kbi should NOT trigger more RdfParser calls
        elec_result = calc.electrolyte_kbi()
        assert MockRdf.call_count == rdf_call_count

        # Basic shape check
        assert elec_result.value.shape == (1, 2, 2)


# ===========================================================================
# 9. Edge cases
# ===========================================================================


class TestEdgeCases:
    @patch(_PATCH_INTEG)
    @patch(_PATCH_RDF)
    def test_empty_rdf_directory(self, MockRdf, MockInteg):
        """System has rdf_path but the directory is empty - no files processed."""
        meta = _make_system_meta(name="sys0", rdf_files=[])
        systems = _make_systems(n_sys=1, residue_molecules=["A", "B"], metas=[meta])

        calc = KBICalculator(systems)
        result = calc.residue_kbi()

        MockRdf.assert_not_called()
        # Everything stays NaN
        assert np.all(np.isnan(result.value))

    @patch(_PATCH_INTEG)
    @patch(_PATCH_RDF)
    def test_non_xvg_txt_files_are_skipped(self, MockRdf, MockInteg):
        """Files with non .xvg/.txt suffixes in rdf_path are ignored."""
        meta = _make_system_meta(name="sys0", rdf_files=["readme.md", "data.csv"])
        systems = _make_systems(n_sys=1, residue_molecules=["A", "B"], metas=[meta])

        calc = KBICalculator(systems)
        result = calc.residue_kbi()

        MockRdf.assert_not_called()
        assert np.all(np.isnan(result.value))

    def test_ion_fraction_single_system(self):
        """Single-system edge case for _ion_fraction."""
        counts = np.array([[7, 3]], dtype=float)
        systems = _make_systems(n_sys=1, residue_molecules=["C", "A"], residue_counts=counts)
        calc = KBICalculator(systems)

        xc, xa = calc._ion_fraction(0, 1)
        np.testing.assert_allclose(xc, [0.7])
        np.testing.assert_allclose(xa, [0.3])
