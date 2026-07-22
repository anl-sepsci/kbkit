"""Unit tests for KBIntegrator class."""

import warnings
from pathlib import Path
from unittest.mock import MagicMock, PropertyMock, patch

import numpy as np
import pytest

warnings.filterwarnings("ignore")

_MODULE = "kbkit.kbi.integrator"
_RDF_PARSER = f"{_MODULE}.RdfParser"
_LOAD_STYLE = f"{_MODULE}.load_mplstyle"
_HANDLE_ERROR = f"{_MODULE}.handle_error"

from kbkit.io.rdf import RdfParser
from kbkit.kbi.integrator import KBIntegrator
from kbkit.utils.exceptions import KBIConvergenceError, LinearityError


# ===========================================================================
# Helpers / Factories
# ===========================================================================
def _make_rdf(
    n: int = 200,
    r_max: float = 4.0,
    amplitude: float = 0.5,
    decay: float = 2.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate a physically plausible RDF:
        g(r) = 1 + amplitude * exp(-decay * r) * cos(2π r)
    with g(r) → 1 at large r.
    """
    r = np.linspace(0.1, r_max, n)
    gr = 1.0 + amplitude * np.exp(-decay * r) * np.cos(2 * np.pi * r)
    gr = np.clip(gr, 0.0, None)
    return r, gr


def _make_converged_rdf(
    n: int = 500,
    r_max: float = 6.0,
    decay: float = 15.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate an RDF whose running KBI converges to a flat plateau
    almost immediately, guaranteeing R² > 0.99 for geometric extrapolation.

    Design rationale
    ----------------
    The geometric running KBI is:

        G^V(L) = ∫₀ᴸ h(r) w(r) dr

    For G^V(L) to be nearly constant (flat plateau) over most of [0, r_max],
    h(r) must decay to zero very rapidly.  We use:

        g(r) = 1 + A * exp(-decay * r)

    with a very large decay constant so that h(r) = g(r) - 1 is essentially
    zero beyond r ≈ 0.3 nm.  This means G^V(L) stops changing after the
    first few points, and L * G^V(L) is then perfectly linear in L with
    slope = G^∞ and near-zero residuals (R² ≈ 1).

    Parameters
    ----------
    n : int
        Number of radial grid points.
    r_max : float
        Maximum radial distance (nm). Larger values give more linear range.
    decay : float
        Exponential decay rate (nm⁻¹). Higher = faster convergence.
    """
    r = np.linspace(0.05, r_max, n)
    gr = 1.0 + 3.0 * np.exp(-decay * r)
    gr = np.clip(gr, 0.0, None)
    return r, gr


def _make_kbi(
    n: int = 200,
    r_max: float = 4.0,
    n_ref: int = 500,
    box_volume: float = 216.0,  # 6³ nm³ — large box reduces vdV correction
    delta: int = 0,
    weight_type: str = "geometric",
    errors: str = "raise",
    force: bool = False,
    use_converged_rdf: bool = False,
) -> KBIntegrator:
    """Construct a KBIntegrator with a synthetic RDF."""
    if use_converged_rdf:
        r, gr = _make_converged_rdf(n=n, r_max=r_max)
    else:
        r, gr = _make_rdf(n=n, r_max=r_max)
    return KBIntegrator(
        r=r,
        gr=gr,
        n_ref=n_ref,
        box_volume=box_volume,
        delta=delta,
        weight_type=weight_type,
        errors=errors,
        force=force,
    )


def _make_mock_rdf_parser(
    r: np.ndarray | None = None,
    gr: np.ndarray | None = None,
    filepath: Path = Path("AB.xvg"),
    mol_list: list[str] | None = None,
) -> MagicMock:
    """
    Return a MagicMock with spec=RdfParser so isinstance() checks pass.
    This is critical — without spec=RdfParser the from_rdf TypeError fires.
    """
    if r is None:
        r, gr = _make_rdf()
    # spec=RdfParser makes isinstance(mock, RdfParser) return True
    parser = MagicMock(spec=RdfParser)
    parser.r = r
    parser.gr = gr
    parser.filepath = filepath
    parser.extract_molecules.return_value = mol_list or ["A", "B"]
    return parser


def _make_mock_system_props(
    molecule_count: dict | None = None,
    volume: float = 64.0,
) -> MagicMock:
    """Return a MagicMock mimicking SystemProperties."""
    molecule_count = molecule_count or {"A": 500, "B": 300}
    props = MagicMock()
    props.get.side_effect = lambda name, **kwargs: molecule_count if name == "molecule_count" else volume
    return props


# ===========================================================================
# 1. __init__
# ===========================================================================


class TestInit:
    """Tests for KBIntegrator.__init__."""

    def test_r_stored_as_ndarray(self):
        kbi = _make_kbi()
        assert isinstance(kbi.r, np.ndarray)

    def test_gr_stored_as_ndarray(self):
        kbi = _make_kbi()
        assert isinstance(kbi.gr, np.ndarray)

    def test_n_ref_stored_as_int(self):
        kbi = _make_kbi(n_ref=300)
        assert kbi.n_ref == 300
        assert isinstance(kbi.n_ref, int)

    def test_box_volume_stored_as_float(self):
        kbi = _make_kbi(box_volume=125.0)
        assert kbi.box_volume == pytest.approx(125.0)
        assert isinstance(kbi.box_volume, float)

    def test_delta_stored_as_int(self):
        kbi = _make_kbi(delta=1)
        assert kbi.delta == 1
        assert isinstance(kbi.delta, int)

    def test_weight_type_lowercased(self):
        kbi = _make_kbi(weight_type="Geometric")
        assert kbi.weight_type == "geometric"

    def test_rho_ref_computed_correctly(self):
        kbi = _make_kbi(n_ref=500, box_volume=100.0)
        assert kbi.rho_ref == pytest.approx(5.0)

    def test_errors_stored(self):
        kbi = _make_kbi(errors="warn")
        assert kbi.errors == "warn"

    def test_force_stored(self):
        kbi = _make_kbi(force=True)
        assert kbi.force is True

    def test_list_inputs_converted_to_ndarray(self):
        r = [0.1, 0.2, 0.3]
        gr = [0.0, 0.5, 1.0]
        kbi = KBIntegrator(r=r, gr=gr, n_ref=100, box_volume=10.0, delta=0)
        assert isinstance(kbi.r, np.ndarray)
        assert isinstance(kbi.gr, np.ndarray)


# ===========================================================================
# 2. from_rdf classmethod
# ===========================================================================


class TestFromRdf:
    """Tests for KBIntegrator.from_rdf."""

    def test_creates_instance_from_rdf_parser(self):
        # spec=RdfParser is required so isinstance(mock, RdfParser) passes
        mock_rdf = _make_mock_rdf_parser(mol_list=["A", "B"])
        mock_props = _make_mock_system_props(molecule_count={"A": 500, "B": 300})
        kbi = KBIntegrator.from_rdf(rdf=mock_rdf, system_properties=mock_props)
        assert isinstance(kbi, KBIntegrator)

    def test_n_ref_set_from_second_molecule(self):
        """n_ref should be the count of the second molecule in the RDF pair."""
        mock_rdf = _make_mock_rdf_parser(mol_list=["A", "B"])
        mock_props = _make_mock_system_props(molecule_count={"A": 500, "B": 300})
        kbi = KBIntegrator.from_rdf(rdf=mock_rdf, system_properties=mock_props)
        assert kbi.n_ref == 300

    def test_delta_one_for_self_rdf(self):
        """When both molecules are the same, delta should be 1."""
        mock_rdf = _make_mock_rdf_parser(mol_list=["A", "A"])
        mock_props = _make_mock_system_props(molecule_count={"A": 500})
        kbi = KBIntegrator.from_rdf(rdf=mock_rdf, system_properties=mock_props)
        assert kbi.delta == 1

    def test_delta_zero_for_cross_rdf(self):
        """When molecules differ, delta should be 0."""
        mock_rdf = _make_mock_rdf_parser(mol_list=["A", "B"])
        mock_props = _make_mock_system_props(molecule_count={"A": 500, "B": 300})
        kbi = KBIntegrator.from_rdf(rdf=mock_rdf, system_properties=mock_props)
        assert kbi.delta == 0

    def test_raises_type_error_for_invalid_rdf(self):
        mock_props = _make_mock_system_props()
        with pytest.raises(TypeError, match="not of type 'RdfParser'"):
            KBIntegrator.from_rdf(rdf=42, system_properties=mock_props)

    def test_weight_type_forwarded(self):
        mock_rdf = _make_mock_rdf_parser(mol_list=["A", "B"])
        mock_props = _make_mock_system_props(molecule_count={"A": 500, "B": 300})
        kbi = KBIntegrator.from_rdf(rdf=mock_rdf, system_properties=mock_props, weight_type="u2")
        assert kbi.weight_type == "u2"

    def test_errors_forwarded(self):
        mock_rdf = _make_mock_rdf_parser(mol_list=["A", "B"])
        mock_props = _make_mock_system_props(molecule_count={"A": 500, "B": 300})
        kbi = KBIntegrator.from_rdf(rdf=mock_rdf, system_properties=mock_props, errors="warn")
        assert kbi.errors == "warn"

    def test_force_forwarded(self):
        mock_rdf = _make_mock_rdf_parser(mol_list=["A", "B"])
        mock_props = _make_mock_system_props(molecule_count={"A": 500, "B": 300})
        kbi = KBIntegrator.from_rdf(rdf=mock_rdf, system_properties=mock_props, force=True)
        assert kbi.force is True


# ===========================================================================
# 3. gr_vdv (cached_property)
# ===========================================================================


class TestGrVdv:
    """Tests for KBIntegrator.gr_vdv."""

    def test_returns_ndarray(self):
        kbi = _make_kbi()
        assert isinstance(kbi.gr_vdv, np.ndarray)

    def test_same_shape_as_gr(self):
        kbi = _make_kbi(n=150)
        assert kbi.gr_vdv.shape == kbi.gr.shape

    def test_non_negative(self):
        kbi = _make_kbi()
        assert np.all(kbi.gr_vdv >= 0)

    def test_converges_to_gr_at_large_box(self):
        """With a huge box, vdV correction → 1 so gr_vdv ≈ gr."""
        r, gr = _make_rdf(n=200, r_max=4.0)
        kbi = KBIntegrator(r=r, gr=gr, n_ref=500, box_volume=1e9, delta=0)
        np.testing.assert_allclose(kbi.gr_vdv, gr, rtol=1e-4)

    def test_delta_one_shifts_correction(self):
        """Self-RDF (delta=1) should produce a different correction than cross-RDF."""
        kbi_cross = _make_kbi(delta=0)
        kbi_self = _make_kbi(delta=1)
        assert not np.allclose(kbi_cross.gr_vdv, kbi_self.gr_vdv)

    def test_is_cached(self):
        kbi = _make_kbi()
        first = kbi.gr_vdv
        second = kbi.gr_vdv
        assert first is second


# ===========================================================================
# 4. compute_hr / hr
# ===========================================================================


class TestComputeHr:
    """Tests for KBIntegrator.compute_hr and hr property."""

    def test_hr_none_uses_raw_gr(self):
        kbi = _make_kbi(weight_type="none")
        expected = kbi.gr - 1
        np.testing.assert_array_equal(kbi.compute_hr("none"), expected)

    def test_hr_geometric_uses_gr_vdv(self):
        kbi = _make_kbi(weight_type="geometric")
        expected = kbi.gr_vdv - 1
        np.testing.assert_array_equal(kbi.compute_hr("geometric"), expected)

    def test_hr_u0_uses_gr_vdv(self):
        kbi = _make_kbi(weight_type="u0")
        expected = kbi.gr_vdv - 1
        np.testing.assert_array_equal(kbi.compute_hr("u0"), expected)

    def test_hr_property_matches_weight_type(self):
        for wt in ("none", "u0", "u1", "u2", "geometric"):
            kbi = _make_kbi(weight_type=wt)
            np.testing.assert_array_equal(kbi.hr, kbi.compute_hr(wt))

    def test_hr_shape_matches_r(self):
        kbi = _make_kbi(n=100)
        assert kbi.hr.shape == kbi.r.shape


# ===========================================================================
# 5. Weight functions
# ===========================================================================


class TestWeightFunctions:
    """Tests for geometric_weight, u0_weight, u1_weight, u2_weight."""

    @pytest.fixture
    def kbi(self):
        return _make_kbi(n=100, r_max=3.0)

    def test_u0_weight_is_4pi_r_squared(self, kbi):
        expected = 4 * np.pi * kbi.r**2
        np.testing.assert_allclose(kbi.u0_weight(kbi.r), expected)

    def test_u1_weight_zero_at_rmax(self, kbi):
        """u1(r_max) = 4π r² (1 - 1) = 0."""
        result = kbi.u1_weight(kbi.r)
        assert result[-1] == pytest.approx(0.0, abs=1e-10)

    def test_geometric_weight_zero_at_rmax(self, kbi):
        """w(r_max) = 4π r² (1 - 3/2 + 1/2) = 0."""
        result = kbi.geometric_weight(kbi.r)
        assert result[-1] == pytest.approx(0.0, abs=1e-10)

    def test_u2_weight_positive_for_all_r(self, kbi):
        result = kbi.u2_weight(kbi.r)
        assert np.all(result >= 0)

    def test_u1_weight_less_than_u0(self, kbi):
        """u1 ≤ u0 everywhere since (1 - x³) ≤ 1."""
        u0 = kbi.u0_weight(kbi.r)
        u1 = kbi.u1_weight(kbi.r)
        assert np.all(u1 <= u0 + 1e-12)

    def test_geometric_weight_shape(self, kbi):
        assert kbi.geometric_weight(kbi.r).shape == kbi.r.shape

    def test_weight_mapped_contains_all_keys(self, kbi):
        expected_keys = {"none", "u0", "u1", "u2", "geometric"}
        assert set(kbi._weight_mapped.keys()) == expected_keys

    def test_weight_mapped_none_equals_u0(self, kbi):
        """'none' and 'u0' should use the same weight function object."""
        np.testing.assert_array_equal(
            kbi._weight_mapped["none"](kbi.r),
            kbi._weight_mapped["u0"](kbi.r),
        )

    def test_geometric_weight_formula(self, kbi):
        """Verify the exact formula: 4π r² (1 - 3/2 x + 1/2 x³)."""
        r = kbi.r
        x = r / r.max()
        expected = 4 * np.pi * r**2 * (1 - 1.5 * x + 0.5 * x**3)
        np.testing.assert_allclose(kbi.geometric_weight(r), expected)

    def test_u1_weight_formula(self, kbi):
        """Verify the exact formula: 4π r² (1 - x³)."""
        r = kbi.r
        x = r / r.max()
        expected = 4 * np.pi * r**2 * (1 - x**3)
        np.testing.assert_allclose(kbi.u1_weight(r), expected)


# ===========================================================================
# 6. compute_running_kbi / running_kbi_map / rkbi
# ===========================================================================


class TestRunningKbi:
    """Tests for compute_running_kbi, running_kbi_map, and rkbi."""

    @pytest.fixture
    def kbi(self):
        return _make_kbi(n=100, r_max=3.0)

    def test_running_kbi_shape_matches_r(self, kbi):
        result = kbi.compute_running_kbi("geometric")
        assert result.shape == kbi.r.shape

    def test_running_kbi_starts_at_zero(self, kbi):
        result = kbi.compute_running_kbi("u2")
        assert result[0] == pytest.approx(0.0, abs=1e-10)

    def test_running_kbi_map_has_all_weight_types(self, kbi):
        expected = {"none", "u0", "u1", "u2", "geometric"}
        assert set(kbi.running_kbi_map.keys()) == expected

    def test_running_kbi_map_values_are_ndarrays(self, kbi):
        for key, val in kbi.running_kbi_map.items():
            assert isinstance(val, np.ndarray), f"Expected ndarray for key '{key}'"

    def test_rkbi_matches_weight_type(self, kbi):
        np.testing.assert_array_equal(kbi.rkbi, kbi.running_kbi_map[kbi.weight_type])

    def test_running_kbi_map_is_cached(self, kbi):
        first = kbi.running_kbi_map
        second = kbi.running_kbi_map
        assert first is second

    @pytest.mark.parametrize("wt", ["none", "u0", "u1", "u2", "geometric"])
    def test_all_weight_types_produce_finite_values(self, wt):
        kbi = _make_kbi(weight_type=wt, n=80)
        result = kbi.compute_running_kbi(wt)
        assert np.all(np.isfinite(result))

    def test_none_and_u0_differ_due_to_different_hr(self):
        """
        'none' uses raw gr; 'u0' uses gr_vdv — so running KBIs should differ
        unless the vdV correction is negligible (large box).
        Use a small box to ensure the correction is significant.
        """
        r, gr = _make_rdf(n=100, r_max=3.0)
        kbi = KBIntegrator(r=r, gr=gr, n_ref=500, box_volume=10.0, delta=0)
        rkbi_none = kbi.compute_running_kbi("none")
        rkbi_u0 = kbi.compute_running_kbi("u0")
        assert not np.allclose(rkbi_none, rkbi_u0)


# ===========================================================================
# 7. compute_geometric_extrapolation
# ===========================================================================


class TestComputeGeometricExtrapolation:
    @pytest.fixture
    def kbi(self):
        return _make_kbi(
            n=500,
            r_max=6.0,
            box_volume=216.0,
            use_converged_rdf=True,
            errors="ignore",
        )

    def test_returns_dict(self, kbi):
        result = kbi.compute_geometric_extrapolation(maximize_r2=False, errors="ignore")
        assert isinstance(result, dict)

    def test_result_contains_required_keys(self, kbi):
        result = kbi.compute_geometric_extrapolation(maximize_r2=False, errors="ignore")
        for key in ("G", "F", "r2", "p_value", "std_error", "index_list", "r_fit", "r_rkbi_pred"):
            assert key in result, f"Missing key: {key}"

    def test_r2_between_zero_and_one(self, kbi):
        result = kbi.compute_geometric_extrapolation(maximize_r2=False, errors="ignore")
        assert 0.0 <= result["r2"] <= 1.0

    def test_g_is_float(self, kbi):
        result = kbi.compute_geometric_extrapolation(maximize_r2=False, errors="ignore")
        assert isinstance(result["G"], float)

    def test_store_false_does_not_set_result(self, kbi):
        if hasattr(kbi, "_result"):
            del kbi._result
        kbi.compute_geometric_extrapolation(maximize_r2=False, errors="ignore", store=False)
        assert not hasattr(kbi, "_result")

    def test_store_true_sets_result(self, kbi):
        kbi.compute_geometric_extrapolation(maximize_r2=False, errors="ignore", store=True)
        assert hasattr(kbi, "_result")

    def test_explicit_positions_respected(self, kbi):
        result = kbi.compute_geometric_extrapolation(positions=(2.0, 6.0), maximize_r2=False, errors="ignore")
        assert result["r_fit"].min() >= 2.0 - 1e-10
        assert result["r_fit"].max() <= 6.0 + 1e-10

    def test_raises_value_error_for_empty_range(self, kbi):
        with pytest.raises(ValueError, match="No data points"):
            kbi.compute_geometric_extrapolation(positions=(100.0, 200.0), maximize_r2=False, errors="raise")

    def test_maximize_r2_returns_dict(self, kbi):
        result = kbi.compute_geometric_extrapolation(maximize_r2=True, errors="ignore")
        assert isinstance(result, dict)
        assert "G" in result

    def test_r_rkbi_pred_shape_matches_r_fit(self, kbi):
        result = kbi.compute_geometric_extrapolation(maximize_r2=False, errors="ignore")
        assert result["r_rkbi_pred"].shape == result["r_fit"].shape

    def test_linearity_error_raised_when_r2_below_threshold(self):
        rng = np.random.default_rng(0)
        r = np.linspace(0.1, 4.0, 200)
        gr = 1.0 + rng.uniform(-2.0, 2.0, size=r.size)
        gr = np.clip(gr, 0.0, None)
        kbi = KBIntegrator(r=r, gr=gr, n_ref=500, box_volume=64.0, delta=0)
        with pytest.raises((LinearityError, Exception)):
            kbi.compute_geometric_extrapolation(
                maximize_r2=False,
                r2_threshold=0.9999,
                errors="raise",
            )

    def test_high_r2_for_well_behaved_rdf(self, kbi):
        """Fast-decaying RDF must yield R² > 0.99."""
        result = kbi.compute_geometric_extrapolation(maximize_r2=True, errors="ignore")
        assert result["r2"] > 0.99


class TestGeometricExtrapolationResult:
    @pytest.fixture
    def kbi(self):
        return _make_kbi(
            n=500,
            r_max=6.0,
            box_volume=216.0,
            use_converged_rdf=True,
            errors="raise",
        )

    def test_returns_dict_on_first_access(self, kbi):
        result = kbi.geometric_extrapolation_result
        assert isinstance(result, dict)
        assert "G" in result

    def test_cached_after_first_access(self, kbi):
        r1 = kbi.geometric_extrapolation_result
        r2 = kbi.geometric_extrapolation_result
        assert r1 is r2

    def test_manual_store_overrides_cached(self, kbi):
        kbi._result = {"G": 999.0, "F": 0.0, "r2": 1.0}
        assert kbi.geometric_extrapolation_result["G"] == pytest.approx(999.0)


class TestComputeKbi:
    @pytest.mark.parametrize("wt", ["none", "u0", "u1", "u2"])
    def test_non_geometric_returns_float(self, wt):
        kbi = _make_kbi(weight_type=wt, n=200, r_max=4.0)
        assert isinstance(kbi.compute_kbi(wt), float)

    @pytest.mark.parametrize("wt", ["none", "u0", "u1", "u2"])
    def test_non_geometric_returns_finite_value(self, wt):
        kbi = _make_kbi(weight_type=wt, n=200, r_max=4.0)
        assert np.isfinite(kbi.compute_kbi(wt))

    def test_geometric_returns_float(self):
        kbi = _make_kbi(
            weight_type="geometric",
            n=500,
            r_max=6.0,
            box_volume=216.0,
            use_converged_rdf=True,
        )
        assert isinstance(kbi.compute_kbi("geometric"), float)

    def test_kbi_property_matches_weight_type(self):
        for wt in ("u0", "u1", "u2"):
            kbi = _make_kbi(weight_type=wt, n=200, r_max=4.0)
            assert kbi.kbi == pytest.approx(kbi.compute_kbi(wt))

    def test_force_fallback_to_u2_on_convergence_error(self):
        kbi = _make_kbi(weight_type="geometric", force=True, errors="ignore")
        with patch.object(
            type(kbi),
            "geometric_extrapolation_result",
            new_callable=PropertyMock,
            side_effect=KBIConvergenceError("test failure"),
        ):
            result = kbi.compute_kbi("geometric")
        assert isinstance(result, float)

    def test_returns_nan_on_convergence_error_without_force(self):
        kbi = _make_kbi(weight_type="geometric", force=False, errors="ignore")
        with patch.object(
            type(kbi),
            "geometric_extrapolation_result",
            new_callable=PropertyMock,
            side_effect=KBIConvergenceError("test failure"),
        ):
            result = kbi.compute_kbi("geometric")
        assert np.isnan(result)

    def test_case_insensitive_weight_type(self):
        kbi = _make_kbi(weight_type="u2", n=200, r_max=4.0)
        assert isinstance(kbi.compute_kbi("U2"), float)

    def test_kbi_flat_gr_is_zero(self):
        r = np.linspace(0.1, 4.0, 200)
        gr = np.ones_like(r)
        kbi = KBIntegrator(r=r, gr=gr, n_ref=500, box_volume=64.0, delta=0)
        for wt in ("none", "u0", "u1", "u2"):
            assert kbi.compute_kbi(wt) == pytest.approx(0.0, abs=1e-6)


class TestPlottingMethods:
    @pytest.fixture(autouse=True)
    def use_agg_backend(self):
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        yield
        plt.close("all")

    @pytest.fixture
    def kbi(self):
        # converged RDF so geometric extrapolation succeeds inside plot methods
        return _make_kbi(
            n=500,
            r_max=6.0,
            box_volume=216.0,
            use_converged_rdf=True,
            errors="ignore",
        )

    def test_add_kbi_adds_line_to_axes(self, kbi):
        import matplotlib.pyplot as plt

        _, ax = plt.subplots()
        kbi.add_kbi(ax, weight_type="u2")
        assert len(ax.lines) == 1

    def test_add_lkbi_adds_line_to_axes(self, kbi):
        import matplotlib.pyplot as plt

        _, ax = plt.subplots()
        kbi.add_lkbi(ax)
        assert len(ax.lines) == 1

    def test_add_lkbi_fit_adds_line_to_axes(self, kbi):
        import matplotlib.pyplot as plt

        _, ax = plt.subplots()
        result = kbi.compute_geometric_extrapolation(maximize_r2=False, errors="ignore")
        kbi.add_lkbi_fit(ax, result=result)
        assert len(ax.lines) == 1

    def test_add_kbi_value_adds_hline(self, kbi):
        import matplotlib.pyplot as plt

        _, ax = plt.subplots()
        kbi.add_kbi_value(ax, weight_type="u2")
        assert len(ax.lines) == 1

    def test_plot_kbi_saves_file(self, kbi, tmp_path):
        filepath = str(tmp_path / "kbi.png")
        kbi.plot_kbi(weight_type="u2", filepath=filepath)
        assert Path(filepath).exists()

    def test_plot_kbi_compare_saves_file(self, kbi, tmp_path):
        filepath = str(tmp_path / "compare.png")
        kbi.plot_kbi_compare(weight_types=["u1", "u2"], filepath=filepath)
        assert Path(filepath).exists()

    def test_plot_kbi_extrapolation_saves_file(self, kbi, tmp_path):
        filepath = str(tmp_path / "extrap.png")
        kbi.plot_kbi_extrapolation(filepath=filepath)
        assert Path(filepath).exists()

    def test_plot_kbi_compare_extrapolation_saves_file(self, kbi, tmp_path):
        filepath = str(tmp_path / "compare_extrap.png")
        kbi.plot_kbi_compare_extrapolation(weight_types=["u1", "u2", "geometric"], filepath=filepath)
        assert Path(filepath).exists()

    def test_get_colors_default_returns_dict(self, kbi):
        colors = kbi._get_colors()
        assert isinstance(colors, dict)
        assert set(colors.keys()) == {"none", "u0", "u1", "u2", "geometric"}

    def test_get_colors_with_cmap_returns_dict(self, kbi):
        colors = kbi._get_colors(cmap="viridis")
        assert isinstance(colors, dict)
        assert len(colors) == 5

    def test_colors_by_weight_has_all_keys(self, kbi):
        assert set(kbi._colors_by_weight.keys()) == {"none", "u0", "u1", "u2", "geometric"}


# ===========================================================================
# 11. Edge cases and numerical correctness
# ===========================================================================


class TestEdgeCases:
    """Edge cases and numerical sanity checks."""

    def test_flat_gr_gives_zero_kbi(self):
        """g(r) = 1 everywhere → h(r) = 0 → KBI = 0."""
        r = np.linspace(0.1, 4.0, 200)
        gr = np.ones_like(r)
        kbi = KBIntegrator(r=r, gr=gr, n_ref=500, box_volume=64.0, delta=0)
        for wt in ("none", "u0", "u1", "u2"):
            result = kbi.compute_kbi(wt)
            assert result == pytest.approx(0.0, abs=1e-6), f"Failed for weight_type={wt}"

    def test_running_kbi_monotone_for_positive_hr_none(self):
        """
        Use weight_type='none' with gr > 1 everywhere so h(r) > 0 and
        the vdV correction is bypassed — guarantees monotone integral.
        """
        r = np.linspace(0.1, 4.0, 200)
        gr = np.full_like(r, 2.0)  # h(r) = 1 > 0 everywhere
        kbi = KBIntegrator(r=r, gr=gr, n_ref=500, box_volume=64.0, delta=0, weight_type="none")
        rkbi = kbi.compute_running_kbi("none")
        diffs = np.diff(rkbi)
        assert np.all(diffs >= -1e-10)

    def test_large_box_volume_reduces_vdv_correction(self):
        """With a very large box, gr_vdv ≈ gr."""
        r, gr = _make_rdf(n=200, r_max=4.0)
        kbi_large = KBIntegrator(r=r, gr=gr, n_ref=500, box_volume=1e9, delta=0)
        np.testing.assert_allclose(kbi_large.gr_vdv, gr, rtol=1e-4)

    def test_small_box_vdv_differs_from_gr(self):
        """With a small box, the vdV correction should be significant."""
        r, gr = _make_rdf(n=200, r_max=4.0)
        kbi_small = KBIntegrator(r=r, gr=gr, n_ref=500, box_volume=10.0, delta=0)
        assert not np.allclose(kbi_small.gr_vdv, gr, rtol=1e-4)

    def test_weight_type_none_ignores_vdv(self):
        """weight_type='none' should use raw gr, not gr_vdv."""
        kbi = _make_kbi(weight_type="none", n=100)
        expected_hr = kbi.gr - 1
        np.testing.assert_array_equal(kbi.hr, expected_hr)

    def test_rho_ref_scales_with_n_ref(self):
        kbi1 = _make_kbi(n_ref=100, box_volume=50.0)
        kbi2 = _make_kbi(n_ref=200, box_volume=50.0)
        assert kbi2.rho_ref == pytest.approx(2 * kbi1.rho_ref)

    def test_all_weight_types_accessible_via_rkbi_property(self):
        for wt in ("none", "u0", "u1", "u2", "geometric"):
            kbi = _make_kbi(weight_type=wt, n=100)
            assert kbi.rkbi is kbi.running_kbi_map[wt]

    def test_kbi_magnitude_is_physically_reasonable(self):
        """For a typical liquid, |G| should be well below 1000 nm³."""
        kbi = _make_kbi(n=200, r_max=4.0)
        result = kbi.compute_kbi("u2")
        assert abs(result) < 1000.0
