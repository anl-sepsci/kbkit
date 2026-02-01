"""
Unit tests for ThermoPlotter - targeting >95 % branch + line coverage.

Mocking strategy
----------------
* ``matplotlib.pyplot`` - patched via ``plt.subplots``, ``plt.show``,
  ``plt.close``, ``plt.get_cmap`` so nothing renders and no display is needed.
* ``KBThermo`` - a lightweight ``MagicMock`` exposing ``.systems`` (with
  ``.molecules``, ``.x``, ``.get_mol_index``, ``.n_i``) and every property
  method the plotter calls (``kbi``, ``lngamma``, ``dlngamma_dxi``, …).
* ``format_unit_str`` - patched to a simple pass-through so ylabel
  assertions are predictable.
* ``load_mplstyle`` - patched to a no-op so the style file isn't required.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, PropertyMock, call, patch

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Patch targets - must match the *import* location inside the module under test
# ---------------------------------------------------------------------------
MOD = "kbkit.visualization.thermo"
_PATCH_PLT            = f"{MOD}.plt"
_PATCH_FORMAT_UNIT    = f"{MOD}.format_unit_str"
_PATCH_LOAD_STYLE     = f"{MOD}.load_mplstyle"


# ===========================================================================
# Helpers
# ===========================================================================

N_SYS = 5       # number of simulated compositions
N_MOL = 3       # number of molecules (ternary)


def _make_thermo(
    molecules: list[str] | None = None,
    n_i: int | None = None,
    x: np.ndarray | None = None,
    activity_integration_type: str = "polynomial",
):
    """
    Build a mock KBThermo with a nested mock ``.systems``.

    Every thermodynamic property is exposed as a callable that accepts an
    optional *units* argument and returns an ndarray of the right shape.
    """
    if molecules is None:
        molecules = ["A", "B", "C"]
    n_mol = len(molecules)
    if n_i is None:
        n_i = n_mol
    if x is None:
        # shape (N_SYS, n_mol) - each row sums to 1
        x = np.ones((N_SYS, n_mol)) / n_mol

    systems = MagicMock()
    systems.molecules = molecules
    systems.x = x
    systems.n_i = n_i
    systems.get_mol_index = MagicMock(side_effect=lambda m: molecules.index(m))

    thermo = MagicMock()
    thermo.systems = systems
    thermo.activity_integration_type = activity_integration_type

    # -- 1-D properties (N_SYS,) --
    for name in ("h_mix", "s_ex", "g_ex", "g_id", "g_mix", "det_H", "temperature"):
        _attach_property(thermo, name, np.ones(N_SYS))

    # -- 2-D properties (N_SYS, n_mol) --
    for name in ("lngamma", "dlngamma_dxi"):
        _attach_property(thermo, name, np.ones((N_SYS, n_mol)))

    # -- 3-D property  (N_SYS, n_mol, n_mol) --
    _attach_property(thermo, "kbi", np.ones((N_SYS, n_mol, n_mol)))
    _attach_property(thermo, "s0_ij", np.ones((N_SYS, n_mol, n_mol)))

    # activity_metadata for plot_activity_coef_deriv_fits
    thermo.activity_metadata = MagicMock()
    thermo.activity_metadata.by_types = {"derivative": {}}

    return thermo


def _attach_property(thermo, name, array):
    """
    Make ``thermo.<name>`` a callable that accepts an optional *units* arg
    and returns *array*.  Mimics the ``PropertyResult`` call signature.
    """
    def _prop(units=None):
        return array
    setattr(thermo, name, _prop)


def _make_plotter(thermo=None, molecule_map=None):
    """Instantiate ThermoPlotter with all external deps already patched."""
    if thermo is None:
        thermo = _make_thermo()
    # import here so module-level load_mplstyle has already fired
    from kbkit.visualization.thermo import ThermoPlotter
    return ThermoPlotter(thermo=thermo, molecule_map=molecule_map)


# ===========================================================================
# Shared fixture: suppress all matplotlib rendering
# ===========================================================================

@pytest.fixture(autouse=True)
def _patch_matplotlib():
    """
    Patch plt globally for every test.  Returns the mock so individual tests
    can inspect calls if needed.
    """
    with patch(_PATCH_PLT) as mock_plt, \
         patch(_PATCH_FORMAT_UNIT, side_effect=lambda s: s or ""), \
         patch(_PATCH_LOAD_STYLE):

        # plt.subplots → (fig, ax)  where ax supports all chained calls
        fig = MagicMock()
        ax = MagicMock()
        mock_plt.subplots.return_value = (fig, ax)
        mock_plt.get_cmap.return_value = MagicMock(
            __call__=MagicMock(return_value=np.zeros((5, 4)))  # dummy RGBA array
        )
        # Make get_cmap()(linspace(...)) return a 2-D array with enough rows
        cmap_callable = MagicMock()
        cmap_callable.return_value = np.random.rand(20, 4)  # up to 20 colors
        mock_plt.get_cmap.return_value = cmap_callable

        yield mock_plt


# ===========================================================================
# 1. __init__
# ===========================================================================

class TestInit:
    def test_molecule_map_provided(self):
        thermo = _make_thermo(molecules=["Water", "Ethanol"])
        mp = {"Water": "W", "Ethanol": "E"}
        p = _make_plotter(thermo, molecule_map=mp)
        assert p.molecules == ["W", "E"]
        assert p.molecule_map == mp

    def test_molecule_map_defaults_to_identity(self):
        thermo = _make_thermo(molecules=["Water", "Ethanol"])
        p = _make_plotter(thermo, molecule_map=None)
        assert p.molecules == ["Water", "Ethanol"]
        assert p.molecule_map == {"Water": "Water", "Ethanol": "Ethanol"}


# ===========================================================================
# 2. plot()  -  every y.ndim branch + x flattening + save/show paths
# ===========================================================================

class TestPlot:
    # --- 1-D y, 1-D x -------------------------------------------------------
    def test_1d_y_1d_x(self):
        p = _make_plotter()
        x = np.linspace(0, 1, N_SYS)
        y = np.ones(N_SYS)
        fig, ax = p.plot(x, y, show=False)
        ax.plot.assert_called()

    # --- 1-D y, 2-D x  (x[:,0] extracted) -----------------------------------
    def test_1d_y_2d_x_flattens(self):
        p = _make_plotter()
        x = np.ones((N_SYS, 3))
        x[:, 0] = np.linspace(0, 1, N_SYS)
        y = np.ones(N_SYS)
        fig, ax = p.plot(x, y, show=False)
        # The first positional arg to ax.plot should be the flattened column
        plotted_x = ax.plot.call_args[0][0]
        np.testing.assert_array_equal(plotted_x, x[:, 0])

    # --- 2-D y ---------------------------------------------------------------
    def test_2d_y(self):
        p = _make_plotter(molecule_map={"A": "A", "B": "B", "C": "C"})
        x = np.linspace(0, 1, N_SYS)
        y = np.ones((N_SYS, 3))
        fig, ax = p.plot(x, y, show=False)
        # label kwarg should be the molecule list
        _, kwargs = ax.plot.call_args
        assert kwargs["label"] == p.molecules

    # --- 3-D y, 1-D x -------------------------------------------------------
    def test_3d_y_1d_x(self):
        p = _make_plotter()
        x = np.linspace(0, 1, N_SYS)
        y = np.ones((N_SYS, 3, 3))
        fig, ax = p.plot(x, y, show=False)
        # combinations_with_replacement(range(3),2) → 6 pairs → 6 plot calls
        assert ax.plot.call_count == 6

    # --- 3-D y, 2-D x  (x[:,0] extracted) -----------------------------------
    def test_3d_y_2d_x_flattens(self):
        p = _make_plotter()
        x = np.ones((N_SYS, 3))
        x[:, 0] = np.arange(N_SYS, dtype=float)
        y = np.ones((N_SYS, 2, 2))
        fig, ax = p.plot(x, y, show=False)
        # Every plot call's first arg should be x[:,0]
        for c in ax.plot.call_args_list:
            np.testing.assert_array_equal(c[0][0], x[:, 0])

    # --- xlabel / ylabel applied ---------------------------------------------
    def test_labels_applied(self):
        p = _make_plotter()
        x = np.linspace(0, 1, N_SYS)
        y = np.ones(N_SYS)
        fig, ax = p.plot(x, y, xlabel="X", ylabel="Y", show=False)
        ax.set_xlabel.assert_called_with("X")
        ax.set_ylabel.assert_called_with("Y")

    # --- labels NOT applied when None ----------------------------------------
    def test_labels_not_applied_when_none(self):
        p = _make_plotter()
        x = np.linspace(0, 1, N_SYS)
        y = np.ones(N_SYS)
        fig, ax = p.plot(x, y, xlabel=None, ylabel=None, show=False)
        ax.set_xlabel.assert_not_called()
        ax.set_ylabel.assert_not_called()

    # --- xlim / ylim ---------------------------------------------------------
    def test_xlim_ylim_applied(self):
        p = _make_plotter()
        x = np.linspace(0, 1, N_SYS)
        y = np.ones(N_SYS)
        fig, ax = p.plot(x, y, xlim=(0, 1), ylim=(-1, 1), show=False)
        ax.set_xlim.assert_called_with((0, 1))
        ax.set_ylim.assert_called_with((-1, 1))

    def test_xlim_ylim_not_applied_when_none(self):
        p = _make_plotter()
        x = np.linspace(0, 1, N_SYS)
        y = np.ones(N_SYS)
        fig, ax = p.plot(x, y, xlim=None, ylim=None, show=False)
        ax.set_xlim.assert_not_called()
        ax.set_ylim.assert_not_called()

    # --- savepath with explicit suffix ---------------------------------------
    def test_savepath_with_suffix(self, tmp_path):
        p = _make_plotter()
        x = np.linspace(0, 1, N_SYS)
        y = np.ones(N_SYS)
        sp = str(tmp_path / "out.png")
        fig, ax = p.plot(x, y, savepath=sp, show=False)
        fig.savefig.assert_called_once()
        saved_path = fig.savefig.call_args[0][0]
        assert Path(saved_path).name == "out.png"

    # --- savepath WITHOUT suffix (default filename appended) -----------------
    def test_savepath_without_suffix(self, tmp_path):
        p = _make_plotter()
        x = np.linspace(0, 1, N_SYS)
        y = np.ones(N_SYS)
        sp = str(tmp_path / "mydir")
        fig, ax = p.plot(x, y, savepath=sp, show=False)
        fig.savefig.assert_called_once()
        saved_path = fig.savefig.call_args[0][0]
        assert Path(saved_path).name == "thermo_property.pdf"

    # --- show=True calls plt.show() ------------------------------------------
    @pytest.mark.usefixtures(_patch_matplotlib())
    def test_show_true(self):
        p = _make_plotter()
        x = np.linspace(0, 1, N_SYS)
        y = np.ones(N_SYS)
        p.plot(x, y, show=True)
        _patch_matplotlib.show.assert_called()

    # --- show=False calls plt.close() ----------------------------------------
    @pytest.mark.usefixtures(_patch_matplotlib())
    def test_show_false(self):
        p = _make_plotter()
        x = np.linspace(0, 1, N_SYS)
        y = np.ones(N_SYS)
        p.plot(x, y, show=False)
        _patch_matplotlib.close.assert_called()


# ===========================================================================
# 3. plot_property()
# ===========================================================================

class TestPlotProperty:
    # --- units callable succeeds (normal path) --------------------------------
    def test_units_callable_succeeds(self):
        thermo = _make_thermo()
        # Override kbi so calling with units works
        def kbi_with_units(units=None):
            return np.ones((N_SYS, 3, 3))
        thermo.kbi = kbi_with_units

        p = _make_plotter(thermo)
        fig, ax = p.plot_property("kbi", units="cm^3/mol", show=False)
        # ylabel should contain the unit string
        ylabel_call = ax.set_ylabel.call_args[0][0]
        assert "cm^3/mol" in ylabel_call

    # --- units callable raises → falls back to no-arg call ---------------------
    def test_units_callable_raises_fallback(self):
        thermo = _make_thermo()

        call_log = []
        def bad_kbi(units=None):
            call_log.append(units)
            if units is not None:
                raise TypeError("no units supported")
            return np.ones((N_SYS, 3, 3))
        thermo.kbi = bad_kbi

        p = _make_plotter(thermo)
        fig, ax = p.plot_property("kbi", units="cm^3/mol", show=False)
        # Should have been called twice: once with units (raised), once without
        assert call_log == ["cm^3/mol", None]

    # --- 2-D values + xmol provided ------------------------------------------
    def test_2d_values_xmol_provided(self):
        thermo = _make_thermo(molecules=["A", "B", "C"])
        p = _make_plotter(thermo)
        fig, ax = p.plot_property("lngamma", xmol="B", show=False)
        thermo.systems.get_mol_index.assert_any_call("B")

    # --- 2-D values + xmol is None  (x stays 2-D, no narrowing) ---------------
    def test_2d_values_xmol_none(self):
        thermo = _make_thermo(molecules=["A", "B", "C"])
        p = _make_plotter(thermo)
        # lngamma returns 2-D; xmol=None means x is NOT narrowed
        fig, ax = p.plot_property("lngamma", xmol=None, show=False)
        # xlabel should be generic $x_i$
        xlabel_call = ax.set_xlabel.call_args[0][0]
        assert "x_i" in xlabel_call

    # --- non-2-D values + xmol provided ----------------------------------------
    def test_non2d_values_xmol_provided(self):
        thermo = _make_thermo(molecules=["A", "B", "C"])
        p = _make_plotter(thermo)
        fig, ax = p.plot_property("h_mix", xmol="C", show=False)
        thermo.systems.get_mol_index.assert_any_call("C")

    # --- non-2-D values + xmol is None  (defaults to molecules[0]) ------------
    def test_non2d_values_xmol_none_defaults(self):
        thermo = _make_thermo(molecules=["A", "B", "C"])
        p = _make_plotter(thermo)
        fig, ax = p.plot_property("h_mix", xmol=None, show=False)
        # Should have called get_mol_index with molecules[0] == "A"
        thermo.systems.get_mol_index.assert_any_call("A")

    # --- units=None → ylabel has no unit parenthetical ------------------------
    def test_units_none_ylabel(self):
        thermo = _make_thermo()
        p = _make_plotter(thermo)
        fig, ax = p.plot_property("h_mix", units=None, ylabel="H_mix", show=False)
        ylabel_call = ax.set_ylabel.call_args[0][0]
        # Should be plain ylabel without parentheses wrapping units
        assert ylabel_call == "H_mix"

    # --- savepath without suffix: plot_property computes a local fpath but
    #     never reassigns it back to savepath, so plot() appends its default.
    def test_savepath_no_suffix(self, tmp_path):
        thermo = _make_thermo()
        p = _make_plotter(thermo)
        sp = str(tmp_path / "figures")
        fig, ax = p.plot_property("h_mix", savepath=sp, show=False)
        fig.savefig.assert_called_once()
        saved = Path(fig.savefig.call_args[0][0])
        assert saved.name == "thermo_property.pdf"

    # --- savepath with suffix keeps it as-is ----------------------------------
    def test_savepath_with_suffix(self, tmp_path):
        thermo = _make_thermo()
        p = _make_plotter(thermo)
        sp = str(tmp_path / "custom.png")
        fig, ax = p.plot_property("h_mix", savepath=sp, show=False)
        saved = Path(fig.savefig.call_args[0][0])
        assert saved.name == "custom.png"


# ===========================================================================
# 4. plot_ternary()
# ===========================================================================

class TestPlotTernary:
    def _ternary_x(self):
        """Valid 3-column mole-fraction array (rows sum to 1)."""
        return np.column_stack([
            np.linspace(0.1, 0.8, N_SYS),
            np.linspace(0.1, 0.5, N_SYS),
            np.linspace(0.1, 0.3, N_SYS),
        ])

    # --- x.shape[1] != 3 → ValueError -----------------------------------------
    def test_raises_when_not_ternary(self):
        p = _make_plotter()
        x = np.ones((N_SYS, 2))   # binary, not ternary
        y = np.ones(N_SYS)
        with pytest.raises(ValueError, match="not a ternary system"):
            p.plot_ternary(x, y)

    # --- y.ndim > 1 → ValueError ----------------------------------------------
    def test_raises_when_y_not_1d(self):
        p = _make_plotter()
        x = self._ternary_x()
        y = np.ones((N_SYS, 2))   # 2-D
        with pytest.raises(ValueError, match="only available for 1D"):
            p.plot_ternary(x, y)

    # --- happy path: valid_mask filters out NaN / Inf / negatives -------------
    @pytest.mark.usefixtures(_patch_matplotlib())
    def test_happy_path_filters_invalid(self):
        p = _make_plotter()
        x = self._ternary_x()
        y = np.ones(N_SYS)
        # Inject a NaN and an Inf so the mask actually removes rows
        y[0] = np.nan
        y[1] = np.inf
        x[2, 0] = -0.1   # negative → filtered

        fig, ax = p.plot_ternary(x, y, show=False)
        # tricontourf should have been called (on the ternary ax)
        ax.tricontourf.assert_called_once()
        # The arrays passed in should be shorter than N_SYS
        call_args = ax.tricontourf.call_args[0]
        assert len(call_args[0]) < N_SYS   # a (filtered)
        assert len(call_args[3]) < N_SYS   # values (filtered)

    # --- savepath with suffix -------------------------------------------------
    def test_savepath_with_suffix(self, tmp_path):
        p = _make_plotter()
        x = self._ternary_x()
        y = np.ones(N_SYS)
        sp = str(tmp_path / "tern.pdf")
        fig, ax = p.plot_ternary(x, y, savepath=sp, show=False)
        fig.savefig.assert_called_once()
        assert Path(fig.savefig.call_args[0][0]).name == "tern.pdf"

    # --- savepath without suffix → default name --------------------------------
    def test_savepath_without_suffix(self, tmp_path):
        p = _make_plotter()
        x = self._ternary_x()
        y = np.ones(N_SYS)
        sp = str(tmp_path / "tern_dir")
        fig, ax = p.plot_ternary(x, y, savepath=sp, show=False)
        saved = Path(fig.savefig.call_args[0][0])
        assert saved.name == "ternary_property.pdf"

    # --- show=True path -------------------------------------------------------
    @pytest.mark.usefixtures(_patch_matplotlib())
    def test_show_true(self):
        p = _make_plotter()
        x = self._ternary_x()
        y = np.ones(N_SYS)
        p.plot_ternary(x, y, show=True)
        _patch_matplotlib.show.assert_called()

    # --- cbar_label defaults to "" when None -----------------------------------
    def test_cbar_label_default(self):
        p = _make_plotter()
        x = self._ternary_x()
        y = np.ones(N_SYS)
        fig, ax = p.plot_ternary(x, y, cbar_label=None, show=False)
        # colorbar called with label=""
        fig.colorbar.assert_called_once()
        assert fig.colorbar.call_args[1]["label"] == ""


# ===========================================================================
# 5. plot_property_ternary()
# ===========================================================================

class TestPlotPropertyTernary:
    def _ternary_thermo(self):
        molecules = ["A", "B", "C"]
        x = np.column_stack([
            np.linspace(0.1, 0.8, N_SYS),
            np.linspace(0.1, 0.5, N_SYS),
            np.linspace(0.1, 0.3, N_SYS),
        ])
        return _make_thermo(molecules=molecules, x=x)

    # --- units provided → cbar_label includes formatted units -------------------
    def test_units_provided(self):
        thermo = self._ternary_thermo()
        p = _make_plotter(thermo)
        fig, ax = p.plot_property_ternary("h_mix", units="kJ/mol", show=False)
        # colorbar label should contain the unit string
        label = fig.colorbar.call_args[1]["label"]
        assert "kJ/mol" in label
        # underscores in name replaced with spaces
        assert "h mix" in label

    # --- units is None → cbar_label is just the (space-replaced) name -----------
    def test_units_none(self):
        thermo = self._ternary_thermo()
        p = _make_plotter(thermo)
        fig, ax = p.plot_property_ternary("h_mix", units=None, show=False)
        label = fig.colorbar.call_args[1]["label"]
        assert label == "h mix"

    # --- values(units) raises → fallback -----------------------------------------
    def test_fallback_on_raise(self):
        thermo = self._ternary_thermo()
        call_log = []

        def h_mix_bad(units=None):
            call_log.append(units)
            if units is not None:
                raise TypeError("nope")
            return np.ones(N_SYS)

        thermo.h_mix = h_mix_bad
        p = _make_plotter(thermo)
        fig, ax = p.plot_property_ternary("h_mix", units="kJ/mol", show=False)
        assert call_log == ["kJ/mol", None]


# ===========================================================================
# 6. plot_activity_coef_deriv_fits()
# ===========================================================================

class TestPlotActivityCoefDerivFits:
    def _thermo_with_fits(self, has_fn: bool):
        thermo = _make_thermo(molecules=["A", "B"])
        meta = MagicMock()
        meta.has_fn = has_fn
        meta.x_eval = np.linspace(0, 1, 10)
        meta.y_eval = np.ones(10)
        thermo.activity_metadata.by_types = {"derivative": {"pair_AB": meta}}
        # Override dlngamma_dxi to be callable without units
        thermo.dlngamma_dxi = lambda: np.ones((N_SYS, 2))
        return thermo

    # --- meta.has_fn == True → fit line plotted--------------------------------
    def test_fit_line_plotted_when_has_fn(self):
        thermo = self._thermo_with_fits(has_fn=True)
        p = _make_plotter(thermo)
        fig, ax = p.plot_activity_coef_deriv_fits(show=False)
        # ax.plot called for data + 1 fit line = at least 2 calls
        assert ax.plot.call_count >= 2
        # The fit-line call uses c="k"
        fit_call = ax.plot.call_args_list[-1]
        assert fit_call[1].get("c") == "k"

    # --- meta.has_fn == False → fit line skipped --------------------------------
    def test_fit_line_skipped_when_no_fn(self):
        thermo = self._thermo_with_fits(has_fn=False)
        p = _make_plotter(thermo)
        fig, ax = p.plot_activity_coef_deriv_fits(show=False)
        # Only the data plot call, no fit line
        assert ax.plot.call_count == 1

    # --- xlim / ylim applied ---------------------------------------------------
    def test_xlim_ylim(self):
        thermo = self._thermo_with_fits(has_fn=False)
        p = _make_plotter(thermo)
        fig, ax = p.plot_activity_coef_deriv_fits(
            xlim=(0, 1), ylim=(-2, 2), show=False
        )
        ax.set_xlim.assert_called_with((0, 1))
        ax.set_ylim.assert_called_with((-2, 2))

    # --- savepath without suffix -----------------------------------------------
    def test_savepath_no_suffix(self, tmp_path):
        thermo = self._thermo_with_fits(has_fn=False)
        p = _make_plotter(thermo)
        sp = str(tmp_path / "fits_dir")
        fig, ax = p.plot_activity_coef_deriv_fits(savepath=sp, show=False)
        saved = Path(fig.savefig.call_args[0][0])
        assert saved.name == "activity_coef_deriv_fits.pdf"

    # --- savepath with suffix --------------------------------------------------
    def test_savepath_with_suffix(self, tmp_path):
        thermo = self._thermo_with_fits(has_fn=False)
        p = _make_plotter(thermo)
        sp = str(tmp_path / "fits.png")
        fig, ax = p.plot_activity_coef_deriv_fits(savepath=sp, show=False)
        saved = Path(fig.savefig.call_args[0][0])
        assert saved.name == "fits.png"


# ===========================================================================
# 7. plot_binary_mixing()
# ===========================================================================

class TestPlotBinaryMixing:
    def _binary_thermo(self):
        molecules = ["A", "B"]
        x = np.column_stack([
            np.linspace(0, 1, N_SYS),
            np.linspace(1, 0, N_SYS),
        ])
        return _make_thermo(molecules=molecules, n_i=2, x=x)

    # --- xmol provided ---------------------------------------------------------
    def test_xmol_provided(self):
        thermo = self._binary_thermo()
        p = _make_plotter(thermo)
        fig, ax = p.plot_binary_mixing(xmol="B", show=False)
        thermo.systems.get_mol_index.assert_any_call("B")
        # 5 lines plotted (h_mix, -TS^EX, G^EX, -TS^id, G_mix)
        assert ax.plot.call_count == 5

    # --- xmol is None → defaults to molecules[0] ------------------------------
    def test_xmol_none_defaults(self):
        thermo = self._binary_thermo()
        p = _make_plotter(thermo)
        fig, ax = p.plot_binary_mixing(xmol=None, show=False)
        thermo.systems.get_mol_index.assert_any_call("A")

    # --- savepath without suffix -----------------------------------------------
    def test_savepath_no_suffix(self, tmp_path):
        thermo = self._binary_thermo()
        p = _make_plotter(thermo)
        sp = str(tmp_path / "mix_dir")
        fig, ax = p.plot_binary_mixing(xmol="A", savepath=sp, show=False)
        saved = Path(fig.savefig.call_args[0][0])
        assert saved.name == "thermodyanmic_mixing_properties.pdf"  # matches typo in source

    # --- savepath with suffix --------------------------------------------------
    def test_savepath_with_suffix(self, tmp_path):
        thermo = self._binary_thermo()
        p = _make_plotter(thermo)
        sp = str(tmp_path / "mix.png")
        fig, ax = p.plot_binary_mixing(xmol="A", savepath=sp, show=False)
        saved = Path(fig.savefig.call_args[0][0])
        assert saved.name == "mix.png"

    # --- xlim / ylim applied ---------------------------------------------------
    def test_xlim_ylim(self):
        thermo = self._binary_thermo()
        p = _make_plotter(thermo)
        fig, ax = p.plot_binary_mixing(xmol="A", xlim=(0, 1), ylim=(-5, 5), show=False)
        ax.set_xlim.assert_called_with((0, 1))
        ax.set_ylim.assert_called_with((-5, 5))


# ===========================================================================
# 8. make_figures()
# ===========================================================================

class TestMakeFigures:
    """
    make_figures dispatches to the other plot methods.  We patch them so we
    can verify which ones are called and with which arguments without
    triggering real matplotlib machinery a second time.
    """

    def _patch_plot_methods(self, plotter):
        """Replace every plot_* method on *plotter* with a MagicMock."""
        for name in (
            "plot_property",
            "plot_property_ternary",
            "plot_activity_coef_deriv_fits",
            "plot_binary_mixing",
        ):
            setattr(plotter, name, MagicMock())
        return plotter

    # --- polynomial integration_type → plot_activity_coef_deriv_fits called -----
    def test_polynomial_activity(self, tmp_path):
        thermo = _make_thermo(molecules=["A", "B"], n_i=2,
                              activity_integration_type="polynomial")
        p = self._patch_plot_methods(_make_plotter(thermo))
        p.make_figures(savepath=str(tmp_path), xmol="A")

        p.plot_activity_coef_deriv_fits.assert_called_once()
        # plot_property should NOT have been called with dlngamma_dxi
        for c in p.plot_property.call_args_list:
            assert c[1]["name"] != "dlngamma_dxi"

    # --- non-polynomial → plot_property("dlngamma_dxi") called -----------------
    def test_non_polynomial_activity(self, tmp_path):
        thermo = _make_thermo(molecules=["A", "B"], n_i=2,
                              activity_integration_type="trapezoid")
        p = self._patch_plot_methods(_make_plotter(thermo))
        p.make_figures(savepath=str(tmp_path), xmol="A")

        p.plot_activity_coef_deriv_fits.assert_not_called()
        # One of the plot_property calls should be for dlngamma_dxi
        names = [c[1]["name"] for c in p.plot_property.call_args_list]
        assert "dlngamma_dxi" in names

    # --- n_i == BINARY → binary-specific plots called -------------------------
    def test_binary_system(self, tmp_path):
        thermo = _make_thermo(molecules=["A", "B"], n_i=2,
                              activity_integration_type="trapezoid")
        p = self._patch_plot_methods(_make_plotter(thermo))
        p.make_figures(savepath=str(tmp_path), xmol="A")

        # binary branch calls plot_binary_mixing
        p.plot_binary_mixing.assert_called_once()
        # det_H plotted via plot_property with units
        det_h_calls = [
            c for c in p.plot_property.call_args_list
            if (c[1]["name"] == "det_H")
        ]
        assert len(det_h_calls) == 1
        assert det_h_calls[0][1].get("units") == "kJ/mol"

    # --- n_i == TERNARY → ternary plots called --------------------------------
    def test_ternary_system(self, tmp_path):
        thermo = _make_thermo(molecules=["A", "B", "C"], n_i=3,
                              activity_integration_type="trapezoid")
        p = self._patch_plot_methods(_make_plotter(thermo))
        p.make_figures(savepath=str(tmp_path))

        # ternary branch calls plot_property_ternary for 5 properties
        assert p.plot_property_ternary.call_count == 5
        ternary_names = [
            c[1]["name"] for c in p.plot_property_ternary.call_args_list
        ]
        for expected in ("h_mix", "s_ex", "g_ex", "g_id", "g_mix"):
            assert expected in ternary_names

        # plot_binary_mixing NOT called
        p.plot_binary_mixing.assert_not_called()

    # --- n_i not in {2,3} → early return, no binary/ternary plots -------------
    def test_unsupported_system_size(self, tmp_path):
        thermo = _make_thermo(molecules=["A", "B", "C", "D"], n_i=4,
                              activity_integration_type="trapezoid")
        p = self._patch_plot_methods(_make_plotter(thermo))
        p.make_figures(savepath=str(tmp_path))

        # Neither binary nor ternary-specific methods should be called
        p.plot_binary_mixing.assert_not_called()
        p.plot_property_ternary.assert_not_called()
        # But the common properties (kbi, lngamma, s0_ij, dlngamma_dxi) ARE plotted
        common_names = [
            c[1]["name"] for c in p.plot_property.call_args_list
        ]
        for expected in ("kbi", "lngamma", "s0_ij"):
            assert expected in common_names
