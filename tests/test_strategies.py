from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from strategies151.data.panel import Panel
from strategies151.strategies.base import (
    dollar_neutralize,
    normalize_gross,
    rank_demean,
    top_bottom_weights,
)
from strategies151.strategies.registry import REGISTRY, build, build_alpha_combo, resolve

ALL_STRATEGIES = sorted(REGISTRY)


# ------------------------------------------------------------------ helpers --
def test_normalize_gross_scales_to_unit_gross():
    frame = pd.DataFrame({"A": [2.0, 0.0], "B": [-2.0, 0.0]})
    out = normalize_gross(frame)
    assert out.abs().sum(axis=1).iloc[0] == pytest.approx(1.0)
    # An all-zero row must stay zero rather than blow up.
    assert out.iloc[1].abs().sum() == pytest.approx(0.0)


def test_dollar_neutralize_zeroes_the_net():
    frame = pd.DataFrame({"A": [1.0], "B": [0.5], "C": [0.0]})
    assert dollar_neutralize(frame).sum(axis=1).iloc[0] == pytest.approx(0.0)


def test_rank_demean_sums_to_zero():
    frame = pd.DataFrame({"A": [3.0], "B": [1.0], "C": [2.0]})
    assert rank_demean(frame).sum(axis=1).iloc[0] == pytest.approx(0.0)


def test_top_bottom_weights_is_long_short_and_neutral():
    scores = pd.DataFrame({"A": [4.0], "B": [3.0], "C": [2.0], "D": [1.0]})
    weights = top_bottom_weights(scores, fraction=0.5)
    assert weights.sum(axis=1).iloc[0] == pytest.approx(0.0)
    assert weights.abs().sum(axis=1).iloc[0] == pytest.approx(1.0)
    assert weights.loc[0, "A"] > 0 and weights.loc[0, "D"] < 0


def test_top_bottom_weights_long_only_has_no_shorts():
    scores = pd.DataFrame({"A": [4.0], "B": [3.0], "C": [2.0], "D": [1.0]})
    weights = top_bottom_weights(scores, fraction=0.5, long_only=True)
    assert (weights >= 0).all().all()
    assert weights.abs().sum(axis=1).iloc[0] == pytest.approx(1.0)


# ------------------------------------------------- contract for every strategy --
@pytest.fixture(scope="module")
def fitted_panels(panel: Panel):
    train = panel.slice(0, 900)
    full = panel.slice(0, 1100)
    return train, full


@pytest.mark.parametrize("key", ALL_STRATEGIES)
def test_weights_are_well_formed(key, fitted_panels):
    train, full = fitted_panels
    strategy = build(key)
    strategy.fit(train, context=train)
    weights = strategy.weights(full)

    assert weights.index.equals(full.close.index)
    assert list(weights.columns) == full.tickers
    assert np.isfinite(weights.to_numpy()).all(), "weights must never be NaN or inf"


@pytest.mark.parametrize("key", ALL_STRATEGIES)
def test_gross_exposure_never_exceeds_the_investment_level(key, fitted_panels):
    train, full = fitted_panels
    strategy = build(key)
    strategy.fit(train, context=train)
    gross = strategy.weights(full).abs().sum(axis=1)
    assert (gross <= 1.0 + 1e-9).all()


@pytest.mark.parametrize("key", ALL_STRATEGIES)
def test_long_only_strategies_never_short(key, fitted_panels):
    train, full = fitted_panels
    strategy = build(key)
    if not strategy.long_only:
        pytest.skip("strategy is long/short by construction")
    strategy.fit(train, context=train)
    assert (strategy.weights(full) >= -1e-12).all().all()


@pytest.mark.parametrize("key", ALL_STRATEGIES)
def test_weights_are_causal(key, panel: Panel):
    """Perturbing the future must not change any weight computed before it.

    This is the guard against look-ahead: every strategy is called twice, once
    on history truncated at ``cut`` and once on a panel whose post-``cut`` rows
    have been scrambled.  The overlapping weights must be identical.
    """
    cut = 900
    strategy = build(key)
    train = panel.slice(0, 700)
    strategy.fit(train, context=train)

    truncated = panel.slice(0, cut)
    baseline = strategy.weights(truncated)

    rng = np.random.default_rng(3)
    scrambled = {}
    for field in ("open", "high", "low", "close", "volume"):
        frame = getattr(panel, field).iloc[:cut + 150].copy()
        noise = rng.uniform(0.5, 1.5, size=(150, frame.shape[1]))
        frame.iloc[cut:] = frame.iloc[cut:].to_numpy() * noise
        scrambled[field] = frame
    perturbed = Panel(**scrambled)

    extended = strategy.weights(perturbed).iloc[:cut]
    pd.testing.assert_frame_equal(baseline, extended, check_exact=False, atol=1e-10)


def test_alpha_combo_produces_a_dispersed_blend(fitted_panels):
    train, full = fitted_panels
    combo = build_alpha_combo()
    combo.fit(train, context=train)
    assert combo.combo_weights is not None
    assert combo.combo_weights.abs().sum() == pytest.approx(1.0)
    # A degenerate (underdetermined) residualisation collapses to identical
    # weights; the capped construction must not.
    assert combo.combo_weights.std() > 1e-6
    weights = combo.weights(full)
    assert np.isfinite(weights.to_numpy()).all()


def test_pairs_trading_selects_a_pair_and_trades_only_it(fitted_panels):
    train, full = fitted_panels
    strategy = build("3.8.pairs_trading")
    strategy.fit(train, context=train)
    assert strategy.pair is not None
    weights = strategy.weights(full)
    traded = weights.columns[(weights != 0).any()]
    assert set(traded) <= set(strategy.pair)


def test_resolve_returns_every_registered_strategy():
    keys = {s.key for s in resolve()}
    assert set(REGISTRY) <= keys
    assert "3.20.alpha_combo" in keys


def test_normalize_gross_does_not_amplify_numerical_residue():
    """A cancelled cross-sectional signal must stay flat, not become a position."""
    dust = pd.DataFrame({"A": [1e-19, 0.0], "B": [-3e-18, 0.0]})
    out = normalize_gross(dust)
    assert (out.abs().to_numpy() == 0).all()


def test_normalize_gross_still_scales_real_signals():
    real = pd.DataFrame({"A": [1e-6], "B": [-3e-6]})
    assert normalize_gross(real).abs().sum(axis=1).iloc[0] == pytest.approx(1.0)


#: These rank or demean names against each other, so on a universe of one their
#: signal cancels exactly and only floating-point residue is left.
CROSS_SECTIONAL_KEYS = [
    "3.1.price_momentum",
    "3.4.low_volatility",
    "3.6.multifactor",
    "3.9.mean_reversion_single_cluster",
    "3.10.mean_reversion_weighted_regression",
    "3.18.stat_arb_optimization",
    "10.3.contrarian",
]


@pytest.mark.parametrize("key", CROSS_SECTIONAL_KEYS)
def test_cross_sectional_strategies_stay_flat_on_one_name(key, panel: Panel):
    """A cancelled signal must produce no position, not an arbitrary one.

    Without a gross-exposure floor the normaliser rescales residue of order
    1e-18 into a full-size long or short whose sign is pure numerical accident.
    """
    single = Panel(**{f: getattr(panel, f)[[panel.tickers[0]]] for f in
                      ("open", "high", "low", "close", "volume")})
    strategy = build(key)
    train = single.slice(0, 700)
    strategy.fit(train, context=train)
    weights = strategy.weights(single)
    assert (weights.abs().to_numpy() == 0).all()


@pytest.mark.parametrize("key", ALL_STRATEGIES)
def test_single_name_weights_are_finite_and_bounded(key, panel: Panel):
    single = Panel(**{f: getattr(panel, f)[[panel.tickers[0]]] for f in
                      ("open", "high", "low", "close", "volume")})
    strategy = build(key)
    train = single.slice(0, 700)
    strategy.fit(train, context=train)
    gross = strategy.weights(single).abs().sum(axis=1)
    assert np.isfinite(gross.to_numpy()).all()
    assert (gross <= 1.0 + 1e-9).all()


def _state_machine_reference(enter_long, exit_long, enter_short, exit_short):
    """The original per-name loop, kept as the oracle for the vectorised version."""
    out = np.zeros(enter_long.shape)
    el, xl = enter_long.to_numpy(), exit_long.to_numpy()
    es, xs = enter_short.to_numpy(), exit_short.to_numpy()
    for col in range(enter_long.shape[1]):
        current = 0.0
        for row in range(enter_long.shape[0]):
            if el[row, col]:
                current = 1.0
            elif es[row, col]:
                current = -1.0
            elif current > 0 and xl[row, col]:
                current = 0.0
            elif current < 0 and xs[row, col]:
                current = 0.0
            out[row, col] = current
    return pd.DataFrame(out, index=enter_long.index, columns=enter_long.columns)


@pytest.mark.parametrize("trial", range(6))
def test_vectorised_state_machine_matches_the_reference_loop(trial):
    """The cross-section advances together; that must not change any decision.

    Entries beat exits on the same bar and a long entry beats a short entry -
    precedence that is easy to get subtly wrong when the per-name loop is
    replaced by masks.
    """
    from strategies151.strategies.technical import _run_state_machine

    rng = np.random.default_rng(trial)
    columns = list("ABCDEFGHI")
    rates = rng.uniform(0.02, 0.35, 4)
    frames = [pd.DataFrame(rng.random((400, len(columns))) < rate, columns=columns)
              for rate in rates]
    pd.testing.assert_frame_equal(_run_state_machine(*frames), _state_machine_reference(*frames))


def test_state_machine_holds_a_position_until_its_own_exit():
    from strategies151.strategies.technical import _run_state_machine

    n = 6
    off = pd.DataFrame(False, index=range(n), columns=["A"])
    enter_long = off.copy()
    enter_long.loc[1, "A"] = True
    exit_long = off.copy()
    exit_long.loc[4, "A"] = True
    state = _run_state_machine(enter_long, exit_long, off, off)["A"].tolist()
    assert state == [0.0, 1.0, 1.0, 1.0, 0.0, 0.0]


def test_state_machine_lets_a_long_entry_win_the_bar():
    from strategies151.strategies.technical import _run_state_machine

    on = pd.DataFrame(True, index=range(2), columns=["A"])
    off = pd.DataFrame(False, index=range(2), columns=["A"])
    assert _run_state_machine(on, off, on, off)["A"].tolist() == [1.0, 1.0]


def test_safe_inverse_handles_a_singular_covariance():
    """T <= N makes the sample covariance singular - footnote 62 of the paper.

    A 252-day training window against a 437-name universe is always in this
    regime, and inverting it anyway produces enormous weights along near-null
    eigenvectors, which in-sample tuning then prefers because they fit the
    training window almost perfectly.
    """
    from strategies151.strategies.optimization import _safe_inverse

    rng = np.random.default_rng(0)
    observations, assets = 60, 100          # fewer observations than assets
    returns = rng.normal(size=(observations, assets))
    cov = np.cov(returns, rowvar=False)
    assert np.linalg.matrix_rank(cov) < assets      # genuinely singular

    inverse = _safe_inverse(cov)
    assert np.isfinite(inverse).all()
    assert np.linalg.cond(inverse) < 1e12


def test_safe_inverse_leaves_a_well_conditioned_matrix_alone():
    from strategies151.strategies.optimization import _safe_inverse

    rng = np.random.default_rng(1)
    cov = np.cov(rng.normal(size=(2000, 8)), rowvar=False)
    pd.testing.assert_frame_equal(
        pd.DataFrame(_safe_inverse(cov)), pd.DataFrame(np.linalg.inv(cov)), atol=1e-8
    )


def test_stat_arb_weights_stay_bounded_on_a_wide_universe(panel: Panel):
    """Gross exposure must stay at 1 even when the covariance is rank-deficient."""
    wide = Panel(**{f: pd.concat([getattr(panel, f)] * 12, axis=1) for f in
                    ("open", "high", "low", "close", "volume")})
    wide = Panel(**{f: getattr(wide, f).set_axis(
        [f"N{i}" for i in range(getattr(wide, f).shape[1])], axis=1) for f in
        ("open", "high", "low", "close", "volume")})
    strategy = build("3.18.stat_arb_optimization")
    train = wide.slice(0, 252)          # 252 observations, 72 assets
    strategy.fit(train, context=train)
    weights = strategy.weights(wide.slice(0, 400))
    assert np.isfinite(weights.to_numpy()).all()
    assert (weights.abs().sum(axis=1) <= 1.0 + 1e-9).all()
