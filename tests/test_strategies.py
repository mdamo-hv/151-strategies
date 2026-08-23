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
