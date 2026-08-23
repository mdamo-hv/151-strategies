"""Statistical tests, validated on synthetic data where the answer is known.

The point of a significance module is that its p-values mean what they say, so
these tests check *calibration* (how often it rejects when nothing is there) and
*power* (how often it rejects when something is), not just that it runs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from strategies151.backtest.significance import (
    assess,
    deflated_sharpe_ratio,
    expected_maximum_sharpe,
    mean_test,
    newey_west_lags,
    newey_west_se,
    probabilistic_sharpe_ratio,
    reality_check,
    stationary_bootstrap_indices,
    verdict,
)


def _noise(rng, n, k, edge=0.0, ar=0.0, penalise_rest=0.0):
    e = rng.normal(0, 0.01, size=(n, k))
    if ar:
        for t in range(1, n):
            e[t] += ar * e[t - 1]
    if edge:
        e[:, 0] += edge
    if penalise_rest:
        e[:, 1:] -= penalise_rest
    return pd.DataFrame(e, columns=[f"s{i}" for i in range(k)])


# ------------------------------------------------------------- Newey-West --
def test_newey_west_se_exceeds_the_naive_se_under_autocorrelation():
    rng = np.random.default_rng(1)
    x = np.zeros(3000)
    for t in range(1, 3000):
        x[t] = 0.5 * x[t - 1] + rng.normal(0, 0.01)
    naive = x.std(ddof=1) / np.sqrt(len(x))
    assert newey_west_se(x) > naive * 1.2


def test_newey_west_se_matches_the_naive_se_for_white_noise():
    rng = np.random.default_rng(2)
    x = rng.normal(0, 0.01, 5000)
    naive = x.std(ddof=1) / np.sqrt(len(x))
    assert newey_west_se(x) == pytest.approx(naive, rel=0.15)


def test_bandwidth_grows_with_the_sample():
    assert newey_west_lags(100) < newey_west_lags(10_000)


def test_mean_test_detects_a_real_drift():
    rng = np.random.default_rng(3)
    result = mean_test(rng.normal(0.001, 0.01, 4000))
    assert result.p_value < 0.01
    assert result.t_stat > 2


def test_mean_test_is_not_fooled_by_noise():
    rng = np.random.default_rng(4)
    assert mean_test(rng.normal(0.0, 0.01, 4000)).p_value > 0.05


def test_mean_test_handles_a_series_too_short_to_test():
    assert np.isnan(mean_test(pd.Series([0.01, 0.02])).p_value)


# ------------------------------------------------------- stationary bootstrap --
def test_bootstrap_indices_have_the_right_shape_and_range():
    idx = stationary_bootstrap_indices(500, 40, block=10, rng=np.random.default_rng(5))
    assert idx.shape == (40, 500)
    assert idx.min() >= 0 and idx.max() < 500


def test_bootstrap_blocks_preserve_serial_structure():
    """Long blocks must reproduce autocorrelation that an i.i.d. bootstrap destroys."""
    rng = np.random.default_rng(6)
    x = np.zeros(2000)
    for t in range(1, 2000):
        x[t] = 0.7 * x[t - 1] + rng.normal(0, 0.01)

    def lag1(sample):
        return float(np.corrcoef(sample[:-1], sample[1:])[0, 1])

    blocked = stationary_bootstrap_indices(2000, 8, block=50, rng=np.random.default_rng(7))
    iid = stationary_bootstrap_indices(2000, 8, block=1, rng=np.random.default_rng(8))
    assert np.mean([lag1(x[i]) for i in blocked]) > np.mean([lag1(x[i]) for i in iid]) + 0.3


# ------------------------------------------------------ calibration and power --
@pytest.mark.slow
def test_reality_check_rejects_about_five_percent_under_the_null():
    """With no edge anywhere, a 5% test must reject about 5% of the time."""
    rng = np.random.default_rng(9)
    rc, spa = [], []
    for r in range(60):
        result = reality_check(_noise(rng, 800, 8), draws=400, seed=500 + r)
        rc.append(result.reality_check_p)
        spa.append(result.spa_p)
    # 60 replications: the 95% band around a true 5% size is roughly 0-12%.
    assert np.mean(np.array(rc) < 0.05) < 0.20
    assert np.mean(np.array(spa) < 0.05) < 0.20


@pytest.mark.slow
def test_reality_check_finds_a_genuine_edge():
    rng = np.random.default_rng(10)
    rc, spa = [], []
    for r in range(25):
        result = reality_check(_noise(rng, 800, 8, edge=0.0015), draws=400, seed=900 + r)
        rc.append(result.reality_check_p)
        spa.append(result.spa_p)
    assert np.mean(np.array(rc) < 0.05) > 0.7
    assert np.mean(np.array(spa) < 0.05) > 0.7


@pytest.mark.slow
def test_spa_beats_the_reality_check_when_most_candidates_are_hopeless():
    """Hansen's refinement exists for exactly this case; it must show up."""
    rng = np.random.default_rng(11)
    rc, spa = [], []
    for r in range(25):
        result = reality_check(
            _noise(rng, 800, 10, edge=0.0015, penalise_rest=0.002), draws=400, seed=1300 + r
        )
        rc.append(result.reality_check_p)
        spa.append(result.spa_p)
    assert np.mean(np.array(spa) < 0.05) >= np.mean(np.array(rc) < 0.05)


def test_reality_check_reports_the_best_candidate():
    rng = np.random.default_rng(12)
    result = reality_check(_noise(rng, 600, 6, edge=0.003), draws=200, seed=1)
    assert result.best_name == "s0"
    assert result.n_candidates == 6


def test_reality_check_is_deterministic_for_a_given_seed():
    rng = np.random.default_rng(13)
    frame = _noise(rng, 400, 5)
    a = reality_check(frame, draws=200, seed=42)
    b = reality_check(frame, draws=200, seed=42)
    assert a.reality_check_p == b.reality_check_p and a.spa_p == b.spa_p


def test_reality_check_declines_a_sample_too_short_to_bootstrap():
    rng = np.random.default_rng(14)
    assert np.isnan(reality_check(_noise(rng, 10, 3), draws=50).reality_check_p)


# --------------------------------------------------------- Sharpe inference --
def test_probabilistic_sharpe_is_high_for_a_strong_track_record():
    rng = np.random.default_rng(15)
    assert probabilistic_sharpe_ratio(rng.normal(0.001, 0.01, 3000)) > 0.99


def test_probabilistic_sharpe_averages_a_coin_flip_over_noise_draws():
    """Any single noise draw can land anywhere; the average must sit at 0.5."""
    rng = np.random.default_rng(16)
    values = [probabilistic_sharpe_ratio(rng.normal(0.0, 0.01, 1500)) for _ in range(200)]
    assert 0.42 < float(np.mean(values)) < 0.58


def test_luck_threshold_rises_with_the_number_of_trials():
    assert expected_maximum_sharpe(100, 0.03) > expected_maximum_sharpe(10, 0.03) > 0


def test_luck_threshold_is_zero_for_a_single_trial():
    assert expected_maximum_sharpe(1, 0.03) == 0.0


def test_deflation_penalises_a_wider_search():
    rng = np.random.default_rng(17)
    returns = rng.normal(0.0005, 0.01, 2500)
    few, _ = deflated_sharpe_ratio(returns, rng.normal(0.02, 0.03, 3))
    many, _ = deflated_sharpe_ratio(returns, rng.normal(0.02, 0.03, 200))
    assert many < few
    assert many < probabilistic_sharpe_ratio(returns)


# ------------------------------------------------------------------ assembly --
@pytest.fixture(scope="module")
def assessment():
    rng = np.random.default_rng(18)
    candidates = _noise(rng, 1200, 6)
    index = pd.bdate_range("2018-01-01", periods=1200)
    candidates.index = index
    benchmark = pd.Series(rng.normal(0.0004, 0.011, 1200), index=index)
    best = candidates.mean().idxmax()
    return assess("TEST", best, candidates[best], benchmark, candidates, draws=300)


def test_assess_reports_every_layer_of_the_question(assessment):
    for key in ("p_vs_zero", "p_vs_buy_hold", "reality_check_p", "spa_p",
                "deflated_sharpe_prob", "psr_vs_buy_hold", "luck_threshold_sharpe_ann"):
        assert key in assessment


def test_assess_p_values_are_probabilities(assessment):
    for key in ("p_vs_zero", "p_vs_buy_hold", "reality_check_p", "spa_p",
                "deflated_sharpe_prob"):
        assert 0.0 <= assessment[key] <= 1.0


def test_noise_is_not_declared_a_winner(assessment):
    assert assessment["spa_p"] > 0.05
    assert verdict(assessment) == "not distinguishable from luck"


def test_verdict_requires_surviving_the_selection_correction():
    beats = {"spa_p": 0.01, "p_vs_buy_hold": 0.01, "p_vs_zero": 0.001}
    assert verdict(beats) == "beats buy & hold after correcting for selection"
    only_positive = {"spa_p": 0.40, "p_vs_buy_hold": 0.40, "p_vs_zero": 0.001}
    assert verdict(only_positive) == "profitable, but no better than buy & hold"
    nothing = {"spa_p": 0.60, "p_vs_buy_hold": 0.60, "p_vs_zero": 0.60}
    assert verdict(nothing) == "not distinguishable from luck"


def test_verdict_calls_out_significant_underperformance():
    """p near 1 on a one-sided test means the difference runs the other way."""
    worse = {"spa_p": 0.90, "p_vs_buy_hold": 0.99, "p_vs_zero": 0.001}
    assert verdict(worse) == "lower return than buy & hold (significant)"


def test_assess_reports_both_sides_of_the_benchmark_test(assessment):
    assert assessment["p_worse_than_buy_hold"] == pytest.approx(
        1.0 - assessment["p_vs_buy_hold"]
    )
