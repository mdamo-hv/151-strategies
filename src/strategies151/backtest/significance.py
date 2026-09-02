"""Is the winning strategy real, or the best of many coin flips?

Three separate questions, because they have different answers:

1. **Is this return series distinguishable from zero?**  A Newey-West t-test on
   the mean daily return, which is autocorrelation-robust - daily strategy P&L
   is not i.i.d., and the naive standard error understates the true one.
2. **Does it beat holding the stock?**  The same test on the *difference*
   series.  Beating zero is easy in a bull market; beating buy & hold is the
   question the study actually poses.
3. **Does it survive having been chosen as the best of N?**  This is the one
   that matters here.  Selecting the maximum Sharpe out of 11 candidates
   guarantees a flattering number even when every candidate is worthless, so
   the first two tests are biased by construction once applied to the winner.

For (3) two standard corrections are implemented:

* **White's Reality Check** and **Hansen's SPA**, which bootstrap the whole
  candidate set jointly and ask how often pure chance produces a maximum as
  large as the one observed.  Both use the stationary bootstrap of
  [Politis and Romano, 1994] so serial dependence survives resampling.
* The **Deflated Sharpe Ratio** of [Bailey and Lopez de Prado, 2014], which
  adjusts the observed Sharpe for the number of trials, the dispersion of their
  Sharpes, the sample length, and the return distribution's skew and kurtosis.

References
----------
Lo (2002), "The Statistics of Sharpe Ratios".
White (2000), "A Reality Check for Data Snooping".
Hansen (2005), "A Test for Superior Predictive Ability".
Bailey and Lopez de Prado (2014), "The Deflated Sharpe Ratio".
Politis and Romano (1994), "The Stationary Bootstrap".
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats

EULER_MASCHERONI = 0.5772156649015329
DEFAULT_BLOCK = 10  # expected stationary-bootstrap block length, in trading days
DEFAULT_DRAWS = 5000
DEFAULT_SEED = 20260823


# --------------------------------------------------------------------------- #
# Autocorrelation-robust mean test
# --------------------------------------------------------------------------- #
def newey_west_lags(n: int) -> int:
    """Newey-West automatic bandwidth, ``floor(4 (T/100)^(2/9))``."""
    return int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0))) if n > 0 else 0


def newey_west_se(x: np.ndarray, lags: int | None = None) -> float:
    """Standard error of the sample mean, robust to serial correlation."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < 3:
        return float("nan")
    lags = newey_west_lags(n) if lags is None else lags
    centred = x - x.mean()
    variance = float(centred @ centred) / n
    for lag in range(1, min(lags, n - 1) + 1):
        weight = 1.0 - lag / (lags + 1.0)
        cov = float(centred[lag:] @ centred[:-lag]) / n
        variance += 2.0 * weight * cov
    variance = max(variance, 0.0)
    return float(np.sqrt(variance / n))


@dataclass
class MeanTest:
    mean: float
    se: float
    t_stat: float
    p_value: float  # one-sided, H1: mean > 0
    lags: int
    n: int

    @property
    def annualized_mean(self) -> float:
        return self.mean * 252


def mean_test(returns: pd.Series | np.ndarray, lags: int | None = None) -> MeanTest:
    """One-sided Newey-West t-test of ``H0: mean <= 0`` against ``H1: mean > 0``."""
    x = np.asarray(pd.Series(returns).dropna(), dtype=float)
    n = len(x)
    if n < 3:
        return MeanTest(float("nan"), float("nan"), float("nan"), float("nan"), 0, n)
    used = newey_west_lags(n) if lags is None else lags
    se = newey_west_se(x, used)
    if not np.isfinite(se) or se == 0:
        return MeanTest(float(x.mean()), se, float("nan"), float("nan"), used, n)
    t_stat = float(x.mean() / se)
    p_value = float(stats.t.sf(t_stat, df=n - 1))
    return MeanTest(float(x.mean()), se, t_stat, p_value, used, n)


# --------------------------------------------------------------------------- #
# Sharpe-ratio inference
# --------------------------------------------------------------------------- #
def probabilistic_sharpe_ratio(
    returns: pd.Series | np.ndarray,
    benchmark_sharpe: float = 0.0,
) -> float:
    """``P(true Sharpe > benchmark)``, correcting for skew and fat tails.

    ``benchmark_sharpe`` is per-period, matching the return frequency.
    """
    x = np.asarray(pd.Series(returns).dropna(), dtype=float)
    n = len(x)
    if n < 3 or x.std(ddof=1) == 0:
        return float("nan")
    sharpe = float(x.mean() / x.std(ddof=1))
    skew = float(stats.skew(x, bias=False))
    kurtosis = float(stats.kurtosis(x, fisher=False, bias=False))
    denominator = 1.0 - skew * sharpe + (kurtosis - 1.0) / 4.0 * sharpe**2
    if denominator <= 0:
        return float("nan")
    z = (sharpe - benchmark_sharpe) * np.sqrt(n - 1) / np.sqrt(denominator)
    return float(stats.norm.cdf(z))


def expected_maximum_sharpe(n_trials: int, sharpe_std: float) -> float:
    """Sharpe the best of ``n_trials`` worthless strategies would show by luck.

    The expected maximum of ``N`` draws from a standard normal, scaled by the
    observed dispersion of trial Sharpes - Eq. (5) of Bailey & Lopez de Prado.
    """
    if n_trials < 2 or not np.isfinite(sharpe_std) or sharpe_std <= 0:
        return 0.0
    gamma = EULER_MASCHERONI
    term = (1.0 - gamma) * stats.norm.ppf(1.0 - 1.0 / n_trials)
    term += gamma * stats.norm.ppf(1.0 - 1.0 / (n_trials * np.e))
    return float(sharpe_std * term)


def deflated_sharpe_ratio(
    returns: pd.Series | np.ndarray,
    trial_sharpes: np.ndarray | pd.Series,
) -> tuple[float, float]:
    """Probability the winner's Sharpe survives the selection it came from.

    Returns ``(deflated probability, the luck threshold it was tested against)``,
    both in per-period Sharpe units.
    """
    trials = np.asarray(pd.Series(trial_sharpes).dropna(), dtype=float)
    n_trials = len(trials)
    sharpe_std = float(trials.std(ddof=1)) if n_trials > 1 else 0.0
    threshold = expected_maximum_sharpe(n_trials, sharpe_std)
    return probabilistic_sharpe_ratio(returns, threshold), threshold


# --------------------------------------------------------------------------- #
# Joint tests over the whole candidate set
# --------------------------------------------------------------------------- #
def stationary_bootstrap_indices(
    n: int,
    draws: int,
    block: float = DEFAULT_BLOCK,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """``draws x n`` resampling indices with geometric blocks.

    Blocks preserve the serial dependence an i.i.d. bootstrap would destroy,
    which otherwise makes every p-value here look far too small.
    """
    rng = rng or np.random.default_rng(DEFAULT_SEED)
    p = 1.0 / max(block, 1.0)
    idx = np.empty((draws, n), dtype=np.int64)
    idx[:, 0] = rng.integers(0, n, size=draws)
    restart = rng.random((draws, n)) < p
    fresh = rng.integers(0, n, size=(draws, n))
    for t in range(1, n):
        carried = (idx[:, t - 1] + 1) % n
        idx[:, t] = np.where(restart[:, t], fresh[:, t], carried)
    return idx


def _bootstrap_means(values: np.ndarray, idx: np.ndarray, chunk: int = 256) -> np.ndarray:
    """Bootstrap means for every candidate, ``draws x n_candidates``."""
    draws = idx.shape[0]
    out = np.empty((draws, values.shape[1]), dtype=float)
    for start in range(0, draws, chunk):
        stop = min(start + chunk, draws)
        out[start:stop] = values[idx[start:stop]].mean(axis=1)
    return out


@dataclass
class SuperiorityTest:
    """Result of testing the best of a candidate set against a benchmark."""

    best_name: str
    best_mean: float
    n_candidates: int
    n_obs: int
    reality_check_p: float
    spa_p: float
    block: float
    draws: int
    candidate_means: dict[str, float] = field(default_factory=dict)


def reality_check(
    excess: pd.DataFrame,
    draws: int = DEFAULT_DRAWS,
    block: float = DEFAULT_BLOCK,
    seed: int = DEFAULT_SEED,
) -> SuperiorityTest:
    """White's Reality Check and Hansen's SPA over a set of candidates.

    ``excess`` holds one column per candidate, each the daily return *in excess
    of the benchmark*.  The null is that no candidate beats the benchmark; the
    p-value is the share of bootstrap worlds in which chance alone produces a
    maximum at least as large as the one observed.

    SPA differs from the Reality Check by studentising each candidate and by
    recentring only the clearly-bad ones, which stops a pile of hopeless
    candidates from inflating the p-value of a genuinely good one.
    """
    frame = excess.dropna()
    values = frame.to_numpy(dtype=float)
    n, k = values.shape
    if n < 30 or k == 0:
        return SuperiorityTest("", float("nan"), k, n, float("nan"), float("nan"), block, draws)

    means = values.mean(axis=0)
    best = int(np.argmax(means))
    rng = np.random.default_rng(seed)
    idx = stationary_bootstrap_indices(n, draws, block, rng)
    boot = _bootstrap_means(values, idx)

    # White (2000): max over candidates of sqrt(T) * (bootstrap mean - sample mean).
    observed_rc = np.sqrt(n) * means.max()
    boot_rc = (np.sqrt(n) * (boot - means)).max(axis=1)
    rc_p = float((boot_rc >= observed_rc).mean())

    # Hansen (2005) SPA: studentise, and recentre only candidates far enough
    # below zero to be certainly uninformative.
    omega = np.array([newey_west_se(values[:, j]) * np.sqrt(n) for j in range(k)])
    omega = np.where(np.isfinite(omega) & (omega > 0), omega, np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        studentised = np.sqrt(n) * means / omega
        observed_spa = float(np.nanmax(np.maximum(studentised, 0.0)))
        threshold = -np.sqrt(2.0 * np.log(np.log(n))) if n > 3 else -np.inf
        # Subtract the sample mean from every candidate that is not clearly
        # hopeless - imposing the null on it, as the Reality Check does to all.
        # A candidate far enough below zero keeps its own (very negative) mean,
        # so it can never contribute to the bootstrap maximum and cannot inflate
        # the p-value of a genuinely good one. Recentring the wrong subset
        # leaves the bootstrap centred on the observed statistic and the test
        # loses all power.
        recentre = np.where(studentised >= threshold, means, 0.0)
        # Hansen's statistic is max(0, max_k ...) on BOTH sides. Without the
        # floor on the bootstrap draws, a candidate set in which every entry is
        # decisively worse than the benchmark inverts the test: the recentring
        # leaves those draws deeply negative, none reaches the observed 0, and
        # the p-value collapses to ~0 - reporting "significant" for a set that
        # contains no evidence of superiority at all.
        boot_spa = np.maximum(
            np.nanmax(np.sqrt(n) * (boot - recentre) / omega, axis=1), 0.0
        )
    spa_p = float((boot_spa >= observed_spa).mean())

    return SuperiorityTest(
        best_name=str(frame.columns[best]),
        best_mean=float(means[best]),
        n_candidates=k,
        n_obs=n,
        reality_check_p=rc_p,
        spa_p=spa_p,
        block=block,
        draws=draws,
        candidate_means={str(c): float(m) for c, m in zip(frame.columns, means)},
    )


# --------------------------------------------------------------------------- #
# Assembly
# --------------------------------------------------------------------------- #
def assess(
    ticker: str,
    best_name: str,
    best_returns: pd.Series,
    benchmark_returns: pd.Series,
    candidate_returns: pd.DataFrame,
    annualization: int = 252,
    draws: int = DEFAULT_DRAWS,
    block: float = DEFAULT_BLOCK,
    seed: int = DEFAULT_SEED,
) -> dict:
    """Every test above, for one ticker's winning strategy."""
    aligned = candidate_returns.dropna()
    benchmark = benchmark_returns.reindex(aligned.index)
    best = best_returns.reindex(aligned.index)

    versus_zero = mean_test(best)
    versus_benchmark = mean_test(best - benchmark)
    excess = aligned.sub(benchmark, axis=0).dropna()
    joint = reality_check(excess, draws=draws, block=block, seed=seed)

    per_period_sharpes = np.array([
        aligned[c].mean() / aligned[c].std(ddof=1) if aligned[c].std(ddof=1) > 0 else np.nan
        for c in aligned.columns
    ])
    dsr, luck_threshold = deflated_sharpe_ratio(best, per_period_sharpes)
    bench_sharpe = float(benchmark.mean() / benchmark.std(ddof=1))

    return {
        "ticker": ticker,
        "best_strategy": best_name,
        "n_days": versus_zero.n,
        "n_candidates": int(aligned.shape[1]),
        "sharpe": float(best.mean() / best.std(ddof=1) * np.sqrt(annualization)),
        "buy_hold_sharpe": bench_sharpe * np.sqrt(annualization),
        # 1. distinguishable from zero
        "t_stat_vs_zero": versus_zero.t_stat,
        "p_vs_zero": versus_zero.p_value,
        "newey_west_lags": versus_zero.lags,
        # 2. beats buy & hold
        "excess_ann_return": versus_benchmark.annualized_mean,
        "t_stat_vs_buy_hold": versus_benchmark.t_stat,
        "p_vs_buy_hold": versus_benchmark.p_value,
        # The mirror-image test: is the strategy significantly *worse*?
        "p_worse_than_buy_hold": 1.0 - versus_benchmark.p_value,
        # 3. survives having been selected as the best of N
        "reality_check_p": joint.reality_check_p,
        "spa_p": joint.spa_p,
        "psr_vs_zero": probabilistic_sharpe_ratio(best, 0.0),
        "psr_vs_buy_hold": probabilistic_sharpe_ratio(best, bench_sharpe),
        "deflated_sharpe_prob": dsr,
        "luck_threshold_sharpe_ann": luck_threshold * np.sqrt(annualization),
        "bootstrap_draws": draws,
        "bootstrap_block": block,
    }


def verdict(row: dict, alpha: float = 0.05) -> str:
    """Plain-language reading of one assessment row.

    Note the third branch: a one-sided test for *outperformance* returning a
    p-value near 1 does not mean "no difference" - it means the difference runs
    the other way, and that is worth saying rather than glossing as a tie.
    """
    selection_p = row.get("spa_p")
    if selection_p is None or not np.isfinite(selection_p):
        return "inconclusive"
    p_better = row.get("p_vs_buy_hold", 1.0)
    if selection_p < alpha and p_better < alpha:
        return "beats buy & hold after correcting for selection"
    if np.isfinite(p_better) and (1.0 - p_better) < alpha:
        return "lower return than buy & hold (significant)"
    if row.get("p_vs_zero", 1.0) < alpha:
        return "profitable, but no better than buy & hold"
    return "not distinguishable from luck"


def assess_study(
    candidate_returns: pd.DataFrame,
    benchmark_returns: pd.Series,
    labels: dict[str, str] | None = None,
    annualization: int = 252,
    draws: int = DEFAULT_DRAWS,
    block: float = DEFAULT_BLOCK,
    seed: int = DEFAULT_SEED,
) -> tuple[pd.DataFrame, dict]:
    """Test a whole strategy library against a benchmark.

    The per-strategy rows answer "did *this* strategy beat the benchmark",
    ignoring that it sits in a library of many.  The joint result answers the
    question that matters once you quote the leaderboard's top row: could the
    best of this many candidates have looked this good by chance alone?

    Returns ``(per-strategy frame, joint result)``.
    """
    aligned = candidate_returns.dropna(how="all")
    benchmark = benchmark_returns.reindex(aligned.index)
    excess = aligned.sub(benchmark, axis=0).dropna()

    rows = []
    for name in excess.columns:
        own = mean_test(aligned[name].dropna())
        against = mean_test(excess[name])
        series = aligned[name].dropna()
        rows.append({
            "key": name,
            "title": (labels or {}).get(name, name),
            "ann_return": own.annualized_mean,
            "sharpe": float(series.mean() / series.std(ddof=1) * np.sqrt(annualization))
            if series.std(ddof=1) > 0 else float("nan"),
            "t_stat_vs_zero": own.t_stat,
            "p_vs_zero": own.p_value,
            "excess_ann_return": against.annualized_mean,
            "t_stat_vs_benchmark": against.t_stat,
            "p_vs_benchmark": against.p_value,
            "p_worse_than_benchmark": 1.0 - against.p_value,
        })
    per_strategy = pd.DataFrame(rows).sort_values("excess_ann_return", ascending=False)

    joint = reality_check(excess, draws=draws, block=block, seed=seed)
    best_series = aligned[joint.best_name] if joint.best_name in aligned else pd.Series(dtype=float)
    per_period = np.array([
        aligned[c].mean() / aligned[c].std(ddof=1) if aligned[c].std(ddof=1) > 0 else np.nan
        for c in aligned.columns
    ])
    dsr, threshold = deflated_sharpe_ratio(best_series, per_period)
    bench_sharpe = float(benchmark.mean() / benchmark.std(ddof=1))

    summary = {
        "best_strategy": joint.best_name,
        "best_title": (labels or {}).get(joint.best_name, joint.best_name),
        "n_candidates": joint.n_candidates,
        "n_days": joint.n_obs,
        "best_excess_ann_return": float(excess[joint.best_name].mean() * annualization)
        if joint.best_name in excess else float("nan"),
        "reality_check_p": joint.reality_check_p,
        "spa_p": joint.spa_p,
        "deflated_sharpe_prob": dsr,
        "luck_threshold_sharpe_ann": threshold * np.sqrt(annualization),
        "psr_vs_benchmark": probabilistic_sharpe_ratio(best_series, bench_sharpe),
        "bootstrap_draws": draws,
        "bootstrap_block": block,
    }
    return per_strategy, summary
