"""Walk-forward engine: tune in-sample on 1 year, trade the next month blind."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from strategies151.config import BacktestConfig, Config
from strategies151.data.panel import Panel
from strategies151.backtest import metrics as m
from strategies151.backtest.windows import Fold, make_folds
from strategies151.strategies.base import Strategy

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# P&L accounting
# --------------------------------------------------------------------------- #
def portfolio_pnl(
    weights: pd.DataFrame,
    returns: pd.DataFrame,
    cost_bps: float = 0.0,
    delay: int = 1,
    prev_weights: pd.Series | None = None,
) -> pd.DataFrame:
    """P&L of a target-weight schedule.

    The weight on row ``t`` is applied to the asset return realised on row
    ``t + delay``; with the default ``delay=1`` a signal computed from the close
    of day ``t`` is traded into day ``t+1``, so nothing is executed at a price
    that was used to generate it.  Costs are charged on the traded notional
    ``sum |w_t - w_{t-1}|``.
    """
    weights = weights.reindex(columns=returns.columns).fillna(0.0)
    aligned = weights.shift(delay)
    if prev_weights is not None and delay > 0:
        first = aligned.index[0]
        aligned.loc[first] = prev_weights.reindex(returns.columns).fillna(0.0)
    aligned = aligned.fillna(0.0)

    gross = (aligned * returns.reindex(index=weights.index)).sum(axis=1)
    previous = aligned.shift(1)
    if prev_weights is not None:
        previous.iloc[0] = prev_weights.reindex(returns.columns).fillna(0.0)
    previous = previous.fillna(0.0)
    turnover = (aligned - previous).abs().sum(axis=1)
    cost = turnover * (cost_bps * 1e-4)
    return pd.DataFrame(
        {
            "gross_return": gross,
            "cost": cost,
            "net_return": gross - cost,
            "turnover": turnover,
            "gross_exposure": aligned.abs().sum(axis=1),
            "net_exposure": aligned.sum(axis=1),
        }
    )


def strategy_returns(
    strategy: Strategy,
    panel: Panel,
    cost_bps: float = 0.0,
    delay: int = 1,
) -> pd.Series:
    """Convenience wrapper used by 3.20 and by the tests."""
    weights = strategy.weights(panel)
    return portfolio_pnl(weights, panel.returns, cost_bps=cost_bps, delay=delay)["net_return"]


def objective_value(pnl: pd.DataFrame, objective: str, annualization: int) -> float:
    net = pnl["net_return"]
    if objective == "sharpe":
        return m.sharpe(net, annualization)
    if objective == "calmar":
        mdd = m.max_drawdown(net)
        return m.cagr(net, annualization) / abs(mdd) if mdd < 0 else float("nan")
    if objective == "mean_return":
        return float(net.mean())
    raise ValueError(f"unknown selection objective: {objective}")


# --------------------------------------------------------------------------- #
# Weight providers
# --------------------------------------------------------------------------- #
class _StatelessProvider:
    """Weights for a strategy whose :meth:`fit` is a no-op.

    Such a strategy is a pure causal function of the price history, so its
    weights over the whole panel can be computed once per parameter set and then
    sliced per fold.  That is what makes sweeping a full grid over ~100 folds
    affordable.
    """

    def __init__(self, strategy_cls, panel: Panel, cfg: BacktestConfig, combos: Sequence[dict]):
        self.panel = panel
        self.cfg = cfg
        self.combos = list(combos)
        self.cache: list[pd.DataFrame] = []
        for params in self.combos:
            strategy = strategy_cls(**params)
            self.cache.append(strategy.weights(panel).reindex(columns=panel.close.columns).fillna(0.0))

    def weights(self, combo_id: int, fold: Fold, part: str) -> pd.DataFrame:
        frame = self.cache[combo_id]
        lo, hi = (fold.train_start, fold.train_end) if part == "train" else (fold.test_start, fold.test_end)
        return frame.iloc[lo:hi]

    def fitted(self, combo_id: int, fold: Fold) -> Strategy | None:
        return None


class _StatefulProvider:
    """Weights for a strategy that estimates something on the training window.

    Everything (covariances, clusters, KNN neighbour pools, the traded pair) is
    fitted on ``fold``'s training slice only, then applied unchanged to the test
    slice.  Weights are evaluated on a context slice that reaches ``warmup`` bars
    back from the training start so rolling indicators are warm.
    """

    def __init__(self, strategy_cls, panel: Panel, cfg: BacktestConfig, combos: Sequence[dict],
                 factory=None):
        self.strategy_cls = strategy_cls
        self.panel = panel
        self.cfg = cfg
        self.combos = list(combos)
        self.factory = factory or (lambda **p: strategy_cls(**p))
        self._cache: dict[int, tuple[pd.DataFrame, Strategy]] = {}
        self._cached_fold: int | None = None

    def _build(self, combo_id: int, fold: Fold) -> tuple[pd.DataFrame, Strategy]:
        if fold.index != self._cached_fold:  # only the current fold is ever needed
            self._cache.clear()
            self._cached_fold = fold.index
        if combo_id in self._cache:
            return self._cache[combo_id]
        strategy = self.factory(**self.combos[combo_id])
        train_panel = self.panel.slice(fold.train_start, fold.train_end)
        context_start = max(0, fold.train_start - strategy.warmup)
        # The fit context stops at the training end: nothing past it may inform
        # an estimate that will be applied to the test window.
        fit_context = self.panel.slice(context_start, fold.train_end)
        strategy.fit(train_panel, context=fit_context)
        context = self.panel.slice(context_start, fold.test_end)
        frame = strategy.weights(context).reindex(columns=self.panel.close.columns).fillna(0.0)
        self._cache[combo_id] = (frame, strategy)
        return frame, strategy

    def weights(self, combo_id: int, fold: Fold, part: str) -> pd.DataFrame:
        frame, _ = self._build(combo_id, fold)
        dates = self.panel.close.index
        lo, hi = (fold.train_start, fold.train_end) if part == "train" else (fold.test_start, fold.test_end)
        return frame.reindex(dates[lo:hi]).fillna(0.0)

    def fitted(self, combo_id: int, fold: Fold) -> Strategy:
        return self._build(combo_id, fold)[1]


def _is_stateful(strategy_cls) -> bool:
    return strategy_cls.fit is not Strategy.fit


# --------------------------------------------------------------------------- #
# Results
# --------------------------------------------------------------------------- #
@dataclass
class WalkForwardResult:
    key: str
    title: str
    section: str
    style: str
    daily: pd.DataFrame  # out-of-sample P&L, one row per test day
    folds: pd.DataFrame  # per-fold diagnostics and chosen parameters
    stats: dict[str, float] = field(default_factory=dict)
    elapsed_s: float = 0.0

    @property
    def net_returns(self) -> pd.Series:
        return self.daily["net_return"]

    def equity_curve(self) -> pd.Series:
        return m.equity_curve(self.net_returns)


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def walk_forward(
    strategy: Strategy,
    panel: Panel,
    cfg: Config,
    folds: Sequence[Fold] | None = None,
) -> WalkForwardResult:
    """Run one strategy through the sliding-window study.

    For every fold: sweep the strategy's parameter grid on the training window,
    keep the best parameter set by the configured objective, then apply it -
    unchanged and unrefitted - to the test window.  The test windows are
    disjoint and consecutive, so concatenating them yields a single continuous
    out-of-sample track record.
    """
    started = time.perf_counter()
    bt, sel = cfg.backtest, cfg.selection
    strategy_cls = type(strategy)
    combos = list(strategy_cls.grid()) or [{}]
    defaults = strategy_cls.defaults()

    if folds is None:
        folds = make_folds(
            len(panel),
            train_days=bt.train_days,
            test_days=bt.test_days,
            step_days=bt.step_days,
            min_train_days=bt.min_train_days,
            warmup=strategy.warmup,
        )
    folds = [f for f in folds if f.train_start >= 0 and f.test_end <= len(panel)]
    if not folds:
        raise ValueError(
            f"{strategy.key}: not enough history for a {bt.train_days}/{bt.test_days} "
            f"walk-forward with a {strategy.warmup}-bar warmup ({len(panel)} bars available)"
        )

    factory = None
    if hasattr(strategy, "components"):  # 3.20 carries its component strategies
        component_keys = [c.key for c in strategy.components]

        def factory(**params):  # noqa: ANN003 - rebuild fresh components per fold
            from strategies151.strategies.registry import build_alpha_combo

            return build_alpha_combo(component_keys, **params)

    stateful = _is_stateful(strategy_cls)
    provider = (
        _StatefulProvider(strategy_cls, panel, bt, combos, factory=factory)
        if stateful
        else _StatelessProvider(strategy_cls, panel, bt, combos)
    )

    returns = panel.returns
    dates = panel.close.index
    chosen_frames: list[pd.DataFrame] = []
    fold_rows: list[dict] = []

    for fold in folds:
        train_returns = returns.iloc[fold.train_start : fold.train_end]
        scores: list[float] = []
        for combo_id in range(len(combos)):
            weights = provider.weights(combo_id, fold, "train")
            pnl = portfolio_pnl(weights, train_returns, bt.cost_bps, bt.delay)
            scores.append(objective_value(pnl, sel.objective, bt.annualization))
        finite = [s if np.isfinite(s) else -np.inf for s in scores]
        best_id = int(np.argmax(finite))
        if finite[best_id] == -np.inf and sel.fallback_to_default:
            best_id = next((i for i, c in enumerate(combos) if c == defaults), 0)

        test_weights = provider.weights(best_id, fold, "test")
        chosen_frames.append(test_weights)
        row = fold.label(dates)
        row.update(
            {
                "in_sample_objective": scores[best_id],
                "params": combos[best_id],
                "grid_size": len(combos),
            }
        )
        fitted = provider.fitted(best_id, fold)
        if fitted is not None:
            described = fitted.describe()
            for extra in ("fitted_pair", "clusters", "combo_weights"):
                if extra in described:
                    row[extra] = described[extra]
        fold_rows.append(row)

    oos_weights = pd.concat(chosen_frames).sort_index()
    oos_weights = oos_weights[~oos_weights.index.duplicated(keep="last")]
    # One extra leading day of returns is needed so the first test-day weight
    # can be traded into the following session under delay=1.
    oos_returns = returns.reindex(oos_weights.index)
    daily = portfolio_pnl(oos_weights, oos_returns, bt.cost_bps, bt.delay)
    daily = daily.iloc[bt.delay :]  # drop the unfunded warm-up day

    stats = m.summarize(
        daily["net_return"],
        gross_returns=daily["gross_return"],
        turnover=daily["turnover"],
        annualization=bt.annualization,
    )
    stats["folds"] = float(len(folds))
    stats["avg_gross_exposure"] = float(daily["gross_exposure"].mean())
    stats["avg_net_exposure"] = float(daily["net_exposure"].mean())

    return WalkForwardResult(
        key=strategy.key,
        title=strategy.title,
        section=strategy.section,
        style=strategy.style,
        daily=daily,
        folds=pd.DataFrame(fold_rows),
        stats=stats,
        elapsed_s=time.perf_counter() - started,
    )


def common_folds(strategies: Iterable[Strategy], panel: Panel, cfg: Config) -> list[Fold]:
    """A single fold schedule shared by every strategy in the study.

    Warmups differ (residual momentum needs three years of betas, IBS needs a
    week), so aligning on the largest warmup is what makes the leaderboard an
    apples-to-apples comparison over one identical out-of-sample period.
    """
    bt = cfg.backtest
    warmup = max((s.warmup for s in strategies), default=0)
    return make_folds(
        len(panel),
        train_days=bt.train_days,
        test_days=bt.test_days,
        step_days=bt.step_days,
        min_train_days=bt.min_train_days,
        warmup=warmup,
    )


def buy_and_hold(panel: Panel, cfg: Config, index: pd.Index | None = None) -> WalkForwardResult:
    """Long-only equal-weight benchmark, evaluated over the same days and under
    the same weight-target accounting as every strategy."""
    returns = panel.returns if index is None else panel.returns.reindex(index)
    n = returns.shape[1]
    weights = pd.DataFrame(1.0 / n, index=returns.index, columns=returns.columns)
    daily = portfolio_pnl(weights, returns, cfg.backtest.cost_bps, cfg.backtest.delay)
    daily = daily.iloc[cfg.backtest.delay :]
    stats = m.summarize(
        daily["net_return"],
        gross_returns=daily["gross_return"],
        turnover=daily["turnover"],
        annualization=cfg.backtest.annualization,
    )
    return WalkForwardResult(
        key="benchmark.equal_weight",
        title="Equal-weighted buy & hold",
        section="-",
        style="benchmark",
        daily=daily,
        folds=pd.DataFrame(),
        stats=stats,
    )
