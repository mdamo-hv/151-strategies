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
def asset_pnl(
    weights: pd.DataFrame,
    returns: pd.DataFrame,
    cost_bps: float = 0.0,
    delay: int = 1,
    prev_weights: pd.Series | None = None,
) -> dict[str, pd.DataFrame]:
    """Per-asset decomposition of a target-weight schedule's P&L.

    The weight on row ``t`` is applied to the asset return realised on row
    ``t + delay``; with the default ``delay=1`` a signal computed from the close
    of day ``t`` is traded into day ``t+1``, so nothing is executed at a price
    that was used to generate it.  Costs are charged per name on that name's
    traded notional ``|w_t - w_{t-1}|``.

    Returns frames keyed ``held`` (the weight actually carried into the day),
    ``gross``, ``cost``, ``net`` and ``turnover``.  Summing any of them across
    columns gives the corresponding portfolio series, so attribution is exact by
    construction rather than an after-the-fact approximation.
    """
    weights = weights.reindex(columns=returns.columns).fillna(0.0)
    held = weights.shift(delay)
    if prev_weights is not None and delay > 0:
        held.loc[held.index[0]] = prev_weights.reindex(returns.columns).fillna(0.0)
    held = held.fillna(0.0)

    gross = held * returns.reindex(index=weights.index)
    previous = held.shift(1)
    if prev_weights is not None:
        previous.iloc[0] = prev_weights.reindex(returns.columns).fillna(0.0)
    previous = previous.fillna(0.0)
    turnover = (held - previous).abs()
    cost = turnover * (cost_bps * 1e-4)
    return {
        "held": held,
        "gross": gross,
        "cost": cost,
        "net": gross - cost,
        "turnover": turnover,
    }


def portfolio_pnl(
    weights: pd.DataFrame,
    returns: pd.DataFrame,
    cost_bps: float = 0.0,
    delay: int = 1,
    prev_weights: pd.Series | None = None,
) -> pd.DataFrame:
    """Portfolio-level P&L: :func:`asset_pnl` summed across the universe."""
    parts = asset_pnl(weights, returns, cost_bps, delay, prev_weights)
    return pd.DataFrame(
        {
            "gross_return": parts["gross"].sum(axis=1),
            "cost": parts["cost"].sum(axis=1),
            "net_return": parts["net"].sum(axis=1),
            "turnover": parts["turnover"].sum(axis=1),
            "gross_exposure": parts["held"].abs().sum(axis=1),
            "net_exposure": parts["held"].sum(axis=1),
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
    #: Per-ticker daily decomposition: ``held``, ``gross``, ``cost``, ``net``,
    #: ``turnover``.  Each frame sums across columns to the matching column of
    #: :attr:`daily`.
    per_ticker: dict[str, pd.DataFrame] = field(default_factory=dict)

    @property
    def net_returns(self) -> pd.Series:
        return self.daily["net_return"]

    def equity_curve(self) -> pd.Series:
        return m.equity_curve(self.net_returns)

    def ticker_contributions(self) -> pd.DataFrame:
        """Daily net P&L contribution of each ticker."""
        return self.per_ticker.get("net", pd.DataFrame(index=self.daily.index))

    def ticker_attribution(self, annualization: int = 252) -> pd.DataFrame:
        """One row per ticker: how much of this strategy's P&L it produced.

        Contributions are arithmetic and therefore additive - the
        ``contribution_ann_%`` column sums to the strategy's annualised
        arithmetic return, so the attribution is exact rather than indicative.
        """
        if not self.per_ticker:
            return pd.DataFrame()
        net = self.per_ticker["net"]
        held = self.per_ticker["held"]
        gross = self.per_ticker["gross"]
        cost = self.per_ticker["cost"]
        turnover = self.per_ticker["turnover"]

        total = float(net.to_numpy().sum())
        rows = []
        for ticker in net.columns:
            contribution = float(net[ticker].sum())
            days_held = int((held[ticker].abs() > 1e-12).sum())
            rows.append(
                {
                    "ticker": ticker,
                    "contribution_ann_%": net[ticker].mean() * annualization * 100,
                    "contribution_total_%": contribution * 100,
                    "share_of_pnl_%": (contribution / total * 100) if total else float("nan"),
                    "gross_contribution_ann_%": gross[ticker].mean() * annualization * 100,
                    "cost_ann_%": cost[ticker].mean() * annualization * 100,
                    "avg_gross_weight": float(held[ticker].abs().mean()),
                    "avg_net_weight": float(held[ticker].mean()),
                    "days_held_%": days_held / len(held) * 100 if len(held) else float("nan"),
                    "long_days_%": float((held[ticker] > 1e-12).mean() * 100),
                    "short_days_%": float((held[ticker] < -1e-12).mean() * 100),
                    "ann_turnover_x": float(turnover[ticker].mean() * annualization),
                }
            )
        frame = pd.DataFrame(rows)
        frame.insert(0, "style", self.style)
        frame.insert(0, "key", self.key)
        frame.insert(0, "title", self.title)
        frame.insert(0, "section", self.section)
        return frame.sort_values("contribution_ann_%", ascending=False).reset_index(drop=True)


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
        finite = np.array([s if np.isfinite(s) else -np.inf for s in scores])
        # Objectives that agree to nine decimals are the same objective; the
        # remaining difference is floating-point noise (summation order alone
        # moves a daily P&L sum by ~1e-17). Left unrounded, argmax lets that
        # noise pick the parameter set whenever a grid contains settings that
        # are inert on the training window - an unbinding leverage cap, a
        # rebalance threshold that never triggers - which made the study
        # irreproducible across refactors. Rounding first makes ties resolve to
        # the earliest grid entry, i.e. the paper's documented default.
        best_id = int(np.argmax(np.round(finite, 9)))
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
    parts = asset_pnl(oos_weights, oos_returns, bt.cost_bps, bt.delay)
    parts = {name: frame.iloc[bt.delay :] for name, frame in parts.items()}
    daily = pd.DataFrame(
        {
            "gross_return": parts["gross"].sum(axis=1),
            "cost": parts["cost"].sum(axis=1),
            "net_return": parts["net"].sum(axis=1),
            "turnover": parts["turnover"].sum(axis=1),
            "gross_exposure": parts["held"].abs().sum(axis=1),
            "net_exposure": parts["held"].sum(axis=1),
        }
    )  # the leading unfunded warm-up day is dropped above

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
        per_ticker=parts,
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
    the same weight-target accounting as every strategy.

    When ``index`` is given the benchmark covers *exactly* those days.  The
    schedule is extended ``delay`` bars earlier so the first day is funded by a
    weight set the session before, then trimmed back - otherwise the benchmark
    would silently start a day after the strategies it is compared against.
    """
    delay = cfg.backtest.delay
    if index is None:
        returns = panel.returns
        target_index = returns.index[delay:]
    else:
        full = panel.returns.index
        start = full.get_indexer([index[0]], method="nearest")[0]
        extended = full[max(0, start - delay) : full.get_indexer([index[-1]], method="nearest")[0] + 1]
        returns = panel.returns.reindex(extended)
        target_index = index
    n = returns.shape[1]
    weights = pd.DataFrame(1.0 / n, index=returns.index, columns=returns.columns)
    parts = asset_pnl(weights, returns, cfg.backtest.cost_bps, delay)
    parts = {name: frame.reindex(target_index) for name, frame in parts.items()}
    daily = portfolio_pnl(weights, returns, cfg.backtest.cost_bps, delay).reindex(target_index)
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
        per_ticker=parts,
    )
