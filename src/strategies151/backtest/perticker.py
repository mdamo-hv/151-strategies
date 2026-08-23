"""Per-ticker study: which strategy works best on each name, on its own.

Running a single-name universe is not free of caveats.  Roughly half the
library is *cross-sectional* - it ranks names against each other or demeans
returns across them - and on a universe of one those constructions collapse:
the demeaned return of a single stock is identically zero, and a "top third"
that is also the "bottom third" nets out.  Others degenerate the other way and
become buy & hold.  Both cases are detected and excluded from the ranking with
the reason recorded, so "best strategy" means best among the strategies that
actually express a view on one name.
"""

from __future__ import annotations

import collections
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from strategies151.backtest import metrics as m
from strategies151.backtest.engine import WalkForwardResult, buy_and_hold, common_folds, walk_forward
from strategies151.config import Config
from strategies151.data.panel import Panel
from strategies151.strategies.registry import implemented_keys, resolve

log = logging.getLogger(__name__)

ACTIVE = "active"
NO_POSITION = "not applicable: takes no position on a one-name universe"
BUY_AND_HOLD = "not applicable: degenerates to buy & hold"


def takes_a_position(strategy_cls, panel: Panel, cfg: Config, factory=None) -> bool:
    """Cheap pre-screen: can any point of the grid hold a position on this panel?

    Sweeping the grid once over the whole history costs roughly one fold's work,
    versus 125 folds for a full walk-forward, so screening out the strategies
    that structurally cannot act on a single name is what makes the per-ticker
    study affordable.  A strategy is only skipped when *every* parameter set is
    flat - some grids mix workable and degenerate settings.
    """
    factory = factory or (lambda **params: strategy_cls(**params))
    train = panel.slice(0, min(cfg.backtest.train_days, len(panel)))
    for params in strategy_cls.grid() or [{}]:
        strategy = factory(**params)
        try:
            strategy.fit(train, context=train)
            weights = strategy.weights(panel)
        except Exception:  # noqa: BLE001 - a failure here is not evidence of flatness
            return True
        if float(weights.abs().sum(axis=1).max()) > 1e-9:
            return True
    return False


def classify(result: WalkForwardResult, ticker_returns: pd.Series) -> str:
    """Decide whether a strategy expresses a real view on a single name."""
    if result.daily.empty:
        return NO_POSITION
    if float(result.daily["gross_exposure"].abs().mean()) < 1e-9:
        return NO_POSITION
    turnover = float(result.stats.get("ann_turnover", 0.0) or 0.0)
    aligned = ticker_returns.reindex(result.daily.index)
    correlation = float(result.net_returns.corr(aligned))
    if turnover < 1.0 and np.isfinite(correlation) and correlation > 0.999:
        return BUY_AND_HOLD
    return ACTIVE


def modal_params(folds: pd.DataFrame) -> dict:
    """The parameter set most often chosen across the training windows.

    Reported per axis rather than as the single most common tuple: with a large
    grid the modal tuple can be a one-off, whereas the per-axis mode says what
    the tuning process actually converged on.
    """
    if folds.empty or "params" not in folds:
        return {}
    parsed = [p if isinstance(p, dict) else json.loads(p) for p in folds["params"]]
    if not parsed:
        return {}
    out = {}
    for key in parsed[0]:
        counts = collections.Counter(str(p.get(key)) for p in parsed)
        value, hits = counts.most_common(1)[0]
        share = hits / len(parsed)
        out[key] = value if share > 0.999 else f"{value} ({share * 100:.0f}% of folds)"
    return out


def _skipped_row(ticker: str, strategy, reason: str) -> dict:
    return {
        "ticker": ticker,
        "section": strategy.section,
        "key": strategy.key,
        "title": strategy.title,
        "style": strategy.style,
        "applicability": reason,
        "sharpe": float("nan"),
        "cagr": float("nan"),
        "max_drawdown": float("nan"),
        "calmar": float("nan"),
        "ann_volatility": float("nan"),
        "ann_turnover": float("nan"),
        "hit_rate": float("nan"),
        "params": {},
    }


@dataclass
class TickerStudy:
    """Everything the chart and the tables need for one ticker."""

    ticker: str
    panel: Panel
    results: list[WalkForwardResult]
    benchmark: WalkForwardResult
    table: pd.DataFrame
    best: WalkForwardResult | None
    best_params: dict = field(default_factory=dict)
    train_days: int = 252
    test_days: int = 21
    cost_bps: float = 5.0

    @property
    def close(self) -> pd.Series:
        return self.panel.close[self.ticker]

    @property
    def oos_index(self) -> pd.DatetimeIndex:
        source = self.best or self.benchmark
        return pd.DatetimeIndex(source.daily.index)

    @property
    def oos_start(self) -> str:
        return str(self.oos_index[0].date()) if len(self.oos_index) else "-"

    @property
    def oos_end(self) -> str:
        return str(self.oos_index[-1].date()) if len(self.oos_index) else "-"

    @property
    def folds(self) -> int:
        source = self.best or self.benchmark
        return int(source.stats.get("folds", 0) or 0)

    @property
    def tested(self) -> int:
        return len(self.table)

    @property
    def applicable(self) -> int:
        return int((self.table["applicability"] == ACTIVE).sum())

    @property
    def ranked(self) -> pd.DataFrame:
        return self.table[self.table["applicability"] == ACTIVE].reset_index(drop=True)

    @property
    def best_stats(self) -> dict:
        return self.best.stats if self.best else {}

    @property
    def benchmark_stats(self) -> dict:
        return self.benchmark.stats

    def best_position(self) -> pd.Series:
        """Signed position held by the winning strategy, one value per test day."""
        if self.best is None or "held" not in self.best.per_ticker:
            return pd.Series(dtype=float)
        return self.best.per_ticker["held"][self.ticker]

    def best_equity(self) -> pd.Series:
        return self.best.equity_curve() if self.best else pd.Series(dtype=float)

    def benchmark_equity(self) -> pd.Series:
        return self.benchmark.equity_curve()


def run_ticker_study(
    ticker: str,
    panel: Panel,
    cfg: Config,
    strategy_keys: Sequence[str] | None = None,
) -> TickerStudy:
    """Backtest every strategy on a one-name universe and rank the applicable ones."""
    single = Panel(**{f: getattr(panel, f)[[ticker]] for f in ("open", "high", "low", "close", "volume")})
    strategies = resolve(list(strategy_keys) if strategy_keys else implemented_keys())
    folds = common_folds(strategies, single, cfg)
    benchmark = buy_and_hold(single, cfg)

    results: list[WalkForwardResult] = []
    rows: list[dict] = []
    for strategy in strategies:
        factory = None
        if hasattr(strategy, "components"):
            component_keys = [c.key for c in strategy.components]

            def factory(**params):  # noqa: ANN003
                from strategies151.strategies.registry import build_alpha_combo

                return build_alpha_combo(component_keys, **params)

        if not takes_a_position(type(strategy), single, cfg, factory=factory):
            log.debug("%s / %s screened out: flat on one name", ticker, strategy.key)
            rows.append(_skipped_row(ticker, strategy, NO_POSITION))
            continue
        try:
            result = walk_forward(strategy, single, cfg, folds=folds)
        except Exception as exc:  # noqa: BLE001 - one failure must not sink the ticker
            log.warning("%s / %s failed: %s", ticker, strategy.key, exc)
            rows.append(_skipped_row(ticker, strategy, f"failed: {exc}"))
            continue
        results.append(result)
        applicability = classify(result, single.returns[ticker])
        rows.append(
            {
                "ticker": ticker,
                "section": result.section,
                "key": result.key,
                "title": result.title,
                "style": result.style,
                "applicability": applicability,
                "sharpe": result.stats.get("sharpe", float("nan")),
                "cagr": result.stats.get("cagr", float("nan")),
                "max_drawdown": result.stats.get("max_drawdown", float("nan")),
                "calmar": result.stats.get("calmar", float("nan")),
                "ann_volatility": result.stats.get("ann_volatility", float("nan")),
                "ann_turnover": result.stats.get("ann_turnover", float("nan")),
                "hit_rate": result.stats.get("hit_rate", float("nan")),
                "params": modal_params(result.folds),
            }
        )

    table = pd.DataFrame(rows)
    if not table.empty:
        table = table.sort_values(
            ["applicability", "sharpe"], ascending=[True, False], na_position="last"
        ).reset_index(drop=True)

    active = table[table["applicability"] == ACTIVE] if not table.empty else table
    best = None
    if not active.empty and np.isfinite(active["sharpe"].iloc[0]):
        best_key = active["key"].iloc[0]
        best = next(r for r in results if r.key == best_key)

    # The benchmark shares the strategies' out-of-sample window so the two
    # equity curves on the chart start on the same day.
    reference = best or (results[0] if results else None)
    benchmark = buy_and_hold(single, cfg, index=reference.daily.index if reference else None)

    return TickerStudy(
        ticker=ticker,
        panel=single,
        results=results,
        benchmark=benchmark,
        table=table,
        best=best,
        best_params=modal_params(best.folds) if best is not None else {},
        train_days=cfg.backtest.train_days,
        test_days=cfg.backtest.test_days,
        cost_bps=cfg.backtest.cost_bps,
    )


def studies_frame(studies: Sequence[TickerStudy]) -> pd.DataFrame:
    """Every (ticker, strategy) pair with its out-of-sample statistics."""
    frames = [s.table for s in studies if not s.table.empty]
    if not frames:
        return pd.DataFrame()
    frame = pd.concat(frames, ignore_index=True)
    frame["params"] = frame["params"].apply(json.dumps, default=str)
    return frame


def winners_frame(studies: Sequence[TickerStudy]) -> pd.DataFrame:
    """One row per ticker: its winning strategy, parameters and margin."""
    rows = []
    for study in studies:
        if study.best is None:
            rows.append({"ticker": study.ticker, "best_strategy": None,
                         "note": "no applicable strategy"})
            continue
        stats, bench = study.best_stats, study.benchmark_stats
        rows.append(
            {
                "ticker": study.ticker,
                "section": study.best.section,
                "best_strategy": study.best.title,
                "key": study.best.key,
                "style": study.best.style,
                "params": json.dumps(study.best_params, default=str),
                "sharpe": stats.get("sharpe"),
                "cagr": stats.get("cagr"),
                "max_drawdown": stats.get("max_drawdown"),
                "calmar": stats.get("calmar"),
                "ann_turnover": stats.get("ann_turnover"),
                "buy_hold_sharpe": bench.get("sharpe"),
                "buy_hold_cagr": bench.get("cagr"),
                "buy_hold_max_drawdown": bench.get("max_drawdown"),
                "sharpe_vs_buy_hold": (stats.get("sharpe") or float("nan"))
                - (bench.get("sharpe") or float("nan")),
                "applicable_strategies": study.applicable,
                "strategies_tested": study.tested,
            }
        )
    return pd.DataFrame(rows)
