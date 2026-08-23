"""Performance statistics for a daily P&L series."""

from __future__ import annotations

import numpy as np
import pandas as pd


def equity_curve(returns: pd.Series) -> pd.Series:
    return (1.0 + returns.fillna(0.0)).cumprod()


def max_drawdown(returns: pd.Series) -> float:
    curve = equity_curve(returns)
    peak = curve.cummax()
    return float((curve / peak - 1.0).min()) if len(curve) else float("nan")


def sharpe(returns: pd.Series, annualization: int = 252) -> float:
    """``mean/sd * sqrt(252)`` - the same statistic as Appendix A's ``calc.sharpe``."""
    clean = returns.dropna()
    if len(clean) < 2:
        return float("nan")
    sd = clean.std(ddof=1)
    if sd == 0 or not np.isfinite(sd):
        return float("nan")
    return float(clean.mean() / sd * np.sqrt(annualization))


def sortino(returns: pd.Series, annualization: int = 252) -> float:
    clean = returns.dropna()
    downside = clean[clean < 0]
    if len(clean) < 2 or downside.std(ddof=1) in (0, np.nan):
        return float("nan")
    sd = downside.std(ddof=1)
    if sd == 0 or not np.isfinite(sd):
        return float("nan")
    return float(clean.mean() / sd * np.sqrt(annualization))


def annualized_return(returns: pd.Series, annualization: int = 252) -> float:
    """Arithmetic annualisation, matching the paper's ``mean(pnl) * 252 / I``."""
    clean = returns.dropna()
    return float(clean.mean() * annualization) if len(clean) else float("nan")


def cagr(returns: pd.Series, annualization: int = 252) -> float:
    clean = returns.dropna()
    if len(clean) == 0:
        return float("nan")
    total = float((1.0 + clean).prod())
    if total <= 0:
        return float("nan")
    return total ** (annualization / len(clean)) - 1.0


def annualized_volatility(returns: pd.Series, annualization: int = 252) -> float:
    clean = returns.dropna()
    return float(clean.std(ddof=1) * np.sqrt(annualization)) if len(clean) > 1 else float("nan")


def hit_rate(returns: pd.Series) -> float:
    clean = returns.dropna()
    return float((clean > 0).mean()) if len(clean) else float("nan")


def summarize(
    returns: pd.Series,
    gross_returns: pd.Series | None = None,
    turnover: pd.Series | None = None,
    annualization: int = 252,
) -> dict[str, float]:
    """Headline statistics for one out-of-sample track record.

    ``returns`` is net of costs; ``gross_returns`` and ``turnover`` are optional
    and add the cost decomposition.
    """
    clean = returns.dropna()
    stats: dict[str, float] = {
        "days": float(len(clean)),
        "total_return": float((1.0 + clean).prod() - 1.0) if len(clean) else float("nan"),
        "cagr": cagr(clean, annualization),
        "ann_return": annualized_return(clean, annualization),
        "ann_volatility": annualized_volatility(clean, annualization),
        "sharpe": sharpe(clean, annualization),
        "sortino": sortino(clean, annualization),
        "max_drawdown": max_drawdown(clean),
        "hit_rate": hit_rate(clean),
    }
    mdd = stats["max_drawdown"]
    stats["calmar"] = stats["cagr"] / abs(mdd) if mdd and np.isfinite(mdd) and mdd < 0 else float("nan")
    if gross_returns is not None:
        gross = gross_returns.dropna()
        stats["gross_sharpe"] = sharpe(gross, annualization)
        stats["gross_ann_return"] = annualized_return(gross, annualization)
        stats["cost_drag_ann"] = stats["gross_ann_return"] - stats["ann_return"]
    if turnover is not None:
        t = turnover.dropna()
        stats["ann_turnover"] = float(t.mean() * annualization) if len(t) else float("nan")
        traded = float(t.sum())
        stats["bps_per_unit_traded"] = (
            float(clean.sum() / traded * 1e4) if traded > 0 else float("nan")
        )
    return stats
