"""Strategy protocol plus the portfolio-construction primitives from the paper."""

from __future__ import annotations

import itertools
from abc import ABC, abstractmethod
from typing import Any, Iterator, Mapping, Sequence

import numpy as np
import pandas as pd

from strategies151.data.panel import Panel


# --------------------------------------------------------------------------- #
# Weight primitives (Section 3.1, 3.9, 3.18 of the paper)
# --------------------------------------------------------------------------- #
def normalize_gross(weights: pd.Series | pd.DataFrame, level: float = 1.0):
    """Scale so ``sum |w_i| == level``, Eq. (272)/(346). All-zero rows stay zero."""
    if isinstance(weights, pd.Series):
        gross = weights.abs().sum()
        return weights * (level / gross) if gross > 0 else weights
    gross = weights.abs().sum(axis=1)
    scale = pd.Series(np.where(gross > 0, level / gross.replace(0, np.nan), 0.0), index=gross.index)
    return weights.mul(scale, axis=0).fillna(0.0)


def dollar_neutralize(weights: pd.DataFrame) -> pd.DataFrame:
    """Impose ``sum w_i == 0``, Eq. (273)/(357), by cross-sectional demeaning."""
    return weights.sub(weights.mean(axis=1), axis=0)


def demean_cross_section(frame: pd.DataFrame) -> pd.DataFrame:
    """``~R_i = R_i - R``, Eq. (288)-(289)/(294)."""
    return frame.sub(frame.mean(axis=1), axis=0)


def rank_demean(frame: pd.DataFrame) -> pd.DataFrame:
    """Demeaned cross-sectional ranks ``s_Ai``, Eq. (276)."""
    ranks = frame.rank(axis=1)
    return ranks.sub(ranks.mean(axis=1), axis=0)


def top_bottom_weights(
    scores: pd.DataFrame,
    fraction: float = 0.34,
    long_only: bool = False,
    inverse_vol: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Buy the top slice, short the bottom slice, Section 3.1.

    ``fraction`` is the decile/quintile size generalised to a fraction of the
    universe; with the 6-name universe used here a "decile" would select zero
    names, so the fraction is a tunable parameter (default ~1/3 => 2 names).
    """
    n = scores.shape[1]
    k = max(1, int(round(fraction * n)))
    ranks = scores.rank(axis=1, ascending=False, method="first")
    longs = (ranks <= k).astype(float)
    shorts = (ranks > n - k).astype(float) if not long_only else 0.0
    raw = longs - shorts
    raw = raw.where(scores.notna(), 0.0)
    if inverse_vol is not None:
        raw = raw * inverse_vol.reindex_like(raw).fillna(0.0)
    return normalize_gross(raw)


def inverse_vol_frame(returns: pd.DataFrame, window: int, power: int = 1) -> pd.DataFrame:
    """``1/sigma_i`` (or ``1/sigma_i^2``) weights, Eq. (372)-(373)."""
    vol = returns.rolling(window, min_periods=max(5, window // 4)).std()
    inv = 1.0 / vol.pow(power)
    return inv.replace([np.inf, -np.inf], np.nan)


def moving_average(prices: pd.DataFrame, length: int, kind: str = "sma", lam: float = 0.9):
    """SMA Eq. (319) or EMA Eq. (320) with decay ``lambda``."""
    if kind == "sma":
        return prices.rolling(length, min_periods=max(2, length // 2)).mean()
    if kind == "ema":
        alpha = 1.0 - lam
        return prices.ewm(alpha=alpha, min_periods=max(2, length // 2), adjust=True).mean()
    raise ValueError(f"unknown moving-average kind: {kind}")


def positions_from_signal(signal: pd.DataFrame, long_only: bool = False) -> pd.DataFrame:
    """Turn a -1/0/+1 state machine output into normalised portfolio weights."""
    if long_only:
        signal = signal.clip(lower=0.0)
    return normalize_gross(signal.astype(float))


# --------------------------------------------------------------------------- #
# Strategy protocol
# --------------------------------------------------------------------------- #
class Strategy(ABC):
    """A portfolio-weight generator for one strategy in the paper.

    Contract
    --------
    * :meth:`fit` sees the *training* window only and may store estimates
      (covariances, KNN training sets, feature ranges).
    * :meth:`weights` is called with a panel whose tail is the test window and
      whose head is lookback context; it must be strictly causal - the weight on
      row ``t`` may only use data up to and including row ``t``.  The engine
      applies row ``t``'s weights to the return realised on ``t+1``.
    """

    key: str = ""
    section: str = ""
    title: str = ""
    style: str = ""
    long_only: bool = False
    #: In-sample tuning grid; the engine sweeps it on every training window.
    param_grid: Mapping[str, Sequence[Any]] = {}
    #: Bars of history the strategy needs before its first usable weight.
    warmup: int = 60

    def __init__(self, **params: Any):
        self.params: dict[str, Any] = {**self.defaults(), **params}

    @classmethod
    def defaults(cls) -> dict[str, Any]:
        """First entry of each grid axis is the documented paper default."""
        return {name: values[0] for name, values in cls.param_grid.items()}

    @classmethod
    def grid(cls) -> Iterator[dict[str, Any]]:
        if not cls.param_grid:
            yield {}
            return
        names = list(cls.param_grid)
        for combo in itertools.product(*(cls.param_grid[n] for n in names)):
            yield dict(zip(names, combo))

    def fit(self, train: Panel, context: Panel | None = None) -> "Strategy":  # noqa: D401
        """Estimate anything that must be frozen before the test window.

        ``train`` is exactly the training window.  ``context`` additionally
        carries ``warmup`` bars of history in front of it, which composite
        strategies need in order to evaluate their components over the training
        window without those components being cold-started inside it.
        """
        return self

    @abstractmethod
    def weights(self, panel: Panel) -> pd.DataFrame:
        """Target weights, indexed like ``panel``, one column per ticker."""

    # ------------------------------------------------------------- utilities --
    def _empty(self, panel: Panel) -> pd.DataFrame:
        return pd.DataFrame(0.0, index=panel.close.index, columns=panel.close.columns)

    def describe(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "section": self.section,
            "title": self.title,
            "style": self.style,
            "long_only": self.long_only,
            "params": dict(self.params),
        }

    def __repr__(self) -> str:
        args = ", ".join(f"{k}={v!r}" for k, v in self.params.items())
        return f"{type(self).__name__}({args})"
