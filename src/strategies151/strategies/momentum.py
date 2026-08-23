"""Momentum / trend strategies: Sections 3.1, 3.7, 4.1, 4.1.1, 4.1.2, 4.6, 10.4."""

from __future__ import annotations

import numpy as np
import pandas as pd

from strategies151.data.panel import Panel
from strategies151.strategies.base import (
    Strategy,
    demean_cross_section,
    inverse_vol_frame,
    moving_average,
    normalize_gross,
    top_bottom_weights,
)


def _cumulative_return(close: pd.DataFrame, formation: int, skip: int) -> pd.DataFrame:
    """``R_cum_i = P_i(S)/P_i(S+T) - 1``, Eq. (267), with an S-day skip period."""
    return close.shift(skip) / close.shift(skip + formation) - 1.0


class PriceMomentum(Strategy):
    """3.1 Price-momentum - Eq. (266)-(273).

    Buys the top slice and shorts the bottom slice of the universe, ranked by
    one of the three selection criteria the paper offers: cumulative return over
    the formation period, mean return, or risk-adjusted mean return.
    """

    key = "3.1.price_momentum"
    section = "3.1"
    title = "Price-momentum"
    style = "momentum"
    warmup = 300
    param_grid = {
        # 12-month formation, 1-month skip is the paper's "usual" choice.
        "formation": (252, 126, 63),
        "skip": (21, 0),
        "criterion": ("cumulative", "mean", "risk_adjusted"),
        "fraction": (0.34, 0.17),
        "weighting": ("uniform", "inverse_vol"),
    }

    def _scores(self, panel: Panel) -> pd.DataFrame:
        close, formation, skip = panel.close, self.params["formation"], self.params["skip"]
        criterion = self.params["criterion"]
        if criterion == "cumulative":
            return _cumulative_return(close, formation, skip)
        rets = panel.returns.shift(skip)
        mean = rets.rolling(formation, min_periods=formation // 2).mean()  # Eq. (268)
        if criterion == "mean":
            return mean
        vol = rets.rolling(formation, min_periods=formation // 2).std()  # Eq. (270)
        return mean / vol.replace(0.0, np.nan)  # Eq. (269)

    def weights(self, panel: Panel) -> pd.DataFrame:
        inv_vol = (
            inverse_vol_frame(panel.returns, self.params["formation"])
            if self.params["weighting"] == "inverse_vol"
            else None
        )
        return top_bottom_weights(
            self._scores(panel),
            fraction=self.params["fraction"],
            long_only=self.long_only,
            inverse_vol=inv_vol,
        )


class PriceMomentumLongOnly(PriceMomentum):
    """3.1 Price-momentum, long-only variant - Eq. (271)."""

    key = "3.1.price_momentum_long_only"
    title = "Price-momentum (long-only)"
    long_only = True


class ResidualMomentum(Strategy):
    """3.7 Residual momentum - Eq. (278)-(282).

    The paper regresses each stock on the Fama-French factors.  With a 6-name
    universe and no external factor file, the market factor is proxied by the
    equal-weighted universe return and the size/value legs are dropped: what
    survives is the market-residual momentum, which is the dominant leg of the
    original construction.  This substitution is recorded in the catalog.
    """

    key = "3.7.residual_momentum"
    section = "3.7"
    title = "Residual momentum (market-residual proxy)"
    style = "momentum"
    warmup = 800
    param_grid = {
        "beta_window": (756, 504, 252),  # 36 months in the paper
        "formation": (252, 126),
        "skip": (21, 0),
        "fraction": (0.34,),
    }

    def weights(self, panel: Panel) -> pd.DataFrame:
        rets = panel.returns
        market = rets.mean(axis=1)  # MKT proxy, Eq. (278)
        beta_window = self.params["beta_window"]
        min_periods = max(60, beta_window // 3)
        var_m = market.rolling(beta_window, min_periods=min_periods).var()
        betas = {}
        for ticker in rets.columns:
            cov = rets[ticker].rolling(beta_window, min_periods=min_periods).cov(market)
            betas[ticker] = cov / var_m.replace(0.0, np.nan)
        beta = pd.DataFrame(betas)
        # Eq. (279): residuals use the *lagged* betas, so nothing peeks ahead.
        resid = rets - beta.shift(1).mul(market, axis=0)
        formation, skip = self.params["formation"], self.params["skip"]
        resid = resid.shift(skip)
        mean = resid.rolling(formation, min_periods=formation // 2).mean()  # Eq. (280)
        vol = resid.rolling(formation, min_periods=formation // 2).std()  # Eq. (282)
        scores = mean / vol.replace(0.0, np.nan)  # Eq. (281)
        return top_bottom_weights(scores, fraction=self.params["fraction"])


class SectorMomentumRotation(Strategy):
    """4.1 Sector momentum rotation - Eq. (361), applied to the equity universe.

    Structurally identical to 3.1 with cumulative-return ranking; the paper
    distinguishes them by the instrument (sector ETFs vs single stocks).
    """

    key = "4.1.momentum_rotation"
    section = "4.1"
    title = "Momentum rotation"
    style = "momentum"
    warmup = 300
    long_only = True
    param_grid = {
        "formation": (252, 189, 126),  # paper: 6-12 months
        "fraction": (0.34, 0.17),
    }

    def _scores(self, panel: Panel) -> pd.DataFrame:
        return _cumulative_return(panel.close, self.params["formation"], skip=0)

    def weights(self, panel: Panel) -> pd.DataFrame:
        return top_bottom_weights(
            self._scores(panel), fraction=self.params["fraction"], long_only=self.long_only
        )


class MomentumRotationMAFilter(SectorMomentumRotation):
    """4.1.1 Momentum rotation with a moving-average filter - Eq. (362)."""

    key = "4.1.1.momentum_rotation_ma_filter"
    section = "4.1.1"
    title = "Momentum rotation with MA filter"
    long_only = False
    param_grid = {
        "formation": (252, 126),
        "fraction": (0.34,),
        "ma_length": (200, 100),  # paper: 100-200 days
        "ma_kind": ("sma", "ema"),
    }
    warmup = 460

    def weights(self, panel: Panel) -> pd.DataFrame:
        base = top_bottom_weights(self._scores(panel), fraction=self.params["fraction"])
        ma = moving_average(panel.close, self.params["ma_length"], self.params["ma_kind"])
        above = panel.close > ma
        # Longs survive only above the MA, shorts only below it, Eq. (362).
        filtered = base.where((base > 0) & above, base.where((base < 0) & ~above, 0.0))
        return normalize_gross(filtered.fillna(0.0))


class DualMomentumRotation(SectorMomentumRotation):
    """4.1.2 Dual-momentum rotation - Eq. (363).

    Long-only.  Relative (cross-sectional) momentum picks the names; absolute
    (time-series) momentum of the equal-weighted universe index decides whether
    to be invested at all.  When the index is below its MA the strategy steps
    aside into cash, which stands in for the paper's uncorrelated ETF leg.
    """

    key = "4.1.2.dual_momentum"
    section = "4.1.2"
    title = "Dual-momentum rotation"
    long_only = True
    warmup = 460
    param_grid = {
        "formation": (252, 126),
        "fraction": (0.34,),
        "ma_length": (200, 100),
    }

    def weights(self, panel: Panel) -> pd.DataFrame:
        base = top_bottom_weights(
            self._scores(panel), fraction=self.params["fraction"], long_only=True
        )
        index = panel.close.mean(axis=1)  # equal-weighted broad-market proxy
        ma = index.rolling(self.params["ma_length"], min_periods=self.params["ma_length"] // 2).mean()
        risk_on = (index > ma).astype(float)
        return base.mul(risk_on, axis=0).fillna(0.0)


class MultiAssetTrendFollowing(Strategy):
    """4.6 Multi-asset trend following - Eq. (371)-(373).

    Long-only: keep names with positive cumulative return (optionally also above
    their long moving average) and weight them by ``R_cum``, ``R_cum/sigma`` or
    ``R_cum/sigma^2``.
    """

    key = "4.6.multi_asset_trend"
    section = "4.6"
    title = "Multi-asset trend following"
    style = "momentum"
    long_only = True
    warmup = 460
    param_grid = {
        "formation": (252, 126),
        "weighting": ("inverse_vol", "raw", "inverse_var"),
        "ma_filter": (True, False),
        "ma_length": (200, 100),
    }

    def weights(self, panel: Panel) -> pd.DataFrame:
        rcum = _cumulative_return(panel.close, self.params["formation"], skip=0)
        keep = rcum > 0
        if self.params["ma_filter"]:
            ma = moving_average(panel.close, self.params["ma_length"])
            keep = keep & (panel.close > ma)
        raw = rcum.where(keep, 0.0)
        mode = self.params["weighting"]
        if mode != "raw":
            power = 1 if mode == "inverse_vol" else 2
            raw = raw * inverse_vol_frame(panel.returns, self.params["formation"], power).fillna(0.0)
        return normalize_gross(raw.fillna(0.0))


class FuturesTrendFollowing(Strategy):
    """10.4 Trend following - Eq. (474)-(480).

    ``w_i = gamma * eta_i / sigma_i`` with ``eta_i = sign(R_i)``.  Options cover
    the paper's refinements: ``tanh`` smoothing of ``eta`` to stop sign flips on
    tiny ``R_i``, and demeaning ``R_i`` against the equal-weighted index so the
    long and short books stay balanced in a trending market.
    """

    key = "10.4.trend_following"
    section = "10.4"
    title = "Trend following (sign/vol weighting)"
    style = "momentum"
    warmup = 300
    param_grid = {
        "formation": (63, 21, 126, 252),
        "vol_window": (63, 21),
        "smoothing": ("tanh", "sign"),
        "demean_returns": (True, False),
        "dollar_neutral": (True, False),
    }

    def weights(self, panel: Panel) -> pd.DataFrame:
        rcum = _cumulative_return(panel.close, self.params["formation"], skip=0)
        if self.params["demean_returns"]:
            rcum = demean_cross_section(rcum)  # ~R_i, footnote 171
        if self.params["smoothing"] == "sign":
            eta = np.sign(rcum)  # Eq. (475)
        else:
            kappa = rcum.std(axis=1).replace(0.0, np.nan)
            eta = np.tanh(rcum.div(kappa, axis=0))
        inv_vol = inverse_vol_frame(panel.returns, self.params["vol_window"]).fillna(0.0)
        raw = eta * inv_vol  # Eq. (474)
        if self.params["dollar_neutral"]:
            raw = raw.sub(raw.mean(axis=1), axis=0)  # Eq. (477)
        return normalize_gross(raw.fillna(0.0))
