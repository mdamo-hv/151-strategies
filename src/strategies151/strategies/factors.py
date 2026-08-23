"""Cross-sectional factor strategies: Sections 3.4, 3.6, 4.3."""

from __future__ import annotations

import numpy as np
import pandas as pd

from strategies151.data.panel import Panel
from strategies151.strategies.base import Strategy, rank_demean, top_bottom_weights
from strategies151.strategies.momentum import _cumulative_return


class LowVolatilityAnomaly(Strategy):
    """3.4 Low-volatility anomaly - Section 3.4, ``sigma_i`` from Eq. (270).

    Buy the bottom slice by historical volatility, short the top slice.  The
    paper's estimation window is 6 months to 1 year with no skip period.
    """

    key = "3.4.low_volatility"
    section = "3.4"
    title = "Low-volatility anomaly"
    style = "factor"
    warmup = 300
    param_grid = {
        "vol_window": (252, 126),
        "fraction": (0.34, 0.17),
        "long_only": (False, True),
    }

    def scores(self, panel: Panel) -> pd.DataFrame:
        window = self.params["vol_window"]
        vol = panel.returns.rolling(window, min_periods=window // 2).std()
        return -vol  # high score == low volatility

    def weights(self, panel: Panel) -> pd.DataFrame:
        return top_bottom_weights(
            self.scores(panel),
            fraction=self.params["fraction"],
            long_only=self.params["long_only"],
        )


class MultifactorPortfolio(Strategy):
    """3.6 Multifactor portfolio - Eq. (275)-(277).

    Blends factor rankings through demeaned ranks ``s_Ai`` averaged into a
    combined score ``s_i``.  With no fundamental data in ``stooq.daily`` the
    available factors are price-momentum, low-volatility and short-horizon
    reversal - the value and earnings legs are unavailable (see catalog).
    """

    key = "3.6.multifactor"
    section = "3.6"
    title = "Multifactor portfolio (rank blend)"
    style = "factor"
    warmup = 320
    param_grid = {
        "factors": (
            ("momentum", "low_vol", "reversal"),
            ("momentum", "low_vol"),
            ("momentum", "reversal"),
        ),
        "fraction": (0.34,),
        "momentum_formation": (252,),
        "momentum_skip": (21,),
        "vol_window": (252,),
        "reversal_lookback": (5,),
    }

    def _factor_frames(self, panel: Panel) -> dict[str, pd.DataFrame]:
        p = self.params
        vol = panel.returns.rolling(p["vol_window"], min_periods=p["vol_window"] // 2).std()
        return {
            "momentum": _cumulative_return(panel.close, p["momentum_formation"], p["momentum_skip"]),
            "low_vol": -vol,
            "reversal": -(panel.close / panel.close.shift(p["reversal_lookback"]) - 1.0),
        }

    def weights(self, panel: Panel) -> pd.DataFrame:
        frames = self._factor_frames(panel)
        chosen = [frames[name] for name in self.params["factors"] if name in frames]
        if not chosen:
            return self._empty(panel)
        blended = sum(rank_demean(f) for f in chosen) / len(chosen)  # Eq. (276)-(277)
        valid = chosen[0].notna()
        for frame in chosen[1:]:
            valid &= frame.notna()
        blended = blended.where(valid)
        return top_bottom_weights(blended, fraction=self.params["fraction"])


class RSquaredSelectivity(Strategy):
    """4.3 R-squared - Eq. (364)-(369).

    Serial regression of each name on the market proxy gives Jensen's alpha and
    ``R^2``; "selectivity" is ``1 - R^2``.  Names are double-sorted: low ``R^2``
    and high alpha are bought, high ``R^2`` and low alpha are sold.  The paper's
    SMB/HML/MOM legs are unavailable for this universe, so the regression is a
    single-factor market model (recorded in the catalog).
    """

    key = "4.3.r_squared"
    section = "4.3"
    title = "R-squared selectivity"
    style = "factor"
    warmup = 320
    param_grid = {
        "window": (252, 126),
        "fraction": (0.34,),
        "blend": ("alpha_over_r2", "alpha_only"),
    }

    def weights(self, panel: Panel) -> pd.DataFrame:
        rets = panel.returns
        market = rets.mean(axis=1)
        window = self.params["window"]
        min_periods = max(40, window // 2)
        var_m = market.rolling(window, min_periods=min_periods).var()
        mean_m = market.rolling(window, min_periods=min_periods).mean()
        alphas, r2s = {}, {}
        for ticker in rets.columns:
            series = rets[ticker]
            cov = series.rolling(window, min_periods=min_periods).cov(market)
            beta = cov / var_m.replace(0.0, np.nan)
            alphas[ticker] = series.rolling(window, min_periods=min_periods).mean() - beta * mean_m
            corr = series.rolling(window, min_periods=min_periods).corr(market)
            r2s[ticker] = corr.pow(2)  # Eq. (366) for a single-factor regression
        alpha = pd.DataFrame(alphas)
        r2 = pd.DataFrame(r2s)
        if self.params["blend"] == "alpha_only":
            scores = alpha
        else:
            selectivity = 1.0 - r2  # [Amihud & Goyenko, 2013]
            scores = (rank_demean(alpha) + rank_demean(selectivity)) / 2.0
        return top_bottom_weights(scores, fraction=self.params["fraction"])


class VolatilityTargeting(Strategy):
    """6.5 Volatility targeting with a risk-free asset.

    ``w = sigma* / sigma`` in the risky asset, ``1 - w`` in cash, rebalanced only
    when ``|Delta w| / w`` exceeds ``kappa``.  The risky asset here is the
    equal-weighted universe index; the cash leg earns nothing, which understates
    the strategy's return by the T-bill yield (recorded in the catalog).
    """

    key = "6.5.volatility_targeting"
    section = "6.5"
    title = "Volatility targeting with risk-free asset"
    style = "allocation"
    long_only = True
    warmup = 90
    param_grid = {
        "target_vol": (0.15, 0.10, 0.20),
        "vol_window": (63, 21),
        "max_leverage": (1.0, 2.0),
        "rebalance_threshold": (0.10, 0.0),
        "annualization": (252,),
    }

    def weights(self, panel: Panel) -> pd.DataFrame:
        index_returns = panel.returns.mean(axis=1)
        window = self.params["vol_window"]
        realised = index_returns.rolling(window, min_periods=window // 2).std() * np.sqrt(
            self.params["annualization"]
        )
        raw = (self.params["target_vol"] / realised.replace(0.0, np.nan)).clip(
            upper=self.params["max_leverage"]
        )
        kappa = self.params["rebalance_threshold"]
        if kappa > 0:
            held, series = np.nan, []
            for value in raw.to_numpy():
                if not np.isfinite(value):
                    series.append(np.nan)
                    continue
                if not np.isfinite(held) or abs(value - held) / max(abs(held), 1e-12) > kappa:
                    held = value
                series.append(held)
            raw = pd.Series(series, index=raw.index)
        raw = raw.fillna(0.0)
        n = panel.close.shape[1]
        return pd.DataFrame(
            np.outer(raw.to_numpy(), np.full(n, 1.0 / n)),
            index=panel.close.index,
            columns=panel.close.columns,
        )
