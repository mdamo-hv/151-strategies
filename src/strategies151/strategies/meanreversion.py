"""Mean-reversion strategies: Sections 3.8, 3.9, 3.9.1, 3.10, 4.4, 10.3, 10.3.1."""

from __future__ import annotations

import numpy as np
import pandas as pd

from strategies151.data.panel import Panel
from strategies151.strategies.base import (
    Strategy,
    demean_cross_section,
    inverse_vol_frame,
    normalize_gross,
    top_bottom_weights,
)


def _lookback_returns(panel: Panel, window: int, use_log: bool) -> pd.DataFrame:
    """``R_i = ln(P_i(t)/P_i(t-window))`` - Eq. (285)-(286) / (292)."""
    if use_log:
        return np.log(panel.close / panel.close.shift(window))
    return panel.close / panel.close.shift(window) - 1.0


class PairsTrading(Strategy):
    """3.8 Pairs trading - Eq. (283)-(291).

    Picks the most correlated pair in the universe on the training window, then
    trades the demeaned two-name return spread dollar-neutrally: short the name
    with the positive demeaned return ("rich"), buy the negative one ("cheap").
    The pair is selected in-sample and frozen for the test window.
    """

    key = "3.8.pairs_trading"
    section = "3.8"
    title = "Pairs trading"
    style = "mean-reversion"
    warmup = 60
    param_grid = {
        "lookback": (21, 5, 63),
        "min_correlation": (0.0, 0.3),
    }

    def __init__(self, **params):
        super().__init__(**params)
        self.pair: tuple[str, str] | None = None
        self.pair_correlation: float = float("nan")

    def fit(self, train: Panel, context: Panel | None = None) -> "PairsTrading":
        rets = train.log_returns.dropna(how="all")
        corr = rets.corr()
        # A Python double loop is 125k iterations at 500 names, per fold, per
        # parameter set; masking the upper triangle and taking an argmax is the
        # same search in one numpy call.
        values = corr.to_numpy(dtype=float).copy()
        values[~np.isfinite(values)] = -np.inf
        values[np.tril_indices_from(values)] = -np.inf
        best, best_rho = None, -np.inf
        if values.size and np.isfinite(values).any():
            flat = int(np.argmax(values))
            i, j = np.unravel_index(flat, values.shape)
            best_rho = float(values[i, j])
            best = (str(corr.columns[i]), str(corr.columns[j]))
        if best is not None and best_rho >= self.params["min_correlation"]:
            self.pair, self.pair_correlation = best, best_rho
        else:  # no pair clears the bar -> sit out the test window
            self.pair, self.pair_correlation = None, best_rho
        return self

    def weights(self, panel: Panel) -> pd.DataFrame:
        out = self._empty(panel)
        if self.pair is None:
            return out
        a, b = self.pair
        rets = _lookback_returns(panel, self.params["lookback"], use_log=True)[[a, b]]
        spread = demean_cross_section(rets)  # Eq. (287)-(289)
        legs = normalize_gross(-spread.fillna(0.0))  # short the rich leg, Eq. (297)
        out[[a, b]] = legs
        return out.fillna(0.0)

    def describe(self) -> dict:
        info = super().describe()
        info["fitted_pair"] = self.pair
        info["pair_correlation"] = self.pair_correlation
        return info


class MeanReversionSingleCluster(Strategy):
    """3.9 Mean-reversion, single cluster - Eq. (292)-(298).

    ``D_i = -gamma * ~R_i`` with ``~R_i`` the universe-demeaned lookback return,
    ``gamma`` set by ``sum |D_i| = I``.  Dollar-neutral by construction.
    """

    key = "3.9.mean_reversion_single_cluster"
    section = "3.9"
    title = "Mean-reversion (single cluster)"
    style = "mean-reversion"
    warmup = 90
    param_grid = {
        "lookback": (5, 1, 10, 21),
        "use_log": (True, False),
        "vol_scaling": ("none", "inverse_vol", "inverse_var"),
        "vol_window": (63,),
    }

    def weights(self, panel: Panel) -> pd.DataFrame:
        rets = _lookback_returns(panel, self.params["lookback"], self.params["use_log"])
        raw = -demean_cross_section(rets)  # Eq. (294) + (297)
        mode = self.params["vol_scaling"]
        if mode != "none":
            power = 1 if mode == "inverse_vol" else 2
            raw = raw * inverse_vol_frame(panel.returns, self.params["vol_window"], power).fillna(0.0)
            raw = raw.sub(raw.mean(axis=1), axis=0)  # restore Eq. (296)
        return normalize_gross(raw.fillna(0.0))


class MeanReversionMultiCluster(Strategy):
    """3.9.1 Mean-reversion, multiple clusters - Eq. (299)-(312).

    Clusters are learned from the training window by correlation-based
    agglomerative clustering (the paper's "clustering on pricing data" option),
    then the returns are demeaned *within* each cluster, i.e. residuals of the
    regression on the binary loadings matrix ``Lambda_iA``.
    """

    key = "3.9.1.mean_reversion_multi_cluster"
    section = "3.9.1"
    title = "Mean-reversion (multiple clusters)"
    style = "mean-reversion"
    warmup = 90
    param_grid = {
        "lookback": (5, 1, 21),
        "n_clusters": (2, 3),
        "vol_scaling": ("none", "inverse_vol"),
        "vol_window": (63,),
    }

    #: Clustering depends only on the training window and the cluster count, not
    #: on the lookback or volatility scaling, so the other grid axes reuse it.
    #: At 437 names an agglomerative fit is ~0.35s, and the grid has six
    #: parameter sets per cluster count.
    _cluster_cache: dict = {}

    def __init__(self, **params):
        super().__init__(**params)
        self.clusters: dict[str, int] = {}

    def fit(self, train: Panel, context: Panel | None = None) -> "MeanReversionMultiCluster":
        from sklearn.cluster import AgglomerativeClustering

        rets = train.log_returns.dropna(how="all").fillna(0.0)
        if rets.shape[1] < 2:  # nothing to cluster on a one-name universe
            self.clusters = {}
            return self
        k = min(self.params["n_clusters"], max(1, rets.shape[1] - 1))
        key = (rets.index[0], rets.index[-1], k, tuple(rets.columns))
        cached = type(self)._cluster_cache.get(key)
        if cached is None:
            corr = rets.corr().fillna(0.0).to_numpy()
            distance = np.clip(1.0 - corr, 0.0, 2.0)
            np.fill_diagonal(distance, 0.0)
            model = AgglomerativeClustering(
                n_clusters=k, metric="precomputed", linkage="average"
            )
            labels = model.fit_predict(distance)
            cached = dict(zip(rets.columns, labels.tolist()))
            if len(type(self)._cluster_cache) > 8:   # only the current fold matters
                type(self)._cluster_cache.clear()
            type(self)._cluster_cache[key] = cached
        self.clusters = dict(cached)
        return self

    def weights(self, panel: Panel) -> pd.DataFrame:
        rets = _lookback_returns(panel, self.params["lookback"], use_log=True)
        if not self.clusters:
            resid = demean_cross_section(rets)
        else:
            resid = rets.copy()
            labels = pd.Series(self.clusters).reindex(rets.columns).fillna(0).astype(int)
            for label in sorted(set(labels)):
                cols = labels.index[labels == label]
                block = rets[cols]
                resid[cols] = block.sub(block.mean(axis=1), axis=0)  # Eq. (309)
        raw = -resid
        if self.params["vol_scaling"] == "inverse_vol":
            raw = raw * inverse_vol_frame(panel.returns, self.params["vol_window"]).fillna(0.0)
        raw = raw.sub(raw.mean(axis=1), axis=0)
        return normalize_gross(raw.fillna(0.0))

    def describe(self) -> dict:
        info = super().describe()
        info["clusters"] = dict(self.clusters)
        return info


class MeanReversionWeightedRegression(Strategy):
    """3.10 Mean-reversion, weighted regression - Eq. (313)-(318).

    ``~R = Z * eps`` where ``eps = R - Omega (Omega^T Z Omega)^-1 Omega^T Z R``.
    The loadings ``Omega`` are the intercept plus the leading principal
    components of the training-window correlation matrix (the paper's non-binary
    loadings option); regression weights are ``z_i = 1/sigma_i^2``.
    """

    key = "3.10.mean_reversion_weighted_regression"
    section = "3.10"
    title = "Mean-reversion (weighted regression)"
    style = "mean-reversion"
    warmup = 90
    param_grid = {
        "lookback": (1, 5, 21),
        "n_factors": (1, 2),
        "vol_window": (63,),
        "weighted": (True, False),
    }

    def __init__(self, **params):
        super().__init__(**params)
        self.loadings: pd.DataFrame | None = None

    def fit(self, train: Panel, context: Panel | None = None) -> "MeanReversionWeightedRegression":
        rets = train.log_returns.dropna(how="all").fillna(0.0)
        corr = rets.corr().fillna(0.0).to_numpy()
        eigvals, eigvecs = np.linalg.eigh(corr)
        order = np.argsort(eigvals)[::-1]
        k = min(self.params["n_factors"], rets.shape[1] - 1)
        pcs = eigvecs[:, order[:k]] if k > 0 else np.zeros((rets.shape[1], 0))
        intercept = np.ones((rets.shape[1], 1))  # Eq. (312): intercept in Omega
        omega = np.hstack([intercept, pcs])
        self.loadings = pd.DataFrame(omega, index=rets.columns)
        return self

    def weights(self, panel: Panel) -> pd.DataFrame:
        rets = _lookback_returns(panel, self.params["lookback"], use_log=True)
        omega = (
            self.loadings.reindex(rets.columns).to_numpy()
            if self.loadings is not None
            else np.ones((rets.shape[1], 1))
        )
        if self.params["weighted"]:
            vol = panel.returns.rolling(
                self.params["vol_window"], min_periods=self.params["vol_window"] // 2
            ).std()
            z_frame = 1.0 / vol.pow(2).replace(0.0, np.nan)  # Eq. (316)
        else:
            z_frame = pd.DataFrame(1.0, index=rets.index, columns=rets.columns)

        out = np.zeros(rets.shape)
        values, z_values = rets.to_numpy(), z_frame.to_numpy()
        for row in range(rets.shape[0]):
            r, z = values[row], z_values[row]
            if not np.isfinite(r).all() or not np.isfinite(z).all():
                continue
            # `Z` is diagonal, so materialising it as an N x N matrix costs
            # ~190k floats per bar at 437 names. Broadcasting the diagonal keeps
            # the same algebra at O(N * K) instead of O(N^2).
            weighted_omega = z[:, None] * omega
            q = omega.T @ weighted_omega  # Eq. (317)
            try:
                q_inv = np.linalg.inv(q)
            except np.linalg.LinAlgError:
                q_inv = np.linalg.pinv(q)
            eps = r - omega @ (q_inv @ (weighted_omega.T @ r))  # Eq. (315)
            out[row] = -(z * eps)  # Eq. (314), traded contrarian
        frame = pd.DataFrame(out, index=rets.index, columns=rets.columns)
        return normalize_gross(frame)


class ETFMeanReversionIBS(Strategy):
    """4.4 Mean-reversion via Internal Bar Strength - Eq. (370).

    ``IBS = (P_C - P_L)/(P_H - P_L)``.  Sell the high-IBS ("rich") names, buy
    the low-IBS ("cheap") names.
    """

    key = "4.4.mean_reversion_ibs"
    section = "4.4"
    title = "Mean-reversion (internal bar strength)"
    style = "mean-reversion"
    warmup = 70
    param_grid = {
        "fraction": (0.34, 0.17),
        "smoothing": (1, 3),
        "vol_scaling": ("none", "inverse_vol"),
        "vol_window": (63,),
    }

    def weights(self, panel: Panel) -> pd.DataFrame:
        span = panel.high - panel.low
        ibs = (panel.close - panel.low) / span.replace(0.0, np.nan)
        if self.params["smoothing"] > 1:
            ibs = ibs.rolling(self.params["smoothing"], min_periods=1).mean()
        inv_vol = (
            inverse_vol_frame(panel.returns, self.params["vol_window"])
            if self.params["vol_scaling"] == "inverse_vol"
            else None
        )
        # Rank ascending on IBS: cheapest names get the long leg.
        return top_bottom_weights(-ibs, fraction=self.params["fraction"], inverse_vol=inv_vol)


class ContrarianTrading(Strategy):
    """10.3 Contrarian trading - Eq. (469)-(471).

    ``w_i = -gamma [R_i - R_m]`` against the equal-weighted index return, with
    the paper's optional ``1/sigma`` / ``1/sigma^2`` suppression.  Same algebra
    as 3.9 but stated on a weekly rebalance in the paper, which is what the
    default 5-day lookback reproduces.
    """

    key = "10.3.contrarian"
    section = "10.3"
    title = "Contrarian trading (market-index demeaned)"
    style = "mean-reversion"
    warmup = 90
    param_grid = {
        "lookback": (5, 10, 21),
        "vol_scaling": ("none", "inverse_vol", "inverse_var"),
        "vol_window": (63,),
    }

    def weights(self, panel: Panel) -> pd.DataFrame:
        rets = _lookback_returns(panel, self.params["lookback"], use_log=False)
        raw = -(rets.sub(rets.mean(axis=1), axis=0))  # Eq. (469)-(470)
        mode = self.params["vol_scaling"]
        if mode != "none":
            power = 1 if mode == "inverse_vol" else 2
            raw = raw * inverse_vol_frame(panel.returns, self.params["vol_window"], power).fillna(0.0)
            raw = raw.sub(raw.mean(axis=1), axis=0)
        return normalize_gross(raw.fillna(0.0))


class ContrarianMarketActivity(ContrarianTrading):
    """10.3.1 Contrarian trading with volume / market-activity filters - Eq. (472)-(473).

    ``v_i = ln(V_i / V'_i)`` compares this week's volume with the prior week's.
    Open interest has no analogue for cash equities, so the traded subset is the
    upper half by ``v_i`` only; the catalog records the partial substitution.
    """

    key = "10.3.1.contrarian_market_activity"
    section = "10.3.1"
    title = "Contrarian trading with volume filter"
    warmup = 90
    param_grid = {
        "lookback": (5, 10),
        "vol_scaling": ("none", "inverse_vol"),
        "vol_window": (63,),
        "activity_window": (5, 10),
    }

    def weights(self, panel: Panel) -> pd.DataFrame:
        base = super().weights(panel)
        window = self.params["activity_window"]
        vol_sum = panel.volume.rolling(window, min_periods=window).sum()
        activity = np.log(vol_sum / vol_sum.shift(window))  # Eq. (472)
        median = activity.median(axis=1)
        keep = activity.ge(median, axis=0)
        return normalize_gross(base.where(keep, 0.0).fillna(0.0))
