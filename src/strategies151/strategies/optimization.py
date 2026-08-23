"""Portfolio-optimisation strategies: Sections 3.18, 3.18.1, 3.20."""

from __future__ import annotations

import numpy as np
import pandas as pd

from strategies151.data.panel import Panel
from strategies151.strategies.base import Strategy, demean_cross_section, normalize_gross


def _shrink(cov: np.ndarray, intensity: float) -> np.ndarray:
    """Shrink toward the diagonal.

    Footnote 62 of the paper: an ``N x N`` sample covariance is unstable unless
    ``T >> N``.  Shrinking toward ``diag(C)`` is the cheapest way to get a
    well-conditioned, out-of-sample-stable matrix without a full risk model.
    """
    if intensity <= 0:
        return cov
    return (1.0 - intensity) * cov + intensity * np.diag(np.diag(cov))


def _principal_component_cov(cov: np.ndarray, n_factors: int) -> np.ndarray:
    """Rebuild the covariance from its top ``n_factors`` principal components
    plus a diagonal specific-risk remainder (Appendix A's ``qrm.cov.pc``)."""
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]
    k = max(1, min(n_factors, len(eigvals) - 1))
    factor = eigvecs[:, :k] @ np.diag(eigvals[:k]) @ eigvecs[:, :k].T
    specific = np.clip(np.diag(cov) - np.diag(factor), 1e-12, None)
    return factor + np.diag(specific)


class StatArbOptimization(Strategy):
    """3.18 Statistical arbitrage - optimisation - Eq. (342)-(358).

    ``w = (1/lambda) C^-1 E``, optionally with the dollar-neutrality Lagrange
    multiplier of Eq. (358).  The covariance ``C`` is estimated on the training
    window and frozen; the expected returns ``E`` are the mean-reversion or
    momentum signal recomputed daily on the test window.
    """

    key = "3.18.stat_arb_optimization"
    section = "3.18"
    title = "Statistical arbitrage (mean-variance optimisation)"
    style = "optimization"
    warmup = 90
    param_grid = {
        "alpha": ("mean_reversion", "momentum"),
        "lookback": (5, 1, 21),
        "cov_model": ("pc", "shrunk", "sample"),
        "n_factors": (1, 2),
        "shrinkage": (0.5, 0.2),
        "dollar_neutral": (True, False),
    }

    def __init__(self, **params):
        super().__init__(**params)
        self.inv_cov: np.ndarray | None = None
        self.columns: list[str] = []

    def fit(self, train: Panel, context: Panel | None = None) -> "StatArbOptimization":
        rets = train.log_returns.dropna(how="all").fillna(0.0)
        self.columns = list(rets.columns)
        cov = np.cov(rets.to_numpy(), rowvar=False)
        cov = np.atleast_2d(cov)
        model = self.params["cov_model"]
        if model == "pc":
            cov = _principal_component_cov(cov, self.params["n_factors"])
        elif model == "shrunk":
            cov = _shrink(cov, self.params["shrinkage"])
        try:
            self.inv_cov = np.linalg.inv(cov)
        except np.linalg.LinAlgError:
            self.inv_cov = np.linalg.pinv(cov)
        return self

    def _expected_returns(self, panel: Panel) -> pd.DataFrame:
        lookback = self.params["lookback"]
        rets = np.log(panel.close / panel.close.shift(lookback))
        if self.params["alpha"] == "mean_reversion":
            return -demean_cross_section(rets)
        return demean_cross_section(rets)

    def weights(self, panel: Panel) -> pd.DataFrame:
        if self.inv_cov is None:
            return self._empty(panel)
        expected = self._expected_returns(panel).reindex(columns=self.columns)
        c_inv = self.inv_cov
        ones = np.ones(len(self.columns))
        denom = float(ones @ c_inv @ ones)
        out = np.zeros(expected.shape)
        values = expected.to_numpy()
        for row in range(len(values)):
            e = values[row]
            if not np.isfinite(e).all():
                continue
            w = c_inv @ e  # Eq. (350)/(353)
            if self.params["dollar_neutral"] and abs(denom) > 1e-12:
                w = w - c_inv @ ones * (ones @ c_inv @ e) / denom  # Eq. (358)
            out[row] = w
        frame = pd.DataFrame(out, index=expected.index, columns=self.columns)
        return normalize_gross(frame).reindex(columns=panel.close.columns).fillna(0.0)


class StatArbDollarNeutral(StatArbOptimization):
    """3.18.1 Statistical arbitrage with the dollar-neutrality constraint - Eq. (354)-(358)."""

    key = "3.18.1.stat_arb_dollar_neutral"
    section = "3.18.1"
    title = "Statistical arbitrage (dollar-neutral)"
    param_grid = {
        "alpha": ("mean_reversion", "momentum"),
        "lookback": (5, 1, 21),
        "cov_model": ("pc", "shrunk"),
        "n_factors": (1, 2),
        "shrinkage": (0.5,),
        "dollar_neutral": (True,),
    }


class AlphaCombo(Strategy):
    """3.20 Alpha combos - the 11-step recipe of [Kakushadze & Yu, 2017b].

    The component "alphas" are the other implemented strategies passed in via
    ``components``.  Their realised return series over the training window feed
    steps 1-7 (demean, normalise, cross-sectionally demean into ``Lambda``);
    steps 8-10 regress the expected alpha returns on ``Lambda`` and set the
    combination weights ``w_i = eta * eps_i / sigma_i``.

    One adaptation is unavoidable.  Step 9 regresses an ``N``-vector of expected
    alpha returns on the ``N x (M-1)`` matrix ``Lambda``, which is only
    well-posed when the number of alphas ``N`` exceeds the number of return
    observations ``M`` - the paper's setting, where ``N`` runs into the hundreds
    of thousands.  With a dozen alphas and a year of daily returns the system is
    underdetermined and every residual collapses to zero.  ``n_factors``
    therefore caps how many columns of ``Lambda`` (i.e. how many common risk
    directions) are retained, bounded above by ``N - 1``; the construction is
    otherwise the paper's, step for step.
    """

    key = "3.20.alpha_combo"
    section = "3.20"
    title = "Alpha combos"
    style = "combo"
    warmup = 320
    param_grid = {
        "d": (21, 63),  # moving-average length for expected alpha returns, Eq. (360)
        "n_factors": (2, 1),  # retained columns of Lambda (see class docstring)
    }
    #: Per-fold memo of component return series; the components are the same for
    #: every point of the grid, so they are evaluated once per (fold, universe).
    _component_cache: dict = {}

    def __init__(self, components=None, **params):
        super().__init__(**params)
        self.components = list(components or [])
        # The combo cannot be warm before its slowest component is.
        self.warmup = max([c.warmup for c in self.components] or [type(self).warmup]) + 21
        self.combo_weights: pd.Series | None = None

    def fit(self, train: Panel, context: Panel | None = None) -> "AlphaCombo":
        from strategies151.backtest.engine import strategy_returns

        if not self.components:
            self.combo_weights = None
            return self
        # Components must be evaluated over the warmed-up context, otherwise a
        # 12-month momentum alpha is all-NaN across a 12-month training window
        # and contributes nothing but zeros to the combination.
        evaluation = context if context is not None else train
        cache_key = (
            tuple(s.key for s in self.components),
            evaluation.close.index[0],
            evaluation.close.index[-1],
            tuple(evaluation.close.columns),
        )
        cached = type(self)._component_cache.get(cache_key)
        if cached is None:
            series = {}
            for strategy in self.components:
                strategy.fit(train, context=context)
                ret = strategy_returns(strategy, evaluation).reindex(train.close.index)
                if ret.notna().sum() > 20 and float(ret.std()) > 0:
                    series[strategy.key] = ret
            cached = pd.DataFrame(series).dropna() if series else pd.DataFrame()
            type(self)._component_cache.clear()  # only the current fold is reused
            type(self)._component_cache[cache_key] = cached
        if cached.shape[1] < 3 or len(cached) < 20:
            self.combo_weights = None
            return self
        self.combo_weights = self._combo_weights(
            cached, self.params["d"], self.params["n_factors"]
        )
        return self

    @staticmethod
    def _combo_weights(returns: pd.DataFrame, d: int, n_factors: int = 2) -> pd.Series:
        r = returns.to_numpy().T  # R_is: alphas x times, step 1
        n_alphas = r.shape[0]
        x = r - r.mean(axis=1, keepdims=True)  # step 2
        sigma = x.std(axis=1, ddof=1)  # step 3
        sigma = np.where(sigma > 0, sigma, np.nan)
        y = x / sigma[:, None]  # step 4
        m = y.shape[1] - 1
        y = y[:, :m]  # step 5
        lam = y - y.mean(axis=0, keepdims=True)  # step 6: cross-sectional demean
        keep = max(1, min(m - 1, n_factors, n_alphas - 1))  # step 7, capped
        # Take the highest-variance columns of Lambda so the retained directions
        # are the dominant common modes rather than an arbitrary prefix.
        order = np.argsort(np.nanvar(lam, axis=0))[::-1]
        lam = lam[:, order[:keep]]
        expected = np.nanmean(r[:, -d:], axis=1)  # step 8, Eq. (360)
        e_tilde = expected / sigma
        good = np.isfinite(e_tilde) & np.isfinite(lam).all(axis=1)
        eps = np.zeros_like(e_tilde)
        if good.sum() > keep:
            design = lam[good]
            coef, *_ = np.linalg.lstsq(design, e_tilde[good], rcond=None)  # step 9
            eps[good] = e_tilde[good] - design @ coef
        w = eps / sigma  # step 10
        w = np.nan_to_num(w)
        gross = np.abs(w).sum()
        if gross > 0:
            w = w / gross  # step 11
        return pd.Series(w, index=returns.columns)

    def weights(self, panel: Panel) -> pd.DataFrame:
        if self.combo_weights is None or not self.components:
            return self._empty(panel)
        total = self._empty(panel)
        for strategy in self.components:
            weight = float(self.combo_weights.get(strategy.key, 0.0))
            if weight == 0.0:
                continue
            total = total.add(strategy.weights(panel).mul(weight), fill_value=0.0)
        return normalize_gross(total.fillna(0.0))

    def describe(self) -> dict:
        info = super().describe()
        info["components"] = [s.key for s in self.components]
        if self.combo_weights is not None:
            info["combo_weights"] = self.combo_weights.round(4).to_dict()
        return info
