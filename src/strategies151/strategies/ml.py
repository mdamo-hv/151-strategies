"""Machine-learning strategy: Section 3.17, single-stock KNN."""

from __future__ import annotations

import numpy as np
import pandas as pd

from strategies151.data.panel import Panel
from strategies151.strategies.base import Strategy, normalize_gross


def _features(panel: Panel, lengths: tuple[int, ...]) -> dict[str, pd.DataFrame]:
    """Predictor variables ``X_a(t)`` - Eq. (333)-(335).

    Moving averages of price and volume of varying lengths, expressed relative
    to the current close/volume so the features are scale-free across the very
    different price levels in the universe.  Everything uses data at or before
    ``t``, keeping the predictors out-of-sample w.r.t. the target.
    """
    feats: dict[str, pd.DataFrame] = {}
    close, volume = panel.close, panel.volume
    for length in lengths:
        ma = close.rolling(length, min_periods=max(2, length // 2)).mean()
        feats[f"price_ma_{length}"] = close / ma - 1.0
        vma = volume.rolling(length, min_periods=max(2, length // 2)).mean()
        feats[f"volume_ma_{length}"] = np.log(volume / vma.replace(0.0, np.nan))
    return feats


class SingleStockKNN(Strategy):
    """3.17 Machine learning - single-stock KNN - Eq. (332)-(341).

    For each name independently: build the normalised feature vectors
    ``~X_a(t)``, take the ``k`` nearest neighbours *inside the training window*
    and average their realised forward returns ``Y(t)`` to get the prediction
    ``Y(0)``.  Trade it through the paper's threshold rules with ``z1``/``z2``.

    The feature min/max used for normalisation (Eq. (337)) and the neighbour
    pool are both frozen from the training window, so the test window never
    informs its own prediction.
    """

    key = "3.17.single_stock_knn"
    section = "3.17"
    title = "Single-stock KNN"
    style = "machine-learning"
    warmup = 260
    param_grid = {
        "horizon": (21, 5),  # T: forward return being predicted
        "k": (0, 5, 20),  # 0 -> floor(sqrt(T*)) heuristic from the paper
        "feature_lengths": ((5, 21, 63),),
        "z1": (0.0, 0.01),  # entry threshold on the predicted return
        "z2": (0.0,),  # liquidation threshold
    }

    def __init__(self, **params):
        super().__init__(**params)
        self.models: dict[str, dict] = {}

    def fit(self, train: Panel, context: Panel | None = None) -> "SingleStockKNN":
        from sklearn.neighbors import NearestNeighbors

        horizon = self.params["horizon"]
        feats = _features(train, tuple(self.params["feature_lengths"]))
        # Target Y(t): realised forward return over the next `horizon` bars, Eq. (332).
        target = train.close.shift(-horizon) / train.close - 1.0
        self.models = {}
        for ticker in train.close.columns:
            frame = pd.DataFrame({name: f[ticker] for name, f in feats.items()})
            frame["__y"] = target[ticker]
            frame = frame.replace([np.inf, -np.inf], np.nan).dropna()
            if len(frame) < 30:
                continue
            x = frame.drop(columns="__y")
            lo, hi = x.min(), x.max()
            span = (hi - lo).replace(0.0, np.nan)
            x_norm = ((x - lo) / span).fillna(0.5)  # Eq. (337)
            k = self.params["k"] or int(np.floor(np.sqrt(len(frame))))
            k = max(1, min(k, len(frame)))
            model = NearestNeighbors(n_neighbors=k).fit(x_norm.to_numpy())
            self.models[ticker] = {
                "model": model,
                "y": frame["__y"].to_numpy(),
                "lo": lo,
                "span": span,
                "columns": list(x.columns),
            }
        return self

    def _predict(self, panel: Panel) -> pd.DataFrame:
        feats = _features(panel, tuple(self.params["feature_lengths"]))
        out = pd.DataFrame(np.nan, index=panel.close.index, columns=panel.close.columns)
        for ticker, state in self.models.items():
            frame = pd.DataFrame({name: f[ticker] for name, f in feats.items()})[state["columns"]]
            frame = frame.replace([np.inf, -np.inf], np.nan)
            usable = frame.dropna()
            if usable.empty:
                continue
            x_norm = ((usable - state["lo"]) / state["span"]).fillna(0.5)
            _, idx = state["model"].kneighbors(x_norm.to_numpy())
            out.loc[usable.index, ticker] = state["y"][idx].mean(axis=1)  # Eq. (339)
        return out

    def weights(self, panel: Panel) -> pd.DataFrame:
        if not self.models:
            return self._empty(panel)
        pred = self._predict(panel)
        z1, z2 = self.params["z1"], self.params["z2"]
        signal = pd.DataFrame(0.0, index=pred.index, columns=pred.columns)
        signal[pred > z1] = 1.0  # Eq. (341)
        signal[pred < -z1] = -1.0
        if z2 > 0:
            signal[(signal > 0) & (pred <= z2)] = 0.0
            signal[(signal < 0) & (pred >= -z2)] = 0.0
        return normalize_gross(signal)

    def describe(self) -> dict:
        info = super().describe()
        info["fitted_names"] = sorted(self.models)
        return info
