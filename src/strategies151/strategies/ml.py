"""Machine-learning strategy: Section 3.17, single-stock KNN."""

from __future__ import annotations

import numpy as np
import pandas as pd

from strategies151.data.panel import Panel
from strategies151.strategies.base import Strategy, normalize_gross


def _feature_stack(panel: Panel, lengths: tuple[int, ...]) -> tuple[np.ndarray, list[str]]:
    """Features as one ``(bars, names, features)`` array.

    The strategy is per-name, but the features are not: building a DataFrame per
    ticker inside the fold loop costs one pandas construction per name per
    parameter set per fold - 1.3 million of them across a 437-name study, which
    dominates everything else the strategy does. Computing the rolling
    statistics once for the whole cross-section and slicing with numpy is the
    same arithmetic without the overhead.
    """
    frames = _features(panel, lengths)
    names = list(frames)
    stack = np.stack([frames[name].to_numpy(dtype=float) for name in names], axis=-1)
    return stack, names


def _features(panel: Panel, lengths: tuple[int, ...]) -> dict[str, pd.DataFrame]:
    """Predictor variables ``X_a(t)`` - Eq. (333)-(335).

    Moving averages of price and volume of varying lengths, expressed relative
    to the current close/volume so the features are scale-free across the very
    different price levels in the universe.  Everything uses data at or before
    ``t``, keeping the predictors out-of-sample w.r.t. the target.
    """
    feats: dict[str, pd.DataFrame] = {}
    close, volume = panel.close, panel.volume
    # Zero-volume bars exist in a 500-name universe; log(0) is -inf, which the
    # callers drop as non-finite. Silencing keeps the run readable.
    with np.errstate(divide="ignore", invalid="ignore"):
        for length in lengths:
            ma = close.rolling(length, min_periods=max(2, length // 2)).mean()
            feats[f"price_ma_{length}"] = close / ma - 1.0
            vma = volume.rolling(length, min_periods=max(2, length // 2)).mean()
            feats[f"volume_ma_{length}"] = np.log(
                volume.replace(0.0, np.nan) / vma.replace(0.0, np.nan)
            )
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
        stack, names = _feature_stack(train, tuple(self.params["feature_lengths"]))
        stack = np.where(np.isfinite(stack), stack, np.nan)
        # Target Y(t): realised forward return over the next `horizon` bars, Eq. (332).
        target = (train.close.shift(-horizon) / train.close - 1.0).to_numpy(dtype=float)
        self.models = {}
        for position, ticker in enumerate(train.close.columns):
            x = stack[:, position, :]
            y = target[:, position]
            usable = np.isfinite(x).all(axis=1) & np.isfinite(y)
            if usable.sum() < 30:
                continue
            x, y = x[usable], y[usable]
            lo, hi = x.min(axis=0), x.max(axis=0)
            span = np.where(hi - lo > 0, hi - lo, np.nan)
            with np.errstate(invalid="ignore"):
                x_norm = np.nan_to_num((x - lo) / span, nan=0.5)  # Eq. (337)
            k = self.params["k"] or int(np.floor(np.sqrt(len(x))))
            k = max(1, min(k, len(x)))
            self.models[ticker] = {
                "model": NearestNeighbors(n_neighbors=k).fit(x_norm),
                "y": y,
                "lo": lo,
                "span": span,
                "features": names,
            }
        return self

    def _predict(self, panel: Panel) -> pd.DataFrame:
        stack, names = _feature_stack(panel, tuple(self.params["feature_lengths"]))
        stack = np.where(np.isfinite(stack), stack, np.nan)
        columns = list(panel.close.columns)
        out = np.full((stack.shape[0], len(columns)), np.nan)
        for position, ticker in enumerate(columns):
            state = self.models.get(ticker)
            if state is None:
                continue
            order = [names.index(f) for f in state["features"]]
            x = stack[:, position, :][:, order]
            usable = np.isfinite(x).all(axis=1)
            if not usable.any():
                continue
            with np.errstate(invalid="ignore"):
                x_norm = np.nan_to_num((x[usable] - state["lo"]) / state["span"], nan=0.5)
            _, idx = state["model"].kneighbors(x_norm)
            out[usable, position] = state["y"][idx].mean(axis=1)  # Eq. (339)
        return pd.DataFrame(out, index=panel.close.index, columns=columns)

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
