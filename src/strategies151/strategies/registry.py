"""Registry of the strategies implemented against ``stooq.daily`` bars."""

from __future__ import annotations

from typing import Iterable, Type

from strategies151.strategies.base import Strategy
from strategies151.strategies.factors import (
    LowVolatilityAnomaly,
    MultifactorPortfolio,
    RSquaredSelectivity,
    VolatilityTargeting,
)
from strategies151.strategies.meanreversion import (
    ContrarianMarketActivity,
    ContrarianTrading,
    ETFMeanReversionIBS,
    MeanReversionMultiCluster,
    MeanReversionSingleCluster,
    MeanReversionWeightedRegression,
    PairsTrading,
)
from strategies151.strategies.ml import SingleStockKNN
from strategies151.strategies.momentum import (
    DualMomentumRotation,
    FuturesTrendFollowing,
    MomentumRotationMAFilter,
    MultiAssetTrendFollowing,
    PriceMomentum,
    PriceMomentumLongOnly,
    ResidualMomentum,
    SectorMomentumRotation,
)
from strategies151.strategies.optimization import (
    AlphaCombo,
    StatArbDollarNeutral,
    StatArbOptimization,
)
from strategies151.strategies.technical import (
    DonchianChannel,
    SingleMovingAverage,
    SupportResistance,
    ThreeMovingAverages,
    TwoMovingAverages,
)

STRATEGY_CLASSES: tuple[Type[Strategy], ...] = (
    PriceMomentum,
    PriceMomentumLongOnly,
    LowVolatilityAnomaly,
    MultifactorPortfolio,
    ResidualMomentum,
    PairsTrading,
    MeanReversionSingleCluster,
    MeanReversionMultiCluster,
    MeanReversionWeightedRegression,
    SingleMovingAverage,
    TwoMovingAverages,
    ThreeMovingAverages,
    SupportResistance,
    DonchianChannel,
    SingleStockKNN,
    StatArbOptimization,
    StatArbDollarNeutral,
    SectorMomentumRotation,
    MomentumRotationMAFilter,
    DualMomentumRotation,
    RSquaredSelectivity,
    ETFMeanReversionIBS,
    MultiAssetTrendFollowing,
    VolatilityTargeting,
    ContrarianTrading,
    ContrarianMarketActivity,
    FuturesTrendFollowing,
)

#: 3.20 combines other strategies, so it is registered separately and built by
#: :func:`build_alpha_combo` once its components exist.
COMBO_CLASS = AlphaCombo

REGISTRY: dict[str, Type[Strategy]] = {cls.key: cls for cls in STRATEGY_CLASSES}


def implemented_keys() -> list[str]:
    return list(REGISTRY) + [COMBO_CLASS.key]


def get(key: str) -> Type[Strategy]:
    if key == COMBO_CLASS.key:
        return COMBO_CLASS
    if key not in REGISTRY:
        raise KeyError(f"unknown strategy '{key}'; known: {sorted(REGISTRY)}")
    return REGISTRY[key]


def build(key: str, **params) -> Strategy:
    return get(key)(**params)


def build_alpha_combo(component_keys: Iterable[str] | None = None, **params) -> AlphaCombo:
    """3.20 alpha combo over the given component strategies (defaults to a
    momentum / mean-reversion / technical spread)."""
    # Only stateless components are used by default: each one is re-evaluated on
    # every fold, and a component with its own fit step would multiply the cost.
    keys = list(component_keys) if component_keys else [
        "3.1.price_momentum",
        "3.1.price_momentum_long_only",
        "3.4.low_volatility",
        "3.6.multifactor",
        "3.9.mean_reversion_single_cluster",
        "3.11.single_moving_average",
        "3.12.two_moving_averages",
        "3.14.support_resistance",
        "3.15.channel",
        "4.4.mean_reversion_ibs",
        "4.6.multi_asset_trend",
        "10.4.trend_following",
    ]
    components = [build(k) for k in keys]
    return COMBO_CLASS(components=components, **params)


def resolve(keys: Iterable[str] | None = None) -> list[Strategy]:
    """Instantiate strategies by key, or every implemented strategy."""
    if keys is None:
        selected = implemented_keys()
    else:
        selected = list(keys)
    out: list[Strategy] = []
    for key in selected:
        out.append(build_alpha_combo() if key == COMBO_CLASS.key else build(key))
    return out
