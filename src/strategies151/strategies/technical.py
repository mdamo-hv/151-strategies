"""Single-stock technical strategies: Sections 3.11, 3.12, 3.13, 3.14, 3.15.

These are state machines: the paper specifies entry and liquidation rules rather
than a target weight, so each strategy first builds a per-name position state in
``{-1, 0, +1}`` and then normalises the states into portfolio weights.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from strategies151.data.panel import Panel
from strategies151.strategies.base import Strategy, moving_average, positions_from_signal


def _run_state_machine(
    enter_long: pd.DataFrame,
    exit_long: pd.DataFrame,
    enter_short: pd.DataFrame,
    exit_short: pd.DataFrame,
) -> pd.DataFrame:
    """Resolve entry/exit rules into a persistent -1/0/+1 position per name.

    Entries win over exits on the same bar (the paper's rules are written as
    "establish long / liquidate short"), and a position is held until its own
    exit rule fires.
    """
    index, columns = enter_long.index, enter_long.columns
    el, xl = enter_long.to_numpy(), exit_long.to_numpy()
    es, xs = enter_short.to_numpy(), exit_short.to_numpy()
    out = np.zeros((len(index), len(columns)))
    # The recursion is over time only, so the whole cross-section advances one
    # bar at a time. Looping over names as well costs rows x names Python
    # iterations, which at 500 names is ~2M per parameter set.
    current = np.zeros(len(columns))
    for row in range(len(index)):
        long_entry, short_entry = el[row], es[row]
        short_only = ~long_entry & short_entry      # long entry wins the bar
        no_entry = ~long_entry & ~short_entry       # exits only apply otherwise
        updated = np.where(long_entry, 1.0, current)
        updated = np.where(short_only, -1.0, updated)
        updated = np.where(no_entry & (current > 0) & xl[row], 0.0, updated)
        updated = np.where(no_entry & (current < 0) & xs[row], 0.0, updated)
        current = updated
        out[row] = current
    return pd.DataFrame(out, index=index, columns=columns)


class SingleMovingAverage(Strategy):
    """3.11 Single moving average - Eq. (319)-(321).

    Long while ``P > MA(T)``, short while ``P < MA(T)``.
    """

    key = "3.11.single_moving_average"
    section = "3.11"
    title = "Single moving average"
    style = "technical"
    warmup = 260
    param_grid = {
        "length": (200, 50, 100, 20),
        "kind": ("sma", "ema"),
        "lam": (0.9,),
        "long_only": (False, True),
    }

    def weights(self, panel: Panel) -> pd.DataFrame:
        ma = moving_average(panel.close, self.params["length"], self.params["kind"], self.params["lam"])
        signal = np.sign(panel.close - ma).fillna(0.0)  # Eq. (321)
        return positions_from_signal(signal, long_only=self.params["long_only"])


class TwoMovingAverages(Strategy):
    """3.12 Two moving averages - Eq. (322)-(323).

    Crossover of a fast and a slow MA, optionally with the paper's percentage
    stop-loss: liquidate a long if ``P < (1-Delta) P_1``, a short if
    ``P > (1+Delta) P_1``, where ``P_1`` is the previous day's price.
    """

    key = "3.12.two_moving_averages"
    section = "3.12"
    title = "Two moving averages"
    style = "technical"
    warmup = 120
    param_grid = {
        "fast": (10, 20, 5),
        "slow": (30, 50, 100),
        "kind": ("sma", "ema"),
        "stop_loss": (0.0, 0.02),  # Delta; 0 disables the stop
    }

    def weights(self, panel: Panel) -> pd.DataFrame:
        fast_len, slow_len = self.params["fast"], self.params["slow"]
        if fast_len >= slow_len:
            return self._empty(panel)
        fast = moving_average(panel.close, fast_len, self.params["kind"])
        slow = moving_average(panel.close, slow_len, self.params["kind"])
        cross_up, cross_down = fast > slow, fast < slow
        delta = self.params["stop_loss"]
        if delta <= 0:
            signal = (cross_up.astype(float) - cross_down.astype(float)).where(
                fast.notna() & slow.notna(), 0.0
            )
            return positions_from_signal(signal)
        prev = panel.close.shift(1)
        stop_long = panel.close < (1.0 - delta) * prev  # Eq. (323)
        stop_short = panel.close > (1.0 + delta) * prev
        valid = fast.notna() & slow.notna()
        signal = _run_state_machine(
            cross_up & valid, stop_long, cross_down & valid, stop_short
        )
        return positions_from_signal(signal)


class ThreeMovingAverages(Strategy):
    """3.13 Three moving averages - Eq. (324).

    Long only while ``MA(T1) > MA(T2) > MA(T3)``; the long is liquidated as soon
    as ``MA(T1) <= MA(T2)``, and mirror-image for shorts.
    """

    key = "3.13.three_moving_averages"
    section = "3.13"
    title = "Three moving averages"
    style = "technical"
    warmup = 120
    param_grid = {
        "t1": (3, 5, 10),
        "t2": (10, 20, 21),
        "t3": (21, 50, 100),
        "kind": ("sma", "ema"),
    }

    def weights(self, panel: Panel) -> pd.DataFrame:
        t1, t2, t3 = (self.params[k] for k in ("t1", "t2", "t3"))
        if not t1 < t2 < t3:
            return self._empty(panel)
        kind = self.params["kind"]
        ma1 = moving_average(panel.close, t1, kind)
        ma2 = moving_average(panel.close, t2, kind)
        ma3 = moving_average(panel.close, t3, kind)
        valid = ma1.notna() & ma2.notna() & ma3.notna()
        signal = _run_state_machine(
            enter_long=(ma1 > ma2) & (ma2 > ma3) & valid,
            exit_long=(ma1 <= ma2),
            enter_short=(ma1 < ma2) & (ma2 < ma3) & valid,
            exit_short=(ma1 >= ma2),
        )
        return positions_from_signal(signal)


class SupportResistance(Strategy):
    """3.14 Support and resistance - Eq. (325)-(328).

    Pivot ``C = (P_H + P_L + P_C)/3`` on the previous bar, resistance
    ``R = 2C - P_L``, support ``S = 2C - P_H``.
    """

    key = "3.14.support_resistance"
    section = "3.14"
    title = "Support and resistance"
    style = "technical"
    warmup = 60
    param_grid = {
        "lookback": (1, 3, 5),  # bars aggregated into the pivot
        "long_only": (False, True),
    }

    def weights(self, panel: Panel) -> pd.DataFrame:
        n = self.params["lookback"]
        high = panel.high.rolling(n, min_periods=n).max().shift(1)
        low = panel.low.rolling(n, min_periods=n).min().shift(1)
        close_prev = panel.close.shift(1)
        pivot = (high + low + close_prev) / 3.0  # Eq. (325)
        resistance = 2.0 * pivot - low  # Eq. (326)
        support = 2.0 * pivot - high  # Eq. (327)
        price = panel.close
        valid = pivot.notna()
        signal = _run_state_machine(
            enter_long=(price > pivot) & valid,
            exit_long=(price >= resistance),
            enter_short=(price < pivot) & valid,
            exit_short=(price <= support),
        )
        return positions_from_signal(signal, long_only=self.params["long_only"])


class DonchianChannel(Strategy):
    """3.15 Channel - Eq. (329)-(331).

    Donchian channel over ``T`` bars.  ``mode='reversion'`` buys the floor and
    sells the ceiling as written in Eq. (331); ``mode='breakout'`` takes the
    paper's alternative reading, where a break of the band starts a new trend.
    """

    key = "3.15.channel"
    section = "3.15"
    title = "Channel (Donchian)"
    style = "technical"
    warmup = 120
    param_grid = {
        "length": (20, 50, 10),
        "mode": ("breakout", "reversion"),
        "long_only": (False, True),
    }

    def weights(self, panel: Panel) -> pd.DataFrame:
        n = self.params["length"]
        upper = panel.high.rolling(n, min_periods=n).max().shift(1)  # Eq. (329)
        lower = panel.low.rolling(n, min_periods=n).min().shift(1)  # Eq. (330)
        price = panel.close
        valid = upper.notna() & lower.notna()
        if self.params["mode"] == "breakout":
            enter_long, enter_short = (price >= upper) & valid, (price <= lower) & valid
        else:
            enter_long, enter_short = (price <= lower) & valid, (price >= upper) & valid
        never = pd.DataFrame(False, index=price.index, columns=price.columns)
        signal = _run_state_machine(enter_long, never, enter_short, never)
        return positions_from_signal(signal, long_only=self.params["long_only"])
