# Best strategy per ticker

* Universe tested one name at a time: `NVDA, TSLA, MSFT, AMZN, WMT, JPM`
* Bars: `stooq.daily`, 3680 rows, 2012-01-03 to 2026-08-21
* Windows: 252 training days -> 21 test days, walked forward
* Transaction cost: 5.0 bps, delay 1

Roughly half the library is cross-sectional and cannot express a view on a single name: demeaning one stock's return against itself gives zero, and a top third that is also the bottom third nets out. Those are detected and excluded from the ranking with the reason recorded in `<TICKER>_strategies.csv`.

| ticker   | section   | best_strategy                             |   ann_return_% |   sharpe |   max_drawdown_% |   calmar |   buy_hold_sharpe |   sharpe_vs_buy_hold |   applicable_strategies |
|:---------|:----------|:------------------------------------------|---------------:|---------:|-----------------:|---------:|------------------:|---------------------:|------------------------:|
| NVDA     | 4.1.2     | Dual-momentum rotation                    |          60.82 |     1.37 |           -41.77 |     1.46 |              1.34 |                 0.02 |                      11 |
| TSLA     | 6.5       | Volatility targeting with risk-free asset |          11.49 |     0.75 |           -25.88 |     0.44 |              0.82 |                -0.07 |                      11 |
| MSFT     | 6.5       | Volatility targeting with risk-free asset |          15.74 |     0.95 |           -24.98 |     0.63 |              0.96 |                -0.01 |                      11 |
| AMZN     | 6.5       | Volatility targeting with risk-free asset |          12.51 |     0.72 |           -35.32 |     0.35 |              0.82 |                -0.1  |                      11 |
| WMT      | 6.5       | Volatility targeting with risk-free asset |          17.14 |     0.95 |           -23.31 |     0.74 |              0.91 |                 0.04 |                      11 |
| JPM      | 4.6       | Multi-asset trend following               |          16.08 |     0.85 |           -33.85 |     0.48 |              0.88 |                -0.03 |                      11 |

## Is it statistically relevant, or luck?

Three different questions, in increasing order of how much they ask:

* `p_vs_buy_hold` - a one-sided Newey-West t-test that the strategy's daily return exceeds that ticker's own buy & hold. Autocorrelation-robust, but it takes the winner as given. A value near 1 does not mean "no difference" - it means the difference runs the other way.
* `reality_check_p` / `spa_p` - White's Reality Check and Hansen's SPA, which bootstrap all applicable candidates jointly (stationary bootstrap, so serial dependence survives resampling) and ask how often chance alone produces a maximum this large. **This is the test that accounts for the winner having been selected as the best of many.**
* `deflated_sharpe_prob` - the probability the winner's Sharpe survives deflation for the number of trials, their dispersion, and the return distribution's skew and kurtosis.

| ticker   | best_strategy            |   excess_ann_return_% |   t_stat_vs_buy_hold |   p_vs_buy_hold |   reality_check_p |   spa_p |   deflated_sharpe_prob | verdict                                    |
|:---------|:-------------------------|----------------------:|---------------------:|----------------:|------------------:|--------:|-----------------------:|:-------------------------------------------|
| AMZN     | 6.5.volatility_targeting |                -13.19 |                -2.61 |           0.996 |             0.998 |       1 |                  0.677 | lower return than buy & hold (significant) |
| JPM      | 4.6.multi_asset_trend    |                 -6.99 |                -1.53 |           0.937 |             0.998 |       1 |                  0.658 | profitable, but no better than buy & hold  |
| MSFT     | 6.5.volatility_targeting |                -10.33 |                -2.88 |           0.998 |             0.997 |       1 |                  0.7   | lower return than buy & hold (significant) |
| NVDA     | 4.1.2.dual_momentum      |                -10.6  |                -1.35 |           0.911 |             0.991 |       1 |                  0.97  | profitable, but no better than buy & hold  |
| TSLA     | 6.5.volatility_targeting |                -35.7  |                -2.63 |           0.996 |             0.994 |       1 |                  0.822 | lower return than buy & hold (significant) |
| WMT      | 6.5.volatility_targeting |                 -2.26 |                -1.05 |           0.853 |             0.985 |       1 |                  0.882 | profitable, but no better than buy & hold  |

![significance](significance.png)

## NVDA

**4.1.2 Dual-momentum rotation** — 11 of 28 strategies applicable.

Parameters selected in-sample:

```
formation = 252
fraction = 0.34
ma_length = 200 (86% of folds)
```

![NVDA](NVDA.png)

## TSLA

**6.5 Volatility targeting with risk-free asset** — 11 of 28 strategies applicable.

Parameters selected in-sample:

```
target_vol = 0.15
vol_window = 63 (61% of folds)
max_leverage = 1.0
rebalance_threshold = 0.1 (78% of folds)
annualization = 252
```

![TSLA](TSLA.png)

## MSFT

**6.5 Volatility targeting with risk-free asset** — 11 of 28 strategies applicable.

Parameters selected in-sample:

```
target_vol = 0.15 (62% of folds)
vol_window = 21 (52% of folds)
max_leverage = 1.0 (64% of folds)
rebalance_threshold = 0.1 (58% of folds)
annualization = 252
```

![MSFT](MSFT.png)

## AMZN

**6.5 Volatility targeting with risk-free asset** — 11 of 28 strategies applicable.

Parameters selected in-sample:

```
target_vol = 0.15 (62% of folds)
vol_window = 21 (78% of folds)
max_leverage = 1.0 (78% of folds)
rebalance_threshold = 0.0 (58% of folds)
annualization = 252
```

![AMZN](AMZN.png)

## WMT

**6.5 Volatility targeting with risk-free asset** — 11 of 28 strategies applicable.

Parameters selected in-sample:

```
target_vol = 0.15 (46% of folds)
vol_window = 21 (67% of folds)
max_leverage = 1.0 (54% of folds)
rebalance_threshold = 0.1 (74% of folds)
annualization = 252
```

![WMT](WMT.png)

## JPM

**4.6 Multi-asset trend following** — 11 of 28 strategies applicable.

Parameters selected in-sample:

```
formation = 252 (50% of folds)
weighting = inverse_vol
ma_filter = True (54% of folds)
ma_length = 200 (75% of folds)
```

![JPM](JPM.png)
