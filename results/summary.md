# 151 Strategies - walk-forward out-of-sample results

* Universe: `NVDA, TSLA, MSFT, AMZN, WMT, JPM`
* Bars: `stooq.daily` in QuestDB, 3680 rows, 2012-01-03 to 2026-08-21
* Windows: 252 training days -> 21 test days, stepped by 21
* Out-of-sample span: 2016-03-11 to 2026-08-18 (125 folds)
* Transaction cost: 5.0 bps per unit traded, signals executed with delay 1


`ann_return_%` is the annualised compound return (CAGR), `max_drawdown_%` the worst peak-to-trough loss, and `calmar` the ratio of the two. `sharpe` is `mean/sd * sqrt(252)`, the statistic used in Appendix A of the paper. All figures are net of costs and entirely out-of-sample: parameters were chosen on the preceding training window only.

| section   | title                                              | style            | folds   |   days |   ann_return_% |   sharpe |   max_drawdown_% |   calmar |   ann_vol_% |   hit_rate_% |   ann_turnover_x |
|:----------|:---------------------------------------------------|:-----------------|:--------|-------:|---------------:|---------:|-----------------:|---------:|------------:|-------------:|-----------------:|
| 6.5       | Volatility targeting with risk-free asset          | allocation       | 125     |   2624 |          28.89 |     1.62 |           -21.4  |     1.35 |       16.47 |        56.59 |             3.11 |
| -         | Equal-weighted buy & hold                          | benchmark        | <NA>    |   2624 |          37.61 |     1.38 |           -39.99 |     0.94 |       25.41 |        56.59 |             0.1  |
| 4.1       | Momentum rotation                                  | momentum         | 125     |   2624 |          61.11 |     1.29 |           -58.06 |     1.05 |       44.58 |        55.41 |            26.31 |
| 4.6       | Multi-asset trend following                        | momentum         | 125     |   2624 |          38.34 |     1.21 |           -47.05 |     0.81 |       30.76 |        55.14 |            23.27 |
| 4.1.2     | Dual-momentum rotation                             | momentum         | 125     |   2624 |          34.26 |     1.1  |           -51.33 |     0.67 |       31.1  |        45.27 |            25.35 |
| 3.1       | Price-momentum (long-only)                         | momentum         | 125     |   2624 |          44.07 |     1.07 |           -60.26 |     0.73 |       42.15 |        55.64 |            41.53 |
| 3.11      | Single moving average                              | technical        | 125     |   2624 |          22.68 |     0.95 |           -44.37 |     0.51 |       24.61 |        56.06 |            40.9  |
| 3.15      | Channel (Donchian)                                 | technical        | 125     |   2624 |          21.42 |     0.88 |           -42.7  |     0.5  |       25.94 |        52.02 |            21.36 |
| 3.14      | Support and resistance                             | technical        | 125     |   2624 |          14.22 |     0.65 |           -53.88 |     0.26 |       25.47 |        50.23 |           211.17 |
| 10.4      | Trend following (sign/vol weighting)               | momentum         | 125     |   2624 |          10.72 |     0.65 |           -36.41 |     0.29 |       18.31 |        53.96 |            35.67 |
| 3.4       | Low-volatility anomaly                             | factor           | 125     |   2624 |          10.46 |     0.58 |           -39.24 |     0.27 |       21.17 |        53.66 |             8.74 |
| 3.12      | Two moving averages                                | technical        | 125     |   2624 |           4.77 |     0.33 |           -37.31 |     0.13 |       21.23 |        52.29 |            13.57 |
| 3.13      | Three moving averages                              | technical        | 125     |   2624 |           4.46 |     0.3  |           -49.59 |     0.09 |       25.79 |        52.67 |            69.43 |
| 3.1       | Price-momentum                                     | momentum         | 125     |   2624 |           3.64 |     0.27 |           -58.47 |     0.06 |       23.57 |        50.11 |            44.54 |
| 3.17      | Single-stock KNN                                   | machine-learning | 125     |   2624 |           3.34 |     0.27 |           -56.48 |     0.06 |       19.64 |        50.8  |           183.01 |
| 4.3       | R-squared selectivity                              | factor           | 125     |   2624 |           2.48 |     0.23 |           -34.38 |     0.07 |       17.23 |        50.15 |            27.9  |
| 4.1.1     | Momentum rotation with MA filter                   | momentum         | 125     |   2624 |           1.2  |     0.19 |           -63.45 |     0.02 |       29.59 |        50.8  |            67.39 |
| 3.7       | Residual momentum (market-residual proxy)          | momentum         | 125     |   2624 |          -1.92 |    -0.03 |           -54.31 |    -0.04 |       16.99 |        49.05 |            27.66 |
| 3.8       | Pairs trading                                      | mean-reversion   | 125     |   2624 |          -2.94 |    -0.13 |           -39.6  |    -0.07 |       14.54 |        49.58 |            62.62 |
| 4.4       | Mean-reversion (internal bar strength)             | mean-reversion   | 125     |   2624 |          -4.77 |    -0.15 |           -59.75 |    -0.08 |       19.43 |        48.36 |           260.77 |
| 10.3      | Contrarian trading (market-index demeaned)         | mean-reversion   | 125     |   2624 |          -5.16 |    -0.2  |           -61.14 |    -0.08 |       18.16 |        49.7  |           110.94 |
| 3.20      | Alpha combos                                       | combo            | 125     |   2624 |          -8.26 |    -0.25 |           -77.07 |    -0.11 |       23.56 |        50.04 |           149.33 |
| 3.9.1     | Mean-reversion (multiple clusters)                 | mean-reversion   | 125     |   2624 |          -6.85 |    -0.28 |           -61.58 |    -0.11 |       18.91 |        48.74 |           180.44 |
| 3.6       | Multifactor portfolio (rank blend)                 | factor           | 125     |   2624 |          -7.52 |    -0.36 |           -59.67 |    -0.13 |       17.56 |        48.67 |            93.54 |
| 3.18.1    | Statistical arbitrage (dollar-neutral)             | optimization     | 125     |   2624 |          -6.8  |    -0.38 |           -65.71 |    -0.1  |       15.33 |        47.94 |           129.74 |
| 3.9       | Mean-reversion (single cluster)                    | mean-reversion   | 125     |   2624 |          -8.93 |    -0.42 |           -69.87 |    -0.13 |       18.41 |        48.86 |           164.23 |
| 3.10      | Mean-reversion (weighted regression)               | mean-reversion   | 125     |   2624 |          -8.47 |    -0.46 |           -67.8  |    -0.12 |       16.45 |        48.82 |           174.71 |
| 10.3.1    | Contrarian trading with volume filter              | mean-reversion   | 125     |   2624 |         -16.45 |    -0.52 |           -88.31 |    -0.19 |       27.47 |        47.9  |           179.56 |
| 3.18      | Statistical arbitrage (mean-variance optimisation) | optimization     | 125     |   2624 |          -9.07 |    -0.6  |           -71.75 |    -0.13 |       14.08 |        48.36 |           122.57 |

## How each ticker behaved on its own

Buy and hold, same out-of-sample window, no strategy involved.

| ticker   |   ann_return_% |   sharpe |   max_drawdown_% |   calmar |   ann_vol_% |   hit_rate_% |
|:---------|---------------:|---------:|-----------------:|---------:|------------:|-------------:|
| NVDA     |          72.01 |     1.34 |           -66.34 |     1.09 |       49.37 |        54.5  |
| MSFT     |          25.38 |     0.96 |           -37.15 |     0.68 |       27.5  |        53.73 |
| WMT      |          19.02 |     0.91 |           -25.74 |     0.74 |       21.83 |        53.62 |
| JPM      |          22.37 |     0.88 |           -43.63 |     0.51 |       27.13 |        52.74 |
| AMZN     |          23.86 |     0.82 |           -56.15 |     0.42 |       32.73 |        53.47 |
| TSLA     |          36.03 |     0.82 |           -73.63 |     0.49 |       58.66 |        51.75 |

## Which tickers the strategies made money on

Contribution of each ticker to strategy P&L, averaged across the strategy library. Contributions are arithmetic and additive, so a strategy's per-ticker contributions sum to its annualised return.

| ticker   |   strategies |   mean_contribution_ann_pct |   best_contribution_ann_pct |   worst_contribution_ann_pct |   profitable_strategies_pct |   avg_gross_weight |   avg_net_weight |
|:---------|-------------:|----------------------------:|----------------------------:|-----------------------------:|----------------------------:|-------------------:|-----------------:|
| NVDA     |           28 |                        6.93 |                       33.53 |                        -3.4  |                       67.86 |               0.19 |             0.09 |
| TSLA     |           28 |                        1.89 |                       18.69 |                        -8.64 |                       53.57 |               0.16 |             0.04 |
| MSFT     |           28 |                        0.26 |                        4.07 |                        -3.7  |                       50    |               0.16 |             0.06 |
| JPM      |           28 |                        0.2  |                        3.56 |                        -4.14 |                       57.14 |               0.15 |             0.05 |
| AMZN     |           28 |                        0.11 |                        5.43 |                        -7.55 |                       53.57 |               0.15 |             0.04 |
| WMT      |           28 |                       -0.34 |                        6.66 |                        -3.92 |                       35.71 |               0.17 |             0.07 |

Per-strategy detail is in `ticker_attribution.csv`.
