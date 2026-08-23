# 151 Strategies - walk-forward out-of-sample results

* Universe: `NVDA, TSLA, MSFT, AMZN, WMT, JPM`
* Bars: `stooq.daily` in QuestDB, 3680 rows, 2012-01-03 to 2026-08-21
* Windows: 252 training days -> 21 test days, stepped by 21
* Out-of-sample span: 2016-03-11 to 2026-08-18 (125 folds)
* Transaction cost: 5.0 bps per unit traded, signals executed with delay 1

Percentages are annualised where applicable. Every number below is out-of-sample: parameters were chosen on the preceding training window only.

| section   | title                                              | style            | folds   |   days |   cagr |   ann_volatility |   sharpe |   max_drawdown |   calmar |   hit_rate |   ann_turnover |
|:----------|:---------------------------------------------------|:-----------------|:--------|-------:|-------:|-----------------:|---------:|---------------:|---------:|-----------:|---------------:|
| 6.5       | Volatility targeting with risk-free asset          | allocation       | 125     |   2624 |  26.02 |            15.3  |    1.589 |         -20.79 |    1.252 |      56.59 |          3.305 |
| -         | Equal-weighted buy & hold                          | benchmark        | <NA>    |   2623 |  37.46 |            25.41 |    1.38  |         -39.99 |    0.937 |      56.58 |          0.096 |
| 4.1       | Momentum rotation                                  | momentum         | 125     |   2624 |  61.11 |            44.58 |    1.291 |         -58.06 |    1.053 |      55.41 |         26.314 |
| 4.6       | Multi-asset trend following                        | momentum         | 125     |   2624 |  38.34 |            30.76 |    1.209 |         -47.05 |    0.815 |      55.14 |         23.27  |
| 4.1.2     | Dual-momentum rotation                             | momentum         | 125     |   2624 |  34.26 |            31.1  |    1.104 |         -51.33 |    0.668 |      45.27 |         25.354 |
| 3.1       | Price-momentum (long-only)                         | momentum         | 125     |   2624 |  44.07 |            42.15 |    1.075 |         -60.26 |    0.731 |      55.64 |         41.533 |
| 3.11      | Single moving average                              | technical        | 125     |   2624 |  22.68 |            24.61 |    0.955 |         -44.37 |    0.511 |      56.06 |         40.896 |
| 3.15      | Channel (Donchian)                                 | technical        | 125     |   2624 |  21.42 |            25.94 |    0.878 |         -42.7  |    0.502 |      52.02 |         21.362 |
| 3.14      | Support and resistance                             | technical        | 125     |   2624 |  14.22 |            25.47 |    0.649 |         -53.88 |    0.264 |      50.23 |        211.172 |
| 10.4      | Trend following (sign/vol weighting)               | momentum         | 125     |   2624 |  10.72 |            18.31 |    0.648 |         -36.41 |    0.294 |      53.96 |         35.669 |
| 3.4       | Low-volatility anomaly                             | factor           | 125     |   2624 |  10.46 |            21.17 |    0.576 |         -39.24 |    0.267 |      53.66 |          8.739 |
| 3.12      | Two moving averages                                | technical        | 125     |   2624 |   4.77 |            21.23 |    0.326 |         -37.31 |    0.128 |      52.29 |         13.573 |
| 3.13      | Three moving averages                              | technical        | 125     |   2624 |   4.46 |            25.79 |    0.299 |         -49.59 |    0.09  |      52.67 |         69.431 |
| 3.1       | Price-momentum                                     | momentum         | 125     |   2624 |   3.64 |            23.57 |    0.27  |         -58.47 |    0.062 |      50.11 |         44.543 |
| 3.17      | Single-stock KNN                                   | machine-learning | 125     |   2624 |   3.34 |            19.64 |    0.265 |         -56.48 |    0.059 |      50.8  |        183.007 |
| 4.3       | R-squared selectivity                              | factor           | 125     |   2624 |   2.48 |            17.23 |    0.228 |         -34.38 |    0.072 |      50.15 |         27.899 |
| 4.1.1     | Momentum rotation with MA filter                   | momentum         | 125     |   2624 |   1.2  |            29.59 |    0.189 |         -63.45 |    0.019 |      50.8  |         67.386 |
| 3.7       | Residual momentum (market-residual proxy)          | momentum         | 125     |   2624 |  -1.92 |            16.99 |   -0.029 |         -54.31 |   -0.035 |      49.05 |         27.659 |
| 3.8       | Pairs trading                                      | mean-reversion   | 125     |   2624 |  -2.94 |            14.54 |   -0.132 |         -39.6  |   -0.074 |      49.58 |         62.616 |
| 4.4       | Mean-reversion (internal bar strength)             | mean-reversion   | 125     |   2624 |  -4.77 |            19.43 |   -0.155 |         -59.75 |   -0.08  |      48.36 |        260.767 |
| 10.3      | Contrarian trading (market-index demeaned)         | mean-reversion   | 125     |   2624 |  -5.16 |            18.16 |   -0.201 |         -61.14 |   -0.084 |      49.7  |        110.937 |
| 3.20      | Alpha combos                                       | combo            | 125     |   2624 |  -8.26 |            23.56 |   -0.248 |         -77.07 |   -0.107 |      50.04 |        149.332 |
| 3.9.1     | Mean-reversion (multiple clusters)                 | mean-reversion   | 125     |   2624 |  -6.85 |            18.91 |   -0.281 |         -61.58 |   -0.111 |      48.74 |        180.443 |
| 3.6       | Multifactor portfolio (rank blend)                 | factor           | 125     |   2624 |  -7.52 |            17.56 |   -0.357 |         -59.67 |   -0.126 |      48.67 |         93.54  |
| 3.18.1    | Statistical arbitrage (dollar-neutral)             | optimization     | 125     |   2624 |  -6.8  |            15.33 |   -0.383 |         -65.71 |   -0.104 |      47.94 |        129.739 |
| 3.9       | Mean-reversion (single cluster)                    | mean-reversion   | 125     |   2624 |  -8.93 |            18.41 |   -0.416 |         -69.87 |   -0.128 |      48.86 |        164.233 |
| 3.10      | Mean-reversion (weighted regression)               | mean-reversion   | 125     |   2624 |  -8.47 |            16.45 |   -0.456 |         -67.8  |   -0.125 |      48.82 |        174.708 |
| 10.3.1    | Contrarian trading with volume filter              | mean-reversion   | 125     |   2624 | -16.45 |            27.47 |   -0.516 |         -88.31 |   -0.186 |      47.9  |        179.561 |
| 3.18      | Statistical arbitrage (mean-variance optimisation) | optimization     | 125     |   2624 |  -9.07 |            14.08 |   -0.605 |         -71.75 |   -0.126 |      48.36 |        122.567 |
