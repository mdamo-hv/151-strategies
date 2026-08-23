"""Research toolkit for Kakushadze & Serur, *151 Trading Strategies* (SSRN 3247865).

The package has three layers:

``strategies151.data``
    QuestDB access (``stooq.daily``) plus the loaders that populate it.
``strategies151.strategies``
    Portfolio-weight generators, one per strategy in the paper that is
    computable from daily OHLCV bars.  ``strategies151.catalog`` tracks all 151
    strategies, implemented or not, with the reason when not.
``strategies151.backtest``
    Walk-forward engine: 1 year of training data, 1 month held out, slid forward.
"""

__version__ = "0.1.0"
