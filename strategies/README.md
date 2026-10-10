# Strategy backtests

Backtests of custom equity strategies, run on this repo's `hftbacktest` engine.

## Setup

```sh
python -m venv venv && . venv/bin/activate
pip install ./py-hftbacktest          # builds the engine from this fork (needs Rust)
pip install yfinance matplotlib
```

## Data

hftbacktest is designed for tick/order-book data. For US equities we usually only have bars, so
`lib/bar_events.py` turns OHLCV bars into a synthetic two-sided book centred on the bar price
(open at the bar start, close just before the bar end). Market orders fill at that price
± a configurable half-spread.

`lib/market_data.py` loads bars from Yahoo Finance (via `yfinance`) or from your own CSV files.
Yahoo look-back limits: 1h bars ~2 years, 2m/5m bars ~60 days, 1m bars ~7 days.

If running in a sandbox, the network must allow `query1.finance.yahoo.com`,
`query2.finance.yahoo.com` and `fc.yahoo.com`.

## Strategies

| Directory | Idea |
|---|---|
| [`amzn_meta_open_hour`](amzn_meta_open_hour) | Long AMZN / short META at the 09:30 ET open, close both 1 hour later, every trading day |
| [`same_slot_momentum`](same_slot_momentum) | Each day at 09:30, long whichever of two stocks did better in the 09:30-10:00 slot on prior days, short the other, close at 10:00 (Heston, Korajczyk & Sadka 2010). `universe.py` runs the paper's version on 40 large stocks (long top / short bottom quintile) and screens stocks and pairs with an out-of-sample split |

```sh
python strategies/amzn_meta_open_hour/backtest.py --help
python strategies/amzn_meta_open_hour/test_backtest.py   # sanity checks on hand-made bars
```
