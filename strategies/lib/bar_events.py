"""
Converts OHLCV bars into hftbacktest feed events.

hftbacktest is built for tick-by-tick order-book data. When only bars are available we synthesize a
minimal two-sided book: at each price observation the best bid/ask are moved to
``price - half_spread`` / ``price + half_spread`` with effectively unlimited size. A marketable order
therefore fills at the observed price plus/minus ``half_spread`` (a simple, explicit cost model).

Two price observations are emitted per bar:
  * the bar's open at the bar start time, and
  * the bar's close 1 microsecond before the bar end time.
So at e.g. 10:30:00 the book reflects the 10:30 bar's open (or, if there is no next bar, the prior bar's close).
"""

import numpy as np
import pandas as pd

from hftbacktest import (
    BUY_EVENT,
    DEPTH_EVENT,
    EXCH_EVENT,
    LOCAL_EVENT,
    SELL_EVENT,
    event_dtype,
)

#: Price grid of the synthetic book. Finer than the real $0.01 tick so that quotes can sit exactly
#: half a cent either side of the observed price.
SYNTHETIC_TICK_SIZE = 0.0001
#: Size shown at each synthetic quote, i.e. "infinite" liquidity for retail-sized orders.
SYNTHETIC_DEPTH_QTY = 1e9

_BID = EXCH_EVENT | LOCAL_EVENT | BUY_EVENT | DEPTH_EVENT
_ASK = EXCH_EVENT | LOCAL_EVENT | SELL_EVENT | DEPTH_EVENT


def to_ns(ts) -> np.ndarray:
    """tz-aware timestamps -> int64 UTC nanoseconds."""
    return pd.DatetimeIndex(ts).tz_convert('UTC').as_unit('ns').asi8


def bars_to_events(
    bars: pd.DataFrame,
    half_spread: float = 0.005,
    tick_size: float = SYNTHETIC_TICK_SIZE,
) -> np.ndarray:
    times = np.concatenate([to_ns(bars.index), to_ns(bars['end']) - 1_000])
    prices = np.concatenate([bars['open'].to_numpy(), bars['close'].to_numpy()])
    order = np.argsort(times, kind='stable')
    times, prices = times[order], prices[order]

    half_ticks = max(int(round(half_spread / tick_size)), 0)
    events = []
    prev_bid = prev_ask = None
    for ts, px in zip(times, prices):
        mid_tick = int(round(px / tick_size))
        bid_tick = mid_tick - half_ticks
        ask_tick = max(mid_tick + half_ticks, bid_tick + 1)
        # Pull the previous quotes, then post the new ones.
        if prev_bid is not None:
            events.append((_BID, ts, ts, prev_bid * tick_size, 0.0, 0, 0, 0.0))
            events.append((_ASK, ts, ts, prev_ask * tick_size, 0.0, 0, 0, 0.0))
        events.append((_BID, ts, ts, bid_tick * tick_size, SYNTHETIC_DEPTH_QTY, 0, 0, 0.0))
        events.append((_ASK, ts, ts, ask_tick * tick_size, SYNTHETIC_DEPTH_QTY, 0, 0, 0.0))
        prev_bid, prev_ask = bid_tick, ask_tick
    if events:
        # Re-post the final quotes a day later so the feed doesn't end right at the last bar; otherwise the
        # backtester reports end-of-data before orders placed at the last bar's close can be processed.
        ts = int(times[-1]) + 86_400_000_000_000
        events.append((_BID, ts, ts, prev_bid * tick_size, SYNTHETIC_DEPTH_QTY, 0, 0, 0.0))
        events.append((_ASK, ts, ts, prev_ask * tick_size, SYNTHETIC_DEPTH_QTY, 0, 0, 0.0))
    return np.array(events, dtype=event_dtype)
