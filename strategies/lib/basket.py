"""
Multi-asset long/short basket trading on hftbacktest: on each trade day, open market-order positions in many
assets at once at a given timestamp and close them all a fixed time later.
"""

import numpy as np
from numba import njit

from hftbacktest import (
    BacktestAsset,
    HashMapMarketDepthBacktest,
    FILLED,
    GTC,
    MARKET,
)

from lib.bar_events import SYNTHETIC_TICK_SIZE, bars_to_events


@njit
def _advance_to(hbt, ts):
    # Before the first elapse() the clock is unset (i64::MAX); elapse(0) starts it at the first feed event.
    if hbt.current_timestamp == np.iinfo(np.int64).max:
        if hbt.elapse(0) != 0:
            return False
    now = hbt.current_timestamp
    if ts > now:
        return hbt.elapse(ts - now) == 0
    return True


@njit
def _submit(hbt, asset, order_id, buy, qty):
    depth = hbt.depth(asset)
    if buy:
        hbt.submit_buy_order(asset, order_id, depth.best_ask, qty, GTC, MARKET, False)
    else:
        hbt.submit_sell_order(asset, order_id, depth.best_bid, qty, GTC, MARKET, False)


@njit
def signed_basket(hbt, entry_ts, weights, hold_ns, notional, qty, entry_px, exit_px, equity):
    """
    weights[i, j] > 0: long asset j on day i, < 0: short, 0: flat. Each position is ~``notional`` dollars
    (whole shares). Fills are written to qty/entry_px/exit_px[i, j] (qty signed) and the cumulative
    engine equity (balance - fees, all assets) after day i to equity[i]. Returns 0, or -1 on a fill error.
    """
    n_assets = weights.shape[1]
    order_id = 0
    ids = np.zeros(n_assets, np.int64)
    for i in range(len(entry_ts)):
        if not _advance_to(hbt, entry_ts[i]):
            break
        # Enter.
        last = -1
        for j in range(n_assets):
            w = weights[i, j]
            if w == 0:
                continue
            depth = hbt.depth(j)
            px = depth.best_ask if w > 0 else depth.best_bid
            if not np.isfinite(px) or px <= 0:
                continue
            q = np.floor(notional / px)
            if q <= 0:
                continue
            ids[j] = order_id
            _submit(hbt, j, order_id, w > 0, q)
            qty[i, j] = q if w > 0 else -q
            last = j
            order_id += 1
        if last < 0:
            continue
        hbt.wait_order_response(last, ids[last], 1_000_000_000)
        for j in range(n_assets):
            if qty[i, j] != 0:
                o = hbt.orders(j).get(ids[j])
                if o.status != FILLED:
                    return -1
                entry_px[i, j] = o.exec_price

        # Hold, then flatten.
        if not _advance_to(hbt, entry_ts[i] + hold_ns):
            break
        for j in range(n_assets):
            if qty[i, j] != 0:
                ids[j] = order_id
                _submit(hbt, j, order_id, qty[i, j] < 0, abs(hbt.position(j)))
                last = j
                order_id += 1
        hbt.wait_order_response(last, ids[last], 1_000_000_000)
        eq = 0.0
        for j in range(n_assets):
            if qty[i, j] != 0:
                o = hbt.orders(j).get(ids[j])
                if o.status != FILLED:
                    return -1
                exit_px[i, j] = o.exec_price
            sv = hbt.state_values(j)
            eq += sv.balance - sv.fee
            hbt.clear_inactive_orders(j)
        equity[i] = eq
    return 0


def run_basket(bars_list, entry_ts, weights, hold_ns, notional, half_spread=0.005, fee=0.0, latency_ns=1_000_000):
    """Returns (qty, entry_px, exit_px, equity) arrays shaped like ``weights`` (equity: one value per day)."""
    # hftbacktest keeps a raw pointer to in-memory event arrays: they must outlive the backtest.
    feeds = [bars_to_events(b, half_spread) for b in bars_list]
    assets = [
        BacktestAsset()
        .data(ev)
        .linear_asset(1.0)
        .constant_order_latency(latency_ns, latency_ns)
        .risk_adverse_queue_model()
        .no_partial_fill_exchange()
        .trading_value_fee_model(fee, fee)
        .tick_size(SYNTHETIC_TICK_SIZE)
        .lot_size(1.0)
        for ev in feeds
    ]
    hbt = HashMapMarketDepthBacktest(assets)
    shape = weights.shape
    qty, entry_px, exit_px = np.zeros(shape), np.full(shape, np.nan), np.full(shape, np.nan)
    equity = np.full(shape[0], np.nan)
    rc = signed_basket(hbt, entry_ts.astype(np.int64), weights.astype(np.float64), hold_ns, float(notional),
                       qty, entry_px, exit_px, equity)
    hbt.close()
    del feeds
    if rc < 0:
        raise RuntimeError('An order was not filled; check the input data.')
    return qty, entry_px, exit_px, equity
