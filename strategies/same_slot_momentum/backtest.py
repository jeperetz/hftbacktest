"""
Same-half-hour momentum on a pair (after Heston, Korajczyk & Sadka, 2010, "Intraday Patterns in the
Cross-Section of Stock Returns", Journal of Finance).

The paper finds that a stock's return in a given half-hour of the day predicts its *relative* return in the
same half-hour on later days. Applied to two stocks and the opening half hour:

    Each day at 09:30 ET, look at the 09:30-10:00 returns of A and B on previous days. Go long the stock
    that did better, short the other (equal dollar notional), and close both at 10:00.

``--lookback 1`` uses only yesterday's opening half hour (the paper's lag-1 effect); larger values average
the A-minus-B return over the last N days.

Usage:
    python strategies/same_slot_momentum/backtest.py                    # AMZN vs META, last ~60 days, 30m bars
    python strategies/same_slot_momentum/backtest.py --lookback 5
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from numba import njit

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from hftbacktest import (  # noqa: E402
    BacktestAsset,
    HashMapMarketDepthBacktest,
    FILLED,
    GTC,
    MARKET,
)

from lib.bar_events import SYNTHETIC_TICK_SIZE, bars_to_events, to_ns  # noqa: E402
from lib.market_data import NY, REGULAR_OPEN, fetch_yahoo, load_csv, save_bars  # noqa: E402

OUT_COLS = ['entry_ts', 'exit_ts', 'direction', 'qty_a', 'entry_a', 'exit_a', 'qty_b', 'entry_b', 'exit_b',
            'equity_after']


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
def signed_pair(hbt, entry_ts, direction, hold_ns, notional, out):
    """
    direction[i] = +1: long asset 0 / short asset 1; -1: the reverse; 0: no trade.
    Enters at entry_ts[i] with market orders, exits hold_ns later. Returns the number of trades, -1 on a fill error.
    """
    order_id = 0
    n_done = 0
    for i in range(len(entry_ts)):
        if direction[i] == 0:
            continue
        if not _advance_to(hbt, entry_ts[i]):
            break
        buy_a = direction[i] > 0
        px_a = hbt.depth(0).best_ask if buy_a else hbt.depth(0).best_bid
        px_b = hbt.depth(1).best_bid if buy_a else hbt.depth(1).best_ask
        if not (np.isfinite(px_a) and np.isfinite(px_b)) or px_a <= 0 or px_b <= 0:
            continue
        qty_a = np.floor(notional / px_a)
        qty_b = np.floor(notional / px_b)
        if qty_a <= 0 or qty_b <= 0:
            continue

        # Enter.
        ea, eb = order_id, order_id + 1
        order_id += 2
        _submit(hbt, 0, ea, buy_a, qty_a)
        _submit(hbt, 1, eb, not buy_a, qty_b)
        hbt.wait_order_response(0, ea, 1_000_000_000)
        hbt.wait_order_response(1, eb, 1_000_000_000)
        oa = hbt.orders(0).get(ea)
        ob = hbt.orders(1).get(eb)
        if oa.status != FILLED or ob.status != FILLED:
            return -1
        entry_a, entry_b = oa.exec_price, ob.exec_price

        # Hold, then flatten.
        exit_ts = entry_ts[i] + hold_ns
        if not _advance_to(hbt, exit_ts):
            break
        xa, xb = order_id, order_id + 1
        order_id += 2
        _submit(hbt, 0, xa, not buy_a, abs(hbt.position(0)))
        _submit(hbt, 1, xb, buy_a, abs(hbt.position(1)))
        hbt.wait_order_response(0, xa, 1_000_000_000)
        hbt.wait_order_response(1, xb, 1_000_000_000)
        oa = hbt.orders(0).get(xa)
        ob = hbt.orders(1).get(xb)
        if oa.status != FILLED or ob.status != FILLED:
            return -1

        sa = hbt.state_values(0)
        sb = hbt.state_values(1)
        out[n_done, 0] = entry_ts[i]
        out[n_done, 1] = exit_ts
        out[n_done, 2] = direction[i]
        out[n_done, 3] = qty_a
        out[n_done, 4] = entry_a
        out[n_done, 5] = oa.exec_price
        out[n_done, 6] = qty_b
        out[n_done, 7] = entry_b
        out[n_done, 8] = ob.exec_price
        out[n_done, 9] = sa.balance + sb.balance - sa.fee - sb.fee
        n_done += 1
        hbt.clear_inactive_orders(0)
        hbt.clear_inactive_orders(1)
    return n_done


def make_asset(events, latency_ns, fee):
    return (
        BacktestAsset()
        .data(events)
        .linear_asset(1.0)
        .constant_order_latency(latency_ns, latency_ns)
        .risk_adverse_queue_model()
        .no_partial_fill_exchange()
        .trading_value_fee_model(fee, fee)
        .tick_size(SYNTHETIC_TICK_SIZE)
        .lot_size(1.0)
    )


def run(bars_a, bars_b, entry_ts, direction, hold_ns, notional, half_spread, fee, latency_ns):
    # hftbacktest keeps a raw pointer to in-memory event arrays: they must outlive the backtest.
    feeds = [bars_to_events(bars_a, half_spread), bars_to_events(bars_b, half_spread)]
    hbt = HashMapMarketDepthBacktest([make_asset(ev, latency_ns, fee) for ev in feeds])
    out = np.full((len(entry_ts), len(OUT_COLS)), np.nan)
    n = signed_pair(hbt, entry_ts.astype(np.int64), direction.astype(np.int64), hold_ns, float(notional), out)
    hbt.close()
    del feeds
    if n < 0:
        raise RuntimeError('An order was not filled; check the input data.')
    res = pd.DataFrame(out[:n], columns=OUT_COLS)
    d = res['direction']
    res['pnl_a'] = d * (res['exit_a'] - res['entry_a']) * res['qty_a']
    res['pnl_b'] = -d * (res['exit_b'] - res['entry_b']) * res['qty_b']
    fees = np.diff(np.concatenate([[0.0], res['equity_after'].to_numpy()])) - (res['pnl_a'] + res['pnl_b']).to_numpy()
    res['fees'] = -fees
    res['net_pnl'] = res['pnl_a'] + res['pnl_b'] - res['fees']
    if len(res) and abs(res['net_pnl'].cumsum().iloc[-1] - res['equity_after'].iloc[-1]) > 1e-6:
        raise AssertionError('Fill-based P&L differs from engine accounting')
    return res


def opening_slot(bars_a, bars_b, slot_minutes):
    """Per-day open/close of the first ``slot_minutes`` of the session for both symbols (days where both exist)."""
    def slot(b):
        first = b[(b.index - b.index.normalize()) == REGULAR_OPEN]
        return first
    a, b = slot(bars_a), slot(bars_b)
    days = a.index.intersection(b.index)
    a, b = a.loc[days], b.loc[days]
    step = pd.Timedelta(minutes=slot_minutes)
    if not ((a['end'] - a.index) == step).all():
        raise ValueError(f'Bars must be {slot_minutes}-minute bars so the first bar is exactly the opening slot.')
    return pd.DataFrame({
        'ret_a': a['close'] / a['open'] - 1,
        'ret_b': b['close'] / b['open'] - 1,
    }, index=days)


def signals(slot, lookback):
    """+1 = long A / short B today, -1 = the reverse, 0 = no trade (not enough history or a tie)."""
    diff = slot['ret_a'] - slot['ret_b']
    past = diff.shift(1).rolling(lookback, min_periods=lookback).mean()
    return np.sign(past).fillna(0).astype(int)


def stats(pnl, notional):
    n = len(pnl)
    if n == 0:
        return {}
    m, sd = pnl.mean(), pnl.std(ddof=1)
    se = sd / np.sqrt(n) if n > 1 else np.nan
    cum = pnl.cumsum()
    return {
        'days': n,
        'net_pnl': pnl.sum(),
        'avg_day': m,
        'ci95_lo': m - 1.96 * se,
        'ci95_hi': m + 1.96 * se,
        'win_rate': (pnl > 0).mean(),
        'std_day': sd,
        'max_dd': (cum - cum.cummax().clip(lower=0)).min(),
        'sharpe': m / sd * np.sqrt(252) if sd > 0 else np.nan,
        't_stat': m / se if se and se > 0 else np.nan,
    }


def main():
    today = pd.Timestamp.now(tz=NY).normalize()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--a', default='AMZN')
    p.add_argument('--b', default='META')
    p.add_argument('--start', default=(today - pd.Timedelta(days=59)).strftime('%Y-%m-%d'),
                   help='default: 59 days ago (Yahoo keeps 30m bars for ~60 days)')
    p.add_argument('--end', default=(today + pd.Timedelta(days=1)).strftime('%Y-%m-%d'), help='exclusive')
    p.add_argument('--interval', default='30m')
    p.add_argument('--lookback', type=int, nargs='+', default=[1, 5, 10],
                   help='days of past opening-slot returns to average for the signal (one run per value)')
    p.add_argument('--notional', type=float, default=10_000.0, help='dollars per leg')
    p.add_argument('--half-spread', type=float, default=0.005)
    p.add_argument('--fee', type=float, default=0.0)
    p.add_argument('--latency-ms', type=float, default=1.0)
    p.add_argument('--csv-dir', help='read <SYMBOL>.csv bars from this directory instead of Yahoo')
    p.add_argument('--out-dir', default=str(HERE / 'results'))
    args = p.parse_args()

    def load(sym):
        if args.csv_dir:
            return load_csv(Path(args.csv_dir) / f'{sym}.csv', args.interval)
        bars = fetch_yahoo(sym, args.start, args.end, args.interval)
        save_bars(bars, HERE / 'data' / f'{sym}_{args.interval}_{args.start}_{args.end}.csv')
        return bars

    bars_a, bars_b = load(args.a), load(args.b)
    slot_minutes = int(pd.Timedelta(args.interval.replace('m', 'min')).total_seconds() // 60)
    slot = opening_slot(bars_a, bars_b, slot_minutes)
    entry_ts = to_ns(slot.index)
    hold_ns = slot_minutes * 60_000_000_000
    kw = dict(hold_ns=hold_ns, notional=args.notional, half_spread=args.half_spread, fee=args.fee,
              latency_ns=int(args.latency_ms * 1_000_000))

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tag = f'{args.a}_{args.b}_{slot.index[0].date()}_{slot.index[-1].date()}_{args.interval}'

    rows = []
    trade_logs = {}
    for lb in args.lookback:
        sig = signals(slot, lb)
        res = run(bars_a, bars_b, entry_ts, sig.to_numpy(), **kw)
        res.insert(0, 'date', pd.to_datetime(res['entry_ts'].astype('int64'), utc=True).dt.tz_convert(NY).dt.date)
        trade_logs[f'momentum_lb{lb}'] = res
        rows.append({'strategy': f'momentum, {lb}-day lookback', **stats(res['net_pnl'], args.notional)})

        # Baselines over exactly the same days: always long A / short B, and always the reverse.
        traded = sig != 0
        for name, d in [(f'always long {args.a} / short {args.b}', 1), (f'always long {args.b} / short {args.a}', -1)]:
            base = run(bars_a, bars_b, entry_ts, np.where(traded, d, 0), **kw)
            rows.append({'strategy': f'  baseline ({lb}-day days): {name}', **stats(base['net_pnl'], args.notional)})

    table = pd.DataFrame(rows).set_index('strategy')
    for name, res in trade_logs.items():
        res.drop(columns=['entry_ts', 'exit_ts']).round(4).rename(columns={
            'qty_a': f'{args.a}_shares', 'entry_a': f'{args.a}_entry', 'exit_a': f'{args.a}_exit',
            'qty_b': f'{args.b}_shares', 'entry_b': f'{args.b}_entry', 'exit_b': f'{args.b}_exit',
            'pnl_a': f'{args.a}_pnl', 'pnl_b': f'{args.b}_pnl',
        }).to_csv(out / f'trades_{name}_{tag}.csv', index=False)

    header = (
        f'Same-half-hour momentum: {args.a} vs {args.b}, enter 09:30 ET, exit {slot_minutes} min later\n'
        f'Data: {args.interval} bars, {slot.index[0].date()} .. {slot.index[-1].date()} ({len(slot)} sessions)\n'
        f'Sizing ~${args.notional:,.0f} per leg; half-spread ${args.half_spread:.4f}/share; fee {args.fee * 1e4:.2f} bp\n'
        f'direction: +1 = long {args.a} / short {args.b}\n'
    )
    fmt = table.copy()
    for c in ['net_pnl', 'avg_day', 'ci95_lo', 'ci95_hi', 'std_day', 'max_dd']:
        fmt[c] = fmt[c].map(lambda v: f'{v:,.2f}')
    fmt['win_rate'] = fmt['win_rate'].map(lambda v: f'{v * 100:.0f}%')
    for c in ['sharpe', 't_stat']:
        fmt[c] = fmt[c].map(lambda v: f'{v:.2f}')
    report = header + '\n' + fmt.to_string()
    (out / f'summary_{tag}.txt').write_text(report + '\n')
    print(report)


if __name__ == '__main__':
    main()
