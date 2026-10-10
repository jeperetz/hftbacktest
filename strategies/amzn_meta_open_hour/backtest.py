"""
Pair trade: every trading day, at the 09:30 ET open, go long ``--long`` (default AMZN) and short
``--short`` (default META) with equal dollar notional, then close both legs ``--hold-minutes``
(default 60) later.

The simulation runs on hftbacktest's multi-asset backtester (built from this repo) using a synthetic
order book derived from intraday bars (see ``strategies/lib/bar_events.py``). Orders are market
orders that cross a configurable half-spread, with constant order latency and an optional
proportional fee.

Usage:
    python strategies/amzn_meta_open_hour/backtest.py                       # last 3 months, Yahoo 1h bars
    python strategies/amzn_meta_open_hour/backtest.py --start 2026-07-10 --end 2026-10-10
    python strategies/amzn_meta_open_hour/backtest.py --csv-dir my_bars/    # use your own <SYMBOL>.csv bars
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

LONG, SHORT = 0, 1

# Columns of the per-day output written by the strategy.
OUT_COLS = [
    'entry_ts', 'exit_ts',
    'long_qty', 'long_entry', 'long_exit',
    'short_qty', 'short_entry', 'short_exit',
    'equity_after', 'fees_after',
]


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
def open_hour_pair(hbt, entry_ts, hold_ns, notional, out):
    """
    For each entry timestamp: buy LONG / sell SHORT at market, wait ``hold_ns``, then flatten both.
    Writes one row per trade day into ``out`` and returns the number of completed days.
    """
    order_id = 0
    n_done = 0
    for i in range(len(entry_ts)):
        # ---- wait for the open ----
        if not _advance_to(hbt, entry_ts[i]):
            break

        long_depth = hbt.depth(LONG)
        short_depth = hbt.depth(SHORT)
        long_ask = long_depth.best_ask
        short_bid = short_depth.best_bid
        if not (np.isfinite(long_ask) and np.isfinite(short_bid)) or long_ask <= 0 or short_bid <= 0:
            continue

        long_qty = np.floor(notional / long_ask)
        short_qty = np.floor(notional / short_bid)
        if long_qty <= 0 or short_qty <= 0:
            continue

        # ---- enter: buy LONG, sell SHORT ----
        long_entry_id = order_id
        short_entry_id = order_id + 1
        order_id += 2
        hbt.submit_buy_order(LONG, long_entry_id, long_ask, long_qty, GTC, MARKET, False)
        hbt.submit_sell_order(SHORT, short_entry_id, short_bid, short_qty, GTC, MARKET, False)
        hbt.wait_order_response(LONG, long_entry_id, 1_000_000_000)
        hbt.wait_order_response(SHORT, short_entry_id, 1_000_000_000)

        lo = hbt.orders(LONG).get(long_entry_id)
        so = hbt.orders(SHORT).get(short_entry_id)
        if lo.status != FILLED or so.status != FILLED:
            return -1
        long_entry_px = lo.exec_price
        short_entry_px = so.exec_price

        # ---- hold ----
        exit_ts = entry_ts[i] + hold_ns
        if not _advance_to(hbt, exit_ts):
            break

        # ---- exit: sell LONG, buy back SHORT ----
        long_exit_id = order_id
        short_exit_id = order_id + 1
        order_id += 2
        long_pos = hbt.position(LONG)
        short_pos = -hbt.position(SHORT)
        hbt.submit_sell_order(LONG, long_exit_id, hbt.depth(LONG).best_bid, long_pos, GTC, MARKET, False)
        hbt.submit_buy_order(SHORT, short_exit_id, hbt.depth(SHORT).best_ask, short_pos, GTC, MARKET, False)
        hbt.wait_order_response(LONG, long_exit_id, 1_000_000_000)
        hbt.wait_order_response(SHORT, short_exit_id, 1_000_000_000)

        lo = hbt.orders(LONG).get(long_exit_id)
        so = hbt.orders(SHORT).get(short_exit_id)
        if lo.status != FILLED or so.status != FILLED:
            return -1

        ls = hbt.state_values(LONG)
        ss = hbt.state_values(SHORT)

        out[n_done, 0] = entry_ts[i]
        out[n_done, 1] = exit_ts
        out[n_done, 2] = long_qty
        out[n_done, 3] = long_entry_px
        out[n_done, 4] = lo.exec_price
        out[n_done, 5] = short_qty
        out[n_done, 6] = short_entry_px
        out[n_done, 7] = so.exec_price
        # Positions are flat here, so equity = balance - fees (cumulative since start).
        out[n_done, 8] = ls.balance + ss.balance - ls.fee - ss.fee
        out[n_done, 9] = ls.fee + ss.fee
        n_done += 1

        hbt.clear_inactive_orders(LONG)
        hbt.clear_inactive_orders(SHORT)
    return n_done


def load_bars(symbol, args):
    if args.csv_dir:
        return load_csv(Path(args.csv_dir) / f'{symbol}.csv', args.interval)
    bars = fetch_yahoo(symbol, args.start, args.end, args.interval)
    save_bars(bars, HERE / 'data' / f'{symbol}_{args.interval}_{args.start}_{args.end}.csv')
    return bars


def session_opens(long_bars, short_bars):
    """09:30 ET timestamps of days where both symbols have a bar starting exactly at the open."""
    def opens(b):
        idx = b.index
        return set(idx[(idx - idx.normalize()) == REGULAR_OPEN])
    common = sorted(opens(long_bars) & opens(short_bars))
    return pd.DatetimeIndex(common)


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


def run_backtest(long_bars, short_bars, opens, hold_ns, notional, half_spread, fee, latency_ns):
    # hftbacktest keeps a raw pointer to in-memory event arrays: they must outlive the backtest.
    feeds = [bars_to_events(long_bars, half_spread), bars_to_events(short_bars, half_spread)]
    hbt = HashMapMarketDepthBacktest([make_asset(ev, latency_ns, fee) for ev in feeds])
    entry_ts = to_ns(opens).astype(np.int64)
    out = np.full((len(entry_ts), len(OUT_COLS)), np.nan)
    n = open_hour_pair(hbt, entry_ts, hold_ns, float(notional), out)
    hbt.close()
    del feeds
    if n < 0:
        raise RuntimeError('An order was not filled; check the input data around the failing day.')
    return pd.DataFrame(out[:n], columns=OUT_COLS)


def price_at(bars, ts):
    """Reference price the synthetic book shows at ``ts``: the latest bar open/close observed at or before ts."""
    obs = pd.concat([
        pd.Series(bars['open'].to_numpy(), index=to_ns(bars.index)),
        pd.Series(bars['close'].to_numpy(), index=to_ns(bars['end']) - 1_000),
    ]).sort_index(kind='stable')
    obs = obs[~obs.index.duplicated(keep='last')]
    pos = obs.index.searchsorted(ts, side='right') - 1
    return obs.to_numpy()[pos]


def build_report(res, long_bars, short_bars, args):
    t = pd.DataFrame({
        'date': pd.to_datetime(res['entry_ts'].astype('int64'), utc=True).dt.tz_convert(NY).dt.date,
        'entry_time': pd.to_datetime(res['entry_ts'].astype('int64'), utc=True).dt.tz_convert(NY).dt.strftime('%H:%M'),
        'exit_time': pd.to_datetime(res['exit_ts'].astype('int64'), utc=True).dt.tz_convert(NY).dt.strftime('%H:%M'),
    })
    L, S = args.long, args.short
    t[f'{L}_shares'] = res['long_qty'].astype(int)
    t[f'{L}_entry'] = res['long_entry']
    t[f'{L}_exit'] = res['long_exit']
    t[f'{S}_shares'] = res['short_qty'].astype(int)
    t[f'{S}_entry'] = res['short_entry']
    t[f'{S}_exit'] = res['short_exit']
    t[f'{L}_pnl'] = (res['long_exit'] - res['long_entry']) * res['long_qty']
    t[f'{S}_pnl'] = (res['short_entry'] - res['short_exit']) * res['short_qty']
    fees_cum = res['fees_after'].to_numpy()
    t['fees'] = np.diff(np.concatenate([[0.0], fees_cum]))
    t['net_pnl'] = t[f'{L}_pnl'] + t[f'{S}_pnl'] - t['fees']
    t['cum_pnl'] = t['net_pnl'].cumsum()

    # Cross-check against hftbacktest's own accounting (balance - fees).
    engine_cum = res['equity_after'].to_numpy()
    mismatch = np.abs(engine_cum - t['cum_pnl'].to_numpy()).max() if len(t) else 0.0
    if mismatch > 1e-6:
        raise AssertionError(f'Fill-based P&L differs from engine accounting by {mismatch}')

    # Independent, frictionless recomputation straight from the bars (no spread, no fees).
    ent, ext = res['entry_ts'].astype('int64').to_numpy(), res['exit_ts'].astype('int64').to_numpy()
    l_in, l_out = price_at(long_bars, ent), price_at(long_bars, ext)
    s_in, s_out = price_at(short_bars, ent), price_at(short_bars, ext)
    t[f'{L}_ret_pct'] = (l_out / l_in - 1) * 100
    t[f'{S}_ret_pct'] = (s_out / s_in - 1) * 100
    t['spread_ret_pct'] = t[f'{L}_ret_pct'] - t[f'{S}_ret_pct']
    gross_frictionless = ((l_out - l_in) * res['long_qty'] + (s_in - s_out) * res['short_qty']).sum()
    return t, gross_frictionless


def summarize(t, gross_frictionless, args):
    L, S = args.long, args.short
    n = len(t)
    pnl = t['net_pnl']
    cum = t['cum_pnl']
    dd = (cum - cum.cummax().clip(lower=0)).min() if n else 0.0
    daily_ret = pnl / args.notional  # return on per-leg capital
    sharpe = daily_ret.mean() / daily_ret.std(ddof=1) * np.sqrt(252) if n > 1 and daily_ret.std() > 0 else np.nan
    tstat = daily_ret.mean() / (daily_ret.std(ddof=1) / np.sqrt(n)) if n > 1 and daily_ret.std() > 0 else np.nan
    lines = [
        f'Strategy: long {L} / short {S} at 09:30 ET open, close after {args.hold_minutes} min',
        f'Period: {t["date"].min()} .. {t["date"].max()} ({n} trading days), data: {args.interval} bars'
        + (f' from {args.csv_dir}' if args.csv_dir else ' from Yahoo Finance'),
        f'Sizing: ~${args.notional:,.0f} per leg (whole shares); half-spread ${args.half_spread:.4f}/share; '
        f'fee {args.fee * 1e4:.2f} bp; latency {args.latency_ms} ms',
        '',
        f'Net P&L:                 ${pnl.sum():,.2f}',
        f'  {L} leg:               ${t[f"{L}_pnl"].sum():,.2f}',
        f'  {S} leg:               ${t[f"{S}_pnl"].sum():,.2f}',
        f'  fees:                  ${t["fees"].sum():,.2f}',
        f'Frictionless gross P&L:  ${gross_frictionless:,.2f}  (no spread, no fees)',
        f'Return on per-leg capital: {pnl.sum() / args.notional * 100:.2f}%   '
        f'(on gross exposure: {pnl.sum() / (2 * args.notional) * 100:.2f}%)',
        '',
        f'Winning days:            {(pnl > 0).sum()} / {n} ({(pnl > 0).mean() * 100:.1f}%)',
        f'Avg / median day:        ${pnl.mean():,.2f} / ${pnl.median():,.2f}',
        f'Std of daily P&L:        ${pnl.std():,.2f}',
        f'Best day:                ${pnl.max():,.2f} ({t.loc[pnl.idxmax(), "date"]})',
        f'Worst day:               ${pnl.min():,.2f} ({t.loc[pnl.idxmin(), "date"]})',
        f'Max drawdown:            ${dd:,.2f}',
        f'Annualized Sharpe:       {sharpe:.2f}   (t-stat of mean daily P&L: {tstat:.2f})',
        f'Avg 1st-hour return:     {L} {t[f"{L}_ret_pct"].mean():+.3f}%, {S} {t[f"{S}_ret_pct"].mean():+.3f}%, '
        f'spread {t["spread_ret_pct"].mean():+.3f}%',
    ]
    return '\n'.join(lines)


def plot(t, args, path):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True, gridspec_kw={'height_ratios': [2, 1]})
    dates = pd.to_datetime(t['date'])
    ax1.plot(dates, t['cum_pnl'], color='#2563eb', lw=2)
    ax1.axhline(0, color='#9ca3af', lw=1)
    ax1.set_ylabel('Cumulative net P&L ($)')
    ax1.set_title(f'Long {args.long} / short {args.short}: 09:30 open, hold {args.hold_minutes} min '
                  f'(${args.notional:,.0f} per leg)')
    colors = np.where(t['net_pnl'] >= 0, '#16a34a', '#dc2626')
    ax2.bar(dates, t['net_pnl'], color=colors, width=0.8)
    ax2.axhline(0, color='#9ca3af', lw=1)
    ax2.set_ylabel('Daily P&L ($)')
    for ax in (ax1, ax2):
        ax.grid(alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    return True


def main():
    today = pd.Timestamp.now(tz=NY).normalize()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--long', default='AMZN')
    p.add_argument('--short', default='META')
    p.add_argument('--start', default=(today - pd.DateOffset(months=3)).strftime('%Y-%m-%d'))
    p.add_argument('--end', default=today.strftime('%Y-%m-%d'), help='exclusive')
    p.add_argument('--interval', default='1h', help='bar size: 1h (Yahoo keeps ~2y), 5m/2m (~60d), 1m (~7d)')
    p.add_argument('--hold-minutes', type=int, default=60)
    p.add_argument('--notional', type=float, default=10_000.0, help='dollars per leg')
    p.add_argument('--half-spread', type=float, default=0.005, help='$/share paid on every fill vs. the bar price')
    p.add_argument('--fee', type=float, default=0.0, help='proportional fee on traded value, e.g. 0.0001 = 1bp')
    p.add_argument('--latency-ms', type=float, default=1.0, help='order entry and response latency')
    p.add_argument('--csv-dir', help='read <SYMBOL>.csv bars from this directory instead of Yahoo')
    p.add_argument('--out-dir', default=str(HERE / 'results'))
    args = p.parse_args()

    step_ns = pd.Timedelta(args.interval.replace('m', 'min')).value
    if (args.hold_minutes * 60_000_000_000) % step_ns:
        print(f'warning: hold of {args.hold_minutes} min is not a multiple of the {args.interval} bar size; '
              f'the exit uses the latest price observed at or before the exit time.', file=sys.stderr)

    long_bars = load_bars(args.long, args)
    short_bars = load_bars(args.short, args)
    start, end = pd.Timestamp(args.start, tz=NY), pd.Timestamp(args.end, tz=NY)
    long_bars = long_bars[(long_bars.index >= start) & (long_bars.index < end)]
    short_bars = short_bars[(short_bars.index >= start) & (short_bars.index < end)]
    opens = session_opens(long_bars, short_bars)
    if len(opens) == 0:
        sys.exit('No trading days with a 09:30 bar for both symbols in the requested range.')

    res = run_backtest(
        long_bars, short_bars, opens,
        hold_ns=args.hold_minutes * 60_000_000_000,
        notional=args.notional,
        half_spread=args.half_spread,
        fee=args.fee,
        latency_ns=int(args.latency_ms * 1_000_000),
    )
    trades, gross_frictionless = build_report(res, long_bars, short_bars, args)
    summary = summarize(trades, gross_frictionless, args)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tag = f'{args.long}_{args.short}_{args.start}_{args.end}_{args.interval}'
    trades.round(4).to_csv(out / f'trades_{tag}.csv', index=False)
    (out / f'summary_{tag}.txt').write_text(summary + '\n')
    has_plot = plot(trades, args, out / f'pnl_{tag}.png')

    print(summary)
    print(f'\nPer-day trades: {out / f"trades_{tag}.csv"}')
    if has_plot:
        print(f'Chart:          {out / f"pnl_{tag}.png"}')


if __name__ == '__main__':
    main()
