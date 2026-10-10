"""
Same-half-hour momentum across a universe of large US stocks (Heston, Korajczyk & Sadka, 2010).

1. Portfolio test (the paper's setup): each day at 09:30 ET rank the stocks by their 09:30-10:00 return over
   the previous N days; buy the top quintile, short the bottom quintile (equal dollars per position), exit at
   10:00. Simulated on hftbacktest with all stocks as assets of one multi-asset backtest.
2. Cross-sectional regressions: average slope of today's opening-slot return on the return k days ago.
3. Candidate search with an out-of-sample check: rank individual stocks and all pairs on the first half of the
   period, then report how the top picks did in the second half.

Usage:
    python strategies/same_slot_momentum/universe.py --data-dir strategies/same_slot_momentum/data/universe
"""

import argparse
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from lib.bar_events import to_ns  # noqa: E402
from lib.basket import run_basket  # noqa: E402
from lib.market_data import NY, REGULAR_OPEN, load_csv  # noqa: E402

SLOT = pd.Timedelta(minutes=30)


def load_universe(data_dir):
    bars = {}
    for f in sorted(Path(data_dir).glob('*.csv')):
        bars[f.stem] = load_csv(f, '30m')
    return bars


def slot_table(bars):
    """Per-day opening-slot return (signal) plus the entry/exit reference prices the backtest trades at."""
    ret, px_in, px_out = {}, {}, {}
    for s, b in bars.items():
        tod = b.index - b.index.normalize()
        first = b[tod == REGULAR_OPEN]
        nxt = b[tod == REGULAR_OPEN + SLOT]
        ret[s] = first['close'] / first['open'] - 1
        px_in[s] = first['open']
        # At 10:00 the synthetic book shows the 10:00 bar's open.
        px_out[s] = pd.Series(nxt['open'].to_numpy(), index=nxt.index - SLOT)
    ret, px_in, px_out = pd.DataFrame(ret), pd.DataFrame(px_in), pd.DataFrame(px_out)
    days = ret.dropna().index.intersection(px_out.dropna().index)
    return ret.loc[days], px_in.loc[days], px_out.loc[days]


def lagged_signal(ret, lookback):
    return ret.shift(1).rolling(lookback, min_periods=lookback).mean()


def quintile_weights(signal, frac=0.2):
    k = max(int(round(signal.shape[1] * frac)), 1)
    w = np.zeros(signal.shape)
    for i, row in enumerate(signal.to_numpy()):
        if np.isnan(row).any():
            continue
        order = np.argsort(row)
        w[i, order[-k:]] = 1
        w[i, order[:k]] = -1
    return w


def day_stats(pnl):
    pnl = pd.Series(pnl).dropna()
    n = len(pnl)
    m, sd = pnl.mean(), pnl.std(ddof=1)
    se = sd / np.sqrt(n)
    return dict(days=n, net_pnl=pnl.sum(), avg_day=m, ci95_lo=m - 1.96 * se, ci95_hi=m + 1.96 * se,
                win_rate=(pnl > 0).mean(), t_stat=m / se if se > 0 else np.nan)


def fama_macbeth(ret, max_lag):
    """Average cross-sectional slope of r_t on r_{t-k} (both demeaned across stocks each day), with t-stats."""
    rows = []
    x = ret.sub(ret.mean(axis=1), axis=0)
    for k in range(1, max_lag + 1):
        slopes = []
        for t in range(k, len(x)):
            y, z = x.iloc[t].to_numpy(), x.iloc[t - k].to_numpy()
            if np.var(z) > 0:
                slopes.append(np.dot(z, y) / np.dot(z, z))
        s = np.array(slopes)
        rows.append(dict(lag=k, avg_slope=s.mean(), t_stat=s.mean() / (s.std(ddof=1) / np.sqrt(len(s))),
                         days=len(s)))
    return pd.DataFrame(rows).set_index('lag')


def pair_pnl(ret, px_in, px_out, a, b, lookback, notional, half_spread):
    """Vectorized daily P&L of the 2-stock version (same fill model as the engine)."""
    sig = np.sign((ret[a] - ret[b]).shift(1).rolling(lookback, min_periods=lookback).mean())
    long_a = sig > 0
    ent_a = np.where(long_a, px_in[a] + half_spread, px_in[a] - half_spread)
    ext_a = np.where(long_a, px_out[a] - half_spread, px_out[a] + half_spread)
    ent_b = np.where(long_a, px_in[b] - half_spread, px_in[b] + half_spread)
    ext_b = np.where(long_a, px_out[b] + half_spread, px_out[b] - half_spread)
    qa, qb = np.floor(notional / ent_a), np.floor(notional / ent_b)
    pnl = sig * ((ext_a - ent_a) * qa - (ext_b - ent_b) * qb)
    return pd.Series(np.where(sig != 0, pnl, np.nan), index=ret.index)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--data-dir', default=str(HERE / 'data' / 'universe'), help='one 30m-bar <SYMBOL>.csv per stock')
    p.add_argument('--lookback', type=int, nargs='+', default=[1, 5, 10])
    p.add_argument('--notional', type=float, default=10_000.0, help='dollars per position')
    p.add_argument('--half-spread', type=float, default=0.005)
    p.add_argument('--out-dir', default=str(HERE / 'results'))
    args = p.parse_args()

    bars = load_universe(args.data_dir)
    ret, px_in, px_out = slot_table(bars)
    syms = list(ret.columns)
    bars_list = [bars[s] for s in syms]
    entry_ts = to_ns(ret.index)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tag = f'universe{len(syms)}_{ret.index[0].date()}_{ret.index[-1].date()}'
    lines = [
        f'Same-half-hour momentum, {len(syms)} large US stocks, {ret.index[0].date()} .. {ret.index[-1].date()} '
        f'({len(ret)} sessions)',
        f'Trade 09:30 -> 10:00 ET, ~${args.notional:,.0f} per position, half-spread ${args.half_spread}/share',
        '',
    ]

    # ---- 1. Portfolio test on hftbacktest ----
    port_rows, daily = [], {}
    for lb in args.lookback:
        w = quintile_weights(lagged_signal(ret, lb))
        for name, sign in [('momentum (long past winners)', 1), ('reversal (long past losers)', -1)]:
            qty, ein, eout, eq = run_basket(bars_list, entry_ts, sign * w, SLOT.value, args.notional,
                                            args.half_spread)
            pnl = np.nansum(qty * (eout - ein), axis=1)
            traded = np.abs(w).sum(axis=1) > 0
            engine = np.diff(np.concatenate([[0.0], eq[traded]]))
            if not np.allclose(engine, pnl[traded], atol=1e-6):
                raise AssertionError('basket P&L differs from engine accounting')
            daily[f'{name}, lb{lb}'] = pd.Series(np.where(traded, pnl, np.nan), index=ret.index)
            port_rows.append({'strategy': f'{name}, {lb}-day lookback',
                              **day_stats(daily[f'{name}, lb{lb}'])})
    port = pd.DataFrame(port_rows).set_index('strategy')
    k = int(np.abs(quintile_weights(lagged_signal(ret, 1))).sum(axis=1).max() // 2)
    lines += [f'1) Portfolio: top {k} long / bottom {k} short each day (${2 * k * args.notional:,.0f} gross)',
              port.round(2).to_string(), '']
    pd.DataFrame(daily).round(2).to_csv(out / f'portfolio_daily_{tag}.csv')

    # ---- 2. Cross-sectional regressions ----
    fm = fama_macbeth(ret, 10)
    lines += ['2) Avg cross-sectional slope of today\'s 09:30-10:00 return on the same slot k days ago',
              '   (positive = same-slot momentum, as in the paper)',
              fm.round(4).to_string(), '']

    # ---- 3. Candidates with an out-of-sample check ----
    n = len(ret)
    split = ret.index[n // 2]
    first, second = ret.index < split, ret.index >= split
    lines.append(f'3) Candidates: picked on {ret.index[0].date()} .. {ret.index[first][-1].date()}, '
                 f'tested on {split.date()} .. {ret.index[-1].date()}')

    # Stocks: how well does each stock's own relative slot return persist (lag 1)?
    rel = ret.sub(ret.mean(axis=1), axis=0)
    stock_pnl = np.sign(rel.shift(1)) * rel * args.notional  # long if it beat the average yesterday, else short
    st = pd.DataFrame({
        'first_half_$': stock_pnl[first].sum(),
        'second_half_$': stock_pnl[second].sum(),
        'full_period_$': stock_pnl.sum(),
    }).sort_values('first_half_$', ascending=False)
    top_st = st.head(5)
    lines += ['   Stocks (long if it beat the group in yesterday\'s slot, short if not; vs. group average, no costs)',
              '   Top 5 picked on the first half:', top_st.round(0).to_string(),
              f'   Their second-half total: ${top_st["second_half_$"].sum():,.0f}; '
              f'all {len(syms)} stocks avg per stock: ${st["second_half_$"].mean():,.0f}', '']

    # Pairs: 2-stock version (yesterday's winner long, loser short), with costs.
    rows = []
    for a, b in itertools.combinations(syms, 2):
        pp = pair_pnl(ret, px_in, px_out, a, b, 1, args.notional, args.half_spread)
        rows.append(dict(pair=f'{a}/{b}', first_half_=pp[first].sum(), second_half_=pp[second].sum(),
                         full_period_=pp.sum(), t_full=day_stats(pp)['t_stat']))
    pairs = pd.DataFrame(rows).set_index('pair').rename(columns=lambda c: c.rstrip('_') + '_$' if c != 't_full' else c)
    pairs.sort_values('full_period_$', ascending=False).round(2).to_csv(out / f'pairs_{tag}.csv')
    top_pairs = pairs.sort_values('first_half_$', ascending=False).head(10)
    best_full = pairs.sort_values('full_period_$', ascending=False).head(10)
    lines += [f'   Pairs ({len(pairs)} total), 1-day lookback, with costs',
              '   Top 10 picked on the first half -> how they did in the second half:',
              top_pairs.round(0).to_string(),
              f'   Top-10 second-half total: ${top_pairs["second_half_$"].sum():,.0f} '
              f'(avg ${top_pairs["second_half_$"].mean():,.0f}/pair vs. ${pairs["second_half_$"].mean():,.0f} for all pairs)',
              f'   Share of pairs profitable over the full period: {(pairs["full_period_$"] > 0).mean() * 100:.0f}%',
              f'   Pairs with |t| > 2 over the full period: {(pairs["t_full"].abs() > 2).sum()} '
              f'(~{0.05 * len(pairs):.0f} expected by pure chance)',
              '   Best 10 pairs over the full period (in-sample, i.e. hindsight):',
              best_full.round(2).to_string(), '']

    # Sanity check: vectorized pair P&L == engine for one pair.
    a, b = ('AMZN', 'META') if {'AMZN', 'META'} <= set(syms) else syms[:2]
    sig = np.sign((ret[a] - ret[b]).shift(1)).fillna(0).to_numpy()
    w = np.zeros((n, 2))
    w[:, 0], w[:, 1] = sig, -sig
    qty, ein, eout, _ = run_basket([bars[a], bars[b]], entry_ts, w, SLOT.value, args.notional, args.half_spread)
    engine_pair = np.nansum(qty * (eout - ein))
    vec_pair = pair_pnl(ret, px_in, px_out, a, b, 1, args.notional, args.half_spread).sum()
    lines.append(f'Check: {a}/{b} pair P&L engine ${engine_pair:,.2f} vs vectorized ${vec_pair:,.2f}')
    # The engine rounds prices to its 0.0001 grid; Yahoo prices carry float noise below that.
    if abs(engine_pair - vec_pair) > 0.01 * n:
        raise AssertionError('vectorized pair P&L differs from engine')

    report = '\n'.join(lines)
    (out / f'summary_{tag}.txt').write_text(report + '\n')
    print(report)


if __name__ == '__main__':
    main()
