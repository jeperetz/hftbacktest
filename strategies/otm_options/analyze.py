"""
Did buying out-of-the-money S&P 500 options pay off? ("black swans are under-priced")

Part 1 - real option prices (Cboe benchmark indexes):
  * PPUT: S&P 500 (with dividends) + a ~5% OTM SPX put bought every month, held to expiry.
  * VXTH: S&P 500 (with dividends) + 1-month 30-delta VIX calls (a tail hedge).
  Comparing each against the S&P 500 total return isolates what the options added or cost.

Part 2 - a wider grid priced at ATM implied volatility:
  Systematically buy SPX puts/calls at a fixed % out of the money with 1-week, 1-month or 3-month expiries,
  hold to expiry, roll. Payoffs come from actual SPX closes; premiums from Black-Scholes using the matching
  Cboe ATM implied-vol index (VIX9D / VIX / VIX3M) and the 13-week T-bill rate.

  Real SPX puts trade at *higher* implied vol than ATM (the skew), real OTM calls usually at *lower* vol.
  So for puts this pricing is a lower bound on cost: a put strategy that loses here lost in reality, one that
  wins here is inconclusive. For calls it is (typically) an upper bound: a call strategy that wins here won.

Usage:
    python strategies/otm_options/analyze.py --start 2024-10-09 --end 2026-10-09
"""

import argparse
from math import erf, exp, log, sqrt
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = HERE / 'data'
DIV_YIELD = 0.013  # S&P 500 dividend yield, approx.


def load_cboe(name, col=None):
    df = pd.read_csv(DATA / f'{name}.csv')
    df['DATE'] = pd.to_datetime(df['DATE'], format='%m/%d/%Y')
    s = df.set_index('DATE')[col or ('CLOSE' if 'CLOSE' in df.columns else name)].astype(float)
    return s[s > 0].sort_index()


def load_yahoo(name):
    df = pd.read_csv(DATA / f'{name}.csv', index_col=0)
    df.index = pd.to_datetime(df.index)
    return df['Close'].astype(float).dropna().sort_index()


def _ncdf(x):
    return 0.5 * (1 + erf(x / sqrt(2)))


def bs_price(is_call, s, k, t, r, q, vol):
    if t <= 0 or vol <= 0:
        return max(s - k, 0.0) if is_call else max(k - s, 0.0)
    d1 = (log(s / k) + (r - q + 0.5 * vol * vol) * t) / (vol * sqrt(t))
    d2 = d1 - vol * sqrt(t)
    if is_call:
        return s * exp(-q * t) * _ncdf(d1) - k * exp(-r * t) * _ncdf(d2)
    return k * exp(-r * t) * _ncdf(-d2) - s * exp(-q * t) * _ncdf(-d1)


def perf(series):
    """CAGR, total return, max drawdown, worst/best day of a price/index level series."""
    s = series.dropna()
    yrs = (s.index[-1] - s.index[0]).days / 365.25
    ret = s.pct_change().dropna()
    dd = (s / s.cummax() - 1).min()
    return dict(total_ret=s.iloc[-1] / s.iloc[0] - 1, cagr=(s.iloc[-1] / s.iloc[0]) ** (1 / yrs) - 1,
                vol=ret.std() * np.sqrt(252), max_dd=dd)


def part1(start, end):
    sptr = load_yahoo('SP500TR')
    rows = []
    for name, label in [('PPUT', 'PPUT: S&P 500 + monthly ~5% OTM puts'),
                        ('VXTH', 'VXTH: S&P 500 + VIX call tail hedge')]:
        idx = load_cboe(name)
        common = idx.index.intersection(sptr.index)
        a, b = idx.loc[common], sptr.loc[common]
        for lo, hi, period in [(start, end, 'selected period'), (a.index[0], end, f'since {a.index[0].year}')]:
            x, y = a[lo:hi], b[lo:hi]
            px, py = perf(x), perf(y)
            rows.append({'strategy': label, 'period': f'{period} ({x.index[0].date()}..{x.index[-1].date()})',
                         'strategy_cagr': px['cagr'], 'sp500_tr_cagr': py['cagr'],
                         'option_effect_per_yr': px['cagr'] - py['cagr'],
                         'strategy_max_dd': px['max_dd'], 'sp500_max_dd': py['max_dd']})
    return pd.DataFrame(rows)


def overlay_episodes(start, end):
    """How the PPUT puts behaved through the worst S&P 500 drawdown in the window."""
    sptr = load_yahoo('SP500TR')[start:end]
    pput = load_cboe('PPUT')[start:end]
    common = sptr.index.intersection(pput.index)
    sptr, pput = sptr.loc[common], pput.loc[common]
    dd = sptr / sptr.cummax() - 1
    trough = dd.idxmin()
    peak = sptr[:trough].idxmax()
    return dict(peak=peak.date(), trough=trough.date(), sp500_dd=sptr[trough] / sptr[peak] - 1,
                pput_dd=pput[trough] / pput[peak] - 1)


def breakeven_vol_add(g, is_call):
    """
    Vol points that could be added to every option's implied vol before total premiums exceed total payoffs.
    Positive = the strategy still broke even when priced that much above ATM vol; NaN = never paid off.
    """
    got = g['payoff'].sum()
    if got <= 0:
        return np.nan

    def paid(add):
        return sum(bs_price(is_call, r.spx0, r.strike, r.t, r.rate, DIV_YIELD, max(r.vol + add, 1e-4))
                   for r in g.itertuples())

    lo, hi = -g['vol'].min() + 1e-4, 3.0
    if paid(lo) > got:
        return lo * 100
    if paid(hi) < got:
        return hi * 100
    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if paid(mid) < got else (lo, mid)
    return (lo + hi) / 2 * 100


def roll_dates(index, start, end, tenor_days):
    """Non-overlapping rolls: each option is bought at a close and expires at the last close <= +tenor days."""
    idx = index[(index >= start) & (index <= end)]
    out = []
    i = 0
    while i < len(idx):
        t0 = idx[i]
        target = t0 + pd.Timedelta(days=tenor_days)
        j = idx.searchsorted(target, side='right') - 1
        if j <= i or idx[-1] < target - pd.Timedelta(days=3):
            break
        out.append((t0, idx[j]))
        i = j
    return out


def part2(start, end, put_otm, call_otm, tenors):
    spx = load_cboe('SPX')
    irx = load_yahoo('IRX') / 100
    vols = {'VIX9D': load_cboe('VIX9D'), 'VIX': load_cboe('VIX'), 'VIX3M': load_cboe('VIX3M')}
    rows, trades = [], []
    for tenor_name, days, vol_name in tenors:
        vol = vols[vol_name]
        common = spx.index.intersection(vol.index)
        for t0, t1 in roll_dates(common, pd.Timestamp(start), pd.Timestamp(end), days):
            s0, s1 = spx[t0], spx[t1]
            sigma = vol[t0] / 100
            r = float(irx[:t0].iloc[-1]) if len(irx[:t0]) else 0.04
            t = (t1 - t0).days / 365.0
            for kind, levels in [('put', put_otm), ('call', call_otm)]:
                for m in levels:
                    k = s0 * (1 - m) if kind == 'put' else s0 * (1 + m)
                    prem = bs_price(kind == 'call', s0, k, t, r, DIV_YIELD, sigma)
                    payoff = max(k - s1, 0.0) if kind == 'put' else max(s1 - k, 0.0)
                    trades.append(dict(tenor=tenor_name, kind=kind, otm=m, start=t0, expiry=t1, spx0=s0, spx1=s1,
                                       strike=k, t=t, rate=r, vol=sigma, premium=prem, payoff=payoff))
    tr = pd.DataFrame(trades)
    for (tenor, kind, otm), g in tr.groupby(['tenor', 'kind', 'otm'], sort=False):
        paid = g['premium'].sum()
        got = g['payoff'].sum()
        best = g.loc[g['payoff'].idxmax()]
        rows.append({'tenor': tenor, 'type': kind, 'otm': f'{otm:.0%}', 'rolls': len(g),
                     'rolls_paid_off': int((g['payoff'] > 0).sum()),
                     'payout_per_$1': got / paid if paid > 0 else np.nan,
                     'breakeven_vol_add': breakeven_vol_add(g, kind == 'call'),
                     'premium_pct_spx_per_yr': g['premium'].div(g['spx0']).sum() / (
                         (g['expiry'].max() - g['start'].min()).days / 365.25),
                     'best_roll': f"{best['start'].date()} x{best['payoff'] / best['premium']:.0f}" if got > 0 else '-'})
    return pd.DataFrame(rows), tr


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--start', default='2024-10-09')
    p.add_argument('--end', default='2026-10-09')
    p.add_argument('--put-otm', type=float, nargs='+', default=[0.02, 0.05, 0.10, 0.15, 0.20, 0.30])
    p.add_argument('--call-otm', type=float, nargs='+', default=[0.02, 0.05, 0.10, 0.15, 0.20])
    p.add_argument('--out-dir', default=str(HERE / 'results'))
    args = p.parse_args()
    tenors = [('1 week', 7, 'VIX9D'), ('1 month', 30, 'VIX'), ('3 months', 91, 'VIX3M')]

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    pd.set_option('display.width', 200)

    p1 = part1(args.start, args.end)
    ep = overlay_episodes(args.start, args.end)
    lines = ['PART 1 - real option prices (Cboe indexes vs S&P 500 total return)',
             p1.to_string(index=False, float_format=lambda v: f'{v * 100:.1f}%'),
             f"Worst S&P 500 drawdown in the window: {ep['peak']} -> {ep['trough']}: S&P 500 TR {ep['sp500_dd']:.1%}, "
             f"PPUT {ep['pput_dd']:.1%}", '']

    for label, lo, hi in [('selected period', args.start, args.end), ('long run', '2011-01-01', args.end)]:
        summary, trades = part2(lo, hi, args.put_otm, args.call_otm, tenors)
        fmt = summary.copy()
        fmt['payout_per_$1'] = fmt['payout_per_$1'].map(lambda v: f'${v:.2f}')
        fmt['premium_pct_spx_per_yr'] = fmt['premium_pct_spx_per_yr'].map(lambda v: f'{v * 100:.2f}%')
        fmt['breakeven_vol_add'] = fmt['breakeven_vol_add'].map(lambda v: '-' if np.isnan(v) else f'{v:+.1f} pts')
        lines += [f'PART 2 - systematic OTM option buying priced at ATM implied vol, {label} ({lo}..{hi})',
                  '  payout_per_$1 > $1 means payoffs exceeded premiums paid (before commissions/spreads)',
                  '  breakeven_vol_add: how many vol points above ATM implied vol the options could have been bought at',
                  '  and still broken even (compare with the real skew: OTM SPX puts trade well above ATM vol)',
                  fmt.to_string(index=False), '']
        trades.to_csv(out / f'trades_{lo}_{hi}.csv', index=False)

    report = '\n'.join(lines)
    (out / f'summary_{args.start}_{args.end}.txt').write_text(report + '\n')
    print(report)


if __name__ == '__main__':
    main()
