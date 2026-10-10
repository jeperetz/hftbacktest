"""
Sanity checks for the open-hour pair backtest on hand-made bars with known prices.

    python strategies/amzn_meta_open_hour/test_backtest.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from backtest import run_backtest, session_opens  # noqa: E402
from lib.market_data import NY, _finalize  # noqa: E402


def make_bars(day_prices):
    """day_prices: {date: [(open, close) for 09:30, 10:30, 11:30 hourly bars]}"""
    rows = []
    for day, bars in day_prices.items():
        for k, (o, c) in enumerate(bars):
            ts = pd.Timestamp(f'{day} 09:30', tz=NY) + pd.Timedelta(hours=k)
            rows.append((ts, o, max(o, c), min(o, c), c, 1000.0))
    df = pd.DataFrame(rows, columns=['ts', 'open', 'high', 'low', 'close', 'volume']).set_index('ts')
    return _finalize(df, '1h')


def test_fills_and_pnl():
    # Spans the US DST change (2026-11-01) to check the 09:30 ET open is handled in UTC.
    amzn = make_bars({
        '2026-10-29': [(200.00, 202.00), (202.10, 203.00), (203.0, 203.5)],
        '2026-10-30': [(210.00, 205.00), (204.90, 204.00), (204.0, 204.0)],
        '2026-11-02': [(220.00, 221.00), (221.50, 222.00), (222.0, 222.0)],
        '2026-11-03': [(230.00, 231.00), (231.50, 232.00), (232.0, 232.0)],
    })
    meta = make_bars({
        '2026-10-29': [(700.00, 690.00), (689.00, 688.00), (688.0, 688.0)],
        '2026-10-30': [(650.00, 660.00), (661.00, 662.00), (662.0, 662.0)],
        '2026-11-02': [(600.00, 600.00), (600.00, 600.00), (600.0, 600.0)],
        '2026-11-03': [(610.00, 611.00), (611.50, 612.00), (612.0, 612.0)],
    })
    # META has no 09:30 bar on 11-03 -> that day must be skipped.
    meta = meta.drop(pd.Timestamp('2026-11-03 09:30', tz=NY))

    opens = session_opens(amzn, meta)
    assert [str(d.date()) for d in opens] == ['2026-10-29', '2026-10-30', '2026-11-02']

    hs, fee, notional = 0.005, 0.0001, 10_000.0
    res = run_backtest(amzn, meta, opens, 3600 * 10**9, notional, hs, fee, 1_000_000)
    assert len(res) == 3

    # Day 1: buy AMZN at 200.005 (open + half spread), sell at 202.095 (10:30 bar open - half spread).
    #        short META at 699.995, cover at 689.005.
    r = res.iloc[0]
    assert r.long_qty == np.floor(notional / 200.005) == 49
    assert r.short_qty == np.floor(notional / 699.995) == 14
    np.testing.assert_allclose([r.long_entry, r.long_exit, r.short_entry, r.short_exit],
                               [200.005, 202.095, 699.995, 689.005], atol=1e-9)
    # Day 2: AMZN falls, META rises -> losing day.
    r = res.iloc[1]
    np.testing.assert_allclose([r.long_entry, r.long_exit, r.short_entry, r.short_exit],
                               [210.005, 204.895, 649.995, 661.005], atol=1e-9)

    # Engine accounting (balance - fees) matches P&L computed from fills.
    pnl = ((res.long_exit - res.long_entry) * res.long_qty
           + (res.short_entry - res.short_exit) * res.short_qty)
    traded = ((res.long_entry + res.long_exit) * res.long_qty
              + (res.short_entry + res.short_exit) * res.short_qty)
    np.testing.assert_allclose(res.equity_after, np.cumsum(pnl - traded * fee), atol=1e-6)
    np.testing.assert_allclose(res.fees_after, np.cumsum(traded * fee), atol=1e-6)

    # Entries happen at 09:30 ET on both sides of the DST change (13:30 UTC, then 14:30 UTC).
    utc = [pd.Timestamp(int(ts), tz='UTC') for ts in res.entry_ts]
    assert [ts.strftime('%H:%M') for ts in utc] == ['13:30', '13:30', '14:30']
    # Day 3 (after DST): AMZN 220.005 -> 221.495, META flat 599.995 -> 600.005.
    r = res.iloc[2]
    np.testing.assert_allclose([r.long_entry, r.long_exit, r.short_entry, r.short_exit],
                               [220.005, 221.495, 599.995, 600.005], atol=1e-9)
    print('ok')


if __name__ == '__main__':
    test_fills_and_pnl()
