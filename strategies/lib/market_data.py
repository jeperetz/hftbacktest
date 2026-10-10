"""
Loading intraday OHLCV bars for US equities.

Bars are returned as a pandas DataFrame indexed by the bar *start* time (tz-aware, America/New_York)
with columns: open, high, low, close, volume, end (bar end time, tz-aware).

Sources:
  * Yahoo Finance via ``yfinance`` (free; 1h bars go back ~2 years, 5m/2m bars ~60 days, 1m bars ~7 days).
  * A local CSV you exported yourself (e.g. from a broker or a paid data vendor).
"""

from pathlib import Path

import pandas as pd

NY = 'America/New_York'
REGULAR_OPEN = pd.Timedelta(hours=9, minutes=30)
REGULAR_CLOSE = pd.Timedelta(hours=16)

_INTERVALS = {
    '1m': pd.Timedelta(minutes=1),
    '2m': pd.Timedelta(minutes=2),
    '5m': pd.Timedelta(minutes=5),
    '15m': pd.Timedelta(minutes=15),
    '30m': pd.Timedelta(minutes=30),
    '1h': pd.Timedelta(hours=1),
}


def interval_to_timedelta(interval: str) -> pd.Timedelta:
    try:
        return _INTERVALS[interval]
    except KeyError:
        raise ValueError(f'Unsupported interval {interval!r}; use one of {list(_INTERVALS)}')


def _finalize(df: pd.DataFrame, interval: str) -> pd.DataFrame:
    df = df[['open', 'high', 'low', 'close', 'volume']].astype(float).dropna(subset=['open', 'close'])
    df = df[~df.index.duplicated(keep='first')].sort_index()
    if df.index.tz is None:
        df.index = df.index.tz_localize(NY)
    else:
        df.index = df.index.tz_convert(NY)
    df.index.name = 'start'

    # Regular trading hours only.
    tod = df.index - df.index.normalize()
    df = df[(tod >= REGULAR_OPEN) & (tod < REGULAR_CLOSE)].copy()

    # A bar ends at start + interval, but never after the session close (Yahoo's last hourly bar is 15:30-16:00).
    step = interval_to_timedelta(interval)
    session_close = df.index.normalize() + REGULAR_CLOSE
    ends = df.index + step
    df['end'] = ends.where(ends <= session_close, session_close)
    return df


def fetch_yahoo(symbol: str, start: str, end: str, interval: str = '1h') -> pd.DataFrame:
    """
    Downloads regular-session bars from Yahoo Finance. ``end`` is exclusive.
    Prices are *not* dividend-adjusted, which is what you want for intraday fills.
    """
    import yfinance as yf

    raw = yf.download(
        symbol,
        start=start,
        end=end,
        interval=interval,
        prepost=False,
        auto_adjust=False,
        actions=False,
        progress=False,
        multi_level_index=False,
    )
    if raw is None or raw.empty:
        raise RuntimeError(
            f'Yahoo Finance returned no data for {symbol} ({interval}, {start}..{end}). '
            f'Check network access to query1/query2.finance.yahoo.com and fc.yahoo.com, '
            f'and that the interval is within Yahoo\'s look-back limit.'
        )
    raw = raw.rename(columns=str.lower)
    return _finalize(raw, interval)


def save_bars(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.drop(columns=['end']).to_csv(path, date_format='%Y-%m-%dT%H:%M:%S%z')


def load_csv(path: str | Path, interval: str) -> pd.DataFrame:
    """
    Loads bars from a CSV with a timestamp column (first column, or one named timestamp/datetime/date/start/time)
    and open/high/low/close[/volume] columns. Naive timestamps are assumed to be America/New_York.
    Each timestamp must be the bar's *start* time.
    """
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    ts_col = next((c for c in ('start', 'timestamp', 'datetime', 'date', 'time') if c in df.columns), df.columns[0])
    idx = pd.to_datetime(df[ts_col], utc=False)
    if getattr(idx.dt, 'tz', None) is None:
        idx = idx.dt.tz_localize(NY)
    df.index = pd.DatetimeIndex(idx).tz_convert(NY)
    if 'volume' not in df.columns:
        df['volume'] = 0.0
    return _finalize(df, interval)
