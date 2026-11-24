"""Fetching and cleaning daily stock prices from Yahoo Finance."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import yfinance as yf

MIN_POINTS = 100
PERIODS = ("1y", "2y", "5y")
OHLCV_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]


class DataError(ValueError):
    """Raised when price data cannot be fetched or is unusable."""


@dataclass
class StockData:
    ticker: str
    ohlcv: pd.DataFrame  # date-indexed, sorted ascending, no missing Close
    currency: str | None

    @property
    def close(self) -> pd.Series:
        return self.ohlcv["Close"]


def fetch_stock_data(ticker: str, period: str = "2y") -> StockData:
    """Download daily OHLCV data and return a cleaned, date-sorted frame.

    Prices are split/dividend adjusted (auto_adjust=True), so the most recent
    Close equals the actual last traded close while older prices have no
    artificial jumps from stock splits.
    """
    ticker = (ticker or "").strip().upper()
    if not ticker:
        raise DataError("Please enter a ticker symbol, e.g. RELIANCE.NS or AAPL.")
    if period not in PERIODS:
        raise DataError(f"Unsupported period '{period}'. Choose one of {', '.join(PERIODS)}.")

    yf_ticker = yf.Ticker(ticker)
    try:
        raw = yf_ticker.history(period=period, interval="1d", auto_adjust=True)
    except Exception as exc:  # yfinance raises a variety of errors for bad symbols
        raise DataError(f"Could not download data for '{ticker}': {exc}") from exc

    if raw is None or raw.empty or "Close" not in raw.columns:
        raise DataError(
            f"No price data found for '{ticker}'. Check the symbol - Indian stocks "
            "need an exchange suffix (RELIANCE.NS for NSE, RELIANCE.BO for BSE)."
        )

    df = raw[[c for c in OHLCV_COLUMNS if c in raw.columns]].copy()
    # Yahoo returns exchange-local timestamps; keep just the calendar date.
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    df.index = pd.DatetimeIndex(df.index).normalize()
    df.index.name = "Date"

    df = df.dropna(subset=["Close"])
    df = df[~df.index.duplicated(keep="last")].sort_index()

    if len(df) < MIN_POINTS:
        raise DataError(
            f"Only {len(df)} daily prices available for '{ticker}' over {period}; "
            f"at least {MIN_POINTS} are needed. Try a longer history period."
        )

    currency = None
    try:
        currency = (yf_ticker.history_metadata or {}).get("currency")
    except Exception:
        pass

    return StockData(ticker=ticker, ohlcv=df, currency=currency)


if __name__ == "__main__":
    for symbol in ("RELIANCE.NS", "TCS.NS", "AAPL"):
        data = fetch_stock_data(symbol, "2y")
        c = data.close
        print(
            f"{symbol}: {len(c)} rows, {c.index[0].date()} -> {c.index[-1].date()}, "
            f"last close {c.iloc[-1]:.2f} {data.currency}, sorted={c.index.is_monotonic_increasing}, "
            f"nans={c.isna().sum()}"
        )
    print(data.ohlcv.tail(3))
    for bad, period in (("NOTAREALTICKER123", "2y"), ("", "2y")):
        try:
            fetch_stock_data(bad, period)
            print(f"ERROR: expected failure for {bad!r}")
        except DataError as e:
            print(f"OK DataError for {bad!r}: {e}")
