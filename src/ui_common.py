"""Streamlit pieces shared by the app's pages: cached loaders, charts and cards."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.backtest import TEST_DAYS, BacktestResult, run_backtest
from src.data import StockData, fetch_stock_data
from src.model import TimesFMForecaster, forecast_series, load_model

DISCLAIMER = (
    "Educational tool only. Predictions are not financial advice. "
    "Stock prices are affected by news and events the model cannot see."
)
CURRENCY_SYMBOLS = {"INR": "₹", "USD": "$", "EUR": "€", "GBP": "£", "JPY": "¥"}

# Categorical slots 1 (blue) and 2 (orange), validated for colour-blind
# separation and contrast in both light and dark mode.
PALETTE = {
    "light": {"history": "#2a78d6", "forecast": "#eb6834", "band": "rgba(235,104,52,0.18)", "rule": "#8a8984"},
    "dark": {"history": "#3987e5", "forecast": "#d95926", "band": "rgba(217,89,38,0.16)", "rule": "#9a9992"},
}
PLOTLY_CONFIG = {"scrollZoom": True, "displaylogo": False}
DATE_INDEX = {"_index": st.column_config.DateColumn("Date", format="YYYY-MM-DD")}


# --------------------------------------------------------------------------- cached loaders
@st.cache_resource(show_spinner="Loading TimesFM 2.5 model (first run downloads ~800 MB)...")
def get_forecaster() -> TimesFMForecaster:
    return load_model()


@st.cache_data(ttl=900, show_spinner=False)  # refresh market data every 15 minutes
def get_stock_data(ticker: str, period: str) -> StockData:
    return fetch_stock_data(ticker, period)


@dataclass
class Prediction:
    data: StockData
    forecast: pd.DataFrame  # index: future dates; columns: forecast, p10, p90
    backtest: BacktestResult
    horizon: int

    @property
    def last_price(self) -> float:
        return float(self.data.close.iloc[-1])

    @property
    def end(self) -> pd.Series:
        return self.forecast.iloc[-1]

    def pct(self, value: float) -> float:
        return (value / self.last_price - 1) * 100


def predict_ticker(ticker: str, period: str, horizon: int) -> Prediction:
    """Fetch the latest prices, forecast `horizon` days ahead and run the backtest.

    Raises DataError for unknown tickers or too little data.
    """
    data = get_stock_data(ticker, period)
    forecaster = get_forecaster()
    forecast = forecast_series(forecaster, data.close, horizon)
    backtest = run_backtest(forecaster, data.close, TEST_DAYS)
    return Prediction(data=data, forecast=forecast, backtest=backtest, horizon=horizon)


# --------------------------------------------------------------------------- formatting
def colors() -> dict:
    try:
        mode = "dark" if st.context.theme.type == "dark" else "light"
    except Exception:
        mode = "light"
    return PALETTE[mode]


def money(value: float, currency: str | None) -> str:
    symbol = CURRENCY_SYMBOLS.get(currency or "")
    if symbol:
        return f"{symbol}{value:,.2f}"
    return f"{value:,.2f} {currency}" if currency else f"{value:,.2f}"


def compact_range(low: float, high: float, currency: str | None) -> str:
    """Short 'low–high' label that fits a summary card, e.g. ₹1,168–1,422."""
    decimals = 0 if high >= 100 else 2
    symbol = CURRENCY_SYMBOLS.get(currency or "", "")
    return f"{symbol}{low:,.{decimals}f}–{high:,.{decimals}f}"


# --------------------------------------------------------------------------- charts
def add_marker_line(fig: go.Figure, x: pd.Timestamp, label: str, color: str) -> None:
    # Shape + annotation instead of add_vline: add_vline's annotation breaks on date axes.
    fig.add_shape(type="line", x0=x, x1=x, y0=0, y1=1, yref="paper", line=dict(color=color, width=1.5, dash="dot"))
    fig.add_annotation(
        x=x, y=1, yref="paper", text=label, showarrow=False, yanchor="bottom", font=dict(size=12, color=color)
    )


def style_figure(fig: go.Figure, currency: str | None, height: int = 480) -> go.Figure:
    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=40, b=10),
        hovermode="x unified",
        dragmode="zoom",
        # "normal": without it Plotly reverses the legend when fill traces are present.
        legend=dict(orientation="h", yanchor="bottom", y=1.04, xanchor="left", x=0, traceorder="normal"),
        yaxis_title=f"Price ({currency})" if currency else "Price",
    )
    fig.update_xaxes(hoverformat="%a %d %b %Y")
    return fig


def forecast_figure(close: pd.Series, forecast: pd.DataFrame, currency: str | None) -> go.Figure:
    c = colors()
    last_date, last_price = close.index[-1], float(close.iloc[-1])
    # Start the forecast and band at the last real close so the lines join up.
    fx = [last_date, *forecast.index]
    fy = [last_price, *forecast["forecast"]]
    lo = [last_price, *forecast["p10"]]
    hi = [last_price, *forecast["p90"]]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=close.index, y=close, name="Historical close", mode="lines", legendrank=1,
            line=dict(color=c["history"], width=2),
            hovertemplate="Close: %{y:,.2f}<extra></extra>",
        )
    )
    fig.add_trace(go.Scatter(x=fx, y=hi, mode="lines", line=dict(width=0), hoverinfo="skip", showlegend=False))
    fig.add_trace(
        go.Scatter(
            x=fx, y=lo, name="10–90% range", mode="lines", line=dict(width=0), legendrank=3,
            fill="tonexty", fillcolor=c["band"], hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=fx, y=fy, name="Predicted", mode="lines", legendrank=2,
            line=dict(color=c["forecast"], width=2, dash="dash"),
            customdata=list(zip(lo, hi)),
            hovertemplate="Predicted: %{y:,.2f}<br>10–90%: %{customdata[0]:,.2f} – %{customdata[1]:,.2f}<extra></extra>",
        )
    )
    add_marker_line(fig, last_date, "Today", c["rule"])
    # Zoom presets sit below the axis so they never collide with the "Today" label.
    fig.update_xaxes(
        rangeselector=dict(
            buttons=[
                dict(count=3, label="3M", step="month", stepmode="backward"),
                dict(count=6, label="6M", step="month", stepmode="backward"),
                dict(count=1, label="1Y", step="year", stepmode="backward"),
                dict(step="all", label="All"),
            ],
            x=0, xanchor="left", y=-0.1, yanchor="top",
        )
    )
    style_figure(fig, currency, height=520)
    fig.update_layout(margin=dict(b=50))
    return fig


def backtest_figure(close: pd.Series, bt: BacktestResult, currency: str | None, lead_in: int = 60) -> go.Figure:
    c = colors()
    frame = bt.frame
    start = frame.index[0]
    before = close.loc[:start].iloc[:-1].iloc[-lead_in:]
    actual = pd.concat([before, frame["actual"]])
    anchor = before.index[-1]
    px = [anchor, *frame.index]
    py = [bt.last_known, *frame["predicted"]]
    lo = [bt.last_known, *frame["p10"]]
    hi = [bt.last_known, *frame["p90"]]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=px, y=hi, mode="lines", line=dict(width=0), hoverinfo="skip", showlegend=False))
    fig.add_trace(
        go.Scatter(
            x=px, y=lo, name="10–90% range", mode="lines", line=dict(width=0), legendrank=3,
            fill="tonexty", fillcolor=c["band"], hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=actual.index, y=actual, name="Actual", mode="lines", legendrank=1,
            line=dict(color=c["history"], width=2),
            hovertemplate="Actual: %{y:,.2f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=px, y=py, name="Predicted", mode="lines", legendrank=2,
            line=dict(color=c["forecast"], width=2, dash="dash"),
            hovertemplate="Predicted: %{y:,.2f}<extra></extra>",
        )
    )
    add_marker_line(fig, anchor, "Prediction made here", c["rule"])
    return style_figure(fig, currency, height=420)


# --------------------------------------------------------------------------- page sections
def render_summary(p: Prediction) -> None:
    cur = p.data.currency
    end = p.end
    change = float(end["forecast"]) - p.last_price

    card = dict(border=True, height="stretch")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(
        "Current price", money(p.last_price, cur), **card,
        help=f"Last close on {p.data.close.index[-1]:%d %b %Y}. If the market is open, this may be an intraday price.",
    )
    c2.metric(
        f"Predicted in {p.horizon} trading days", money(float(end["forecast"]), cur),
        delta=f"{change:+,.2f}", **card,
        help=f"Median (50th percentile) forecast for {p.forecast.index[-1]:%d %b %Y}.",
    )
    c3.metric("Expected change", f"{p.pct(float(end['forecast'])):+.2f}%", **card)
    c4.metric(
        "10–90% range", compact_range(float(end["p10"]), float(end["p90"]), cur), **card,
        help=(
            f"The model gives an 80% chance the price on {p.forecast.index[-1]:%d %b %Y} is between "
            f"{money(float(end['p10']), cur)} and {money(float(end['p90']), cur)}."
        ),
    )


def render_forecast(p: Prediction, period: str, key: str) -> None:
    st.subheader(f"{p.data.ticker}: {period} history and {p.horizon}-day forecast")
    st.plotly_chart(forecast_figure(p.data.close, p.forecast, p.data.currency), config=PLOTLY_CONFIG, key=f"{key}_fc")
    st.caption(
        "Future dates skip weekends but not exchange holidays. Drag to zoom, double-click to reset, "
        "and hover for date and price."
    )
    with st.expander("Forecast values"):
        st.dataframe(
            p.forecast.rename(columns={"forecast": "Predicted", "p10": "10th pct", "p90": "90th pct"}),
            column_config=DATE_INDEX,
        )


def render_latest_prices(data: StockData) -> None:
    st.dataframe(
        data.ohlcv.tail(10),
        column_config={
            **DATE_INDEX,
            **{col: st.column_config.NumberColumn(col, format="%.2f") for col in ("Open", "High", "Low", "Close")},
            "Volume": st.column_config.NumberColumn("Volume", format="localized"),
        },
    )


def render_backtest(p: Prediction, key: str) -> None:
    cur, bt = p.data.currency, p.backtest
    st.subheader(f"Accuracy check: last {len(bt.frame)} trading days")
    st.caption(
        f"The model was given prices only up to {bt.frame.index[0] - pd.Timedelta(days=1):%d %b %Y}, "
        f"asked to predict the next {len(bt.frame)} trading days, and compared with what actually happened."
    )
    c1, c2, c3 = st.columns(3)
    c1.metric("MAE", money(bt.mae, cur), border=True, help="Mean absolute error: average size of the miss per day.")
    c2.metric("MAPE", f"{bt.mape:.2f}%", border=True, help="Mean absolute percentage error.")
    c3.metric(
        "Direction accuracy", f"{bt.direction_accuracy:.0f}%", border=True,
        help="Share of days where the predicted day-to-day move (up or down) matched the actual move. 50% is coin-flip level.",
    )

    inside = ((bt.frame["actual"] >= bt.frame["p10"]) & (bt.frame["actual"] <= bt.frame["p90"])).sum()
    vs_naive = "better" if bt.mae < bt.naive_mae else "worse"
    st.caption(
        f"Overall {len(bt.frame)}-day direction: **{'correct' if bt.trend_correct else 'wrong'}**. "
        f"Actual price stayed inside the 10–90% band on **{inside} of {len(bt.frame)}** days. "
        f"A naive \"price stays flat\" guess has MAE {money(bt.naive_mae, cur)}, so TimesFM was **{vs_naive}** "
        f"than that baseline on this window."
    )
    st.plotly_chart(backtest_figure(p.data.close, bt, cur), config=PLOTLY_CONFIG, key=f"{key}_bt")
    with st.expander("Backtest values"):
        st.dataframe(
            bt.frame.rename(columns=str.capitalize).rename(columns={"P10": "10th pct", "P90": "90th pct"}),
            column_config=DATE_INDEX,
        )
