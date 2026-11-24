"""Page: forecast one ticker typed into the sidebar."""

from __future__ import annotations

import streamlit as st

from src.data import MIN_POINTS, PERIODS, DataError
from src.ui_common import (
    DISCLAIMER,
    Prediction,
    predict_ticker,
    render_backtest,
    render_forecast,
    render_latest_prices,
    render_summary,
)

with st.sidebar:
    st.header("Settings")
    ticker = st.text_input("Ticker symbol", value="RELIANCE.NS", help="Yahoo Finance symbol, e.g. RELIANCE.NS, TCS.NS, AAPL.")
    period = st.selectbox("History period", PERIODS, index=PERIODS.index("2y"))
    horizon = st.slider("Forecast horizon (trading days)", min_value=5, max_value=60, value=30)
    predict = st.button("Predict", type="primary", width="stretch")
    st.caption("Model: TimesFM 2.5 200M (PyTorch), Apache-2.0 weights.")

st.title("📈 Stock price forecast with TimesFM")
st.warning(DISCLAIMER, icon="⚠️")

inputs = (ticker.strip().upper(), period, horizon)
if predict:
    try:
        with st.spinner("Fetching prices and forecasting..."):
            st.session_state["result"] = {"inputs": inputs, "prediction": predict_ticker(*inputs)}
    except DataError as exc:
        st.session_state.pop("result", None)
        st.error(str(exc))
        st.stop()
    except Exception as exc:
        st.session_state.pop("result", None)
        st.error(f"Forecasting failed: {exc}")
        st.stop()

result = st.session_state.get("result")
if result is None:
    st.info(
        f"Choose a ticker, history period and horizon in the sidebar, then click **Predict**. "
        f"At least {MIN_POINTS} daily prices are needed."
    )
    st.stop()

p: Prediction = result["prediction"]
if inputs != result["inputs"]:
    st.info("Settings changed. Click **Predict** to update the results below.")

render_summary(p)
render_forecast(p, period=result["inputs"][1], key="single")
st.subheader("Latest prices")
render_latest_prices(p.data)
render_backtest(p, key="single")
