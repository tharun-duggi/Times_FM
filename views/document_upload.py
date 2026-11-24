"""Page: upload a PDF that lists stocks, then forecast each one from the latest prices."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.data import PERIODS, DataError
from src.document import DocumentError, DocumentTickers, apply_default_suffix, extract_tickers_from_pdf
from src.ui_common import (
    DISCLAIMER,
    PLOTLY_CONFIG,
    Prediction,
    colors,
    money,
    predict_ticker,
    render_backtest,
    render_forecast,
    render_latest_prices,
    render_summary,
)

EXAMPLE_PDF = Path(__file__).resolve().parent.parent / "examples" / "sample_watchlist.pdf"
MAX_STOCKS = 10
SUFFIX_OPTIONS = {
    "Use as written (e.g. US stocks)": "",
    "Add .NS (NSE India)": ".NS",
    "Add .BO (BSE India)": ".BO",
}


@st.cache_data(show_spinner=False, max_entries=20)
def read_document(data: bytes) -> DocumentTickers:
    return extract_tickers_from_pdf(data)


def parse_tickers(text: str, suffix: str) -> list[str]:
    tokens = [t.strip().upper() for t in re.split(r"[,;\s]+", text or "") if t.strip()]
    return apply_default_suffix(tokens, suffix)


def overview_figure(results: list[Prediction], horizon: int) -> go.Figure:
    """Dot = median expected % change, line = 10–90% range, one row per stock."""
    c = colors()
    mid = [p.pct(float(p.end["forecast"])) for p in results]
    lo = [p.pct(float(p.end["p10"])) for p in results]
    hi = [p.pct(float(p.end["p90"])) for p in results]
    custom = [
        (money(p.last_price, p.data.currency), money(float(p.end["forecast"]), p.data.currency), l, h)
        for p, l, h in zip(results, lo, hi)
    ]
    fig = go.Figure(
        go.Scatter(
            x=mid, y=[p.data.ticker for p in results], mode="markers",
            marker=dict(color=c["forecast"], size=12),
            error_x=dict(
                type="data", symmetric=False, array=[h - m for h, m in zip(hi, mid)],
                arrayminus=[m - l for m, l in zip(mid, lo)], color=c["forecast"], thickness=2, width=6,
            ),
            customdata=custom,
            hovertemplate=(
                "<b>%{y}</b><br>Expected change: %{x:+.2f}%<br>"
                "10–90% range: %{customdata[2]:+.2f}% to %{customdata[3]:+.2f}%<br>"
                "Now %{customdata[0]} → predicted %{customdata[1]}<extra></extra>"
            ),
        )
    )
    fig.add_vline(x=0, line=dict(color=c["rule"], width=1))
    fig.update_yaxes(type="category", autorange="reversed")  # same order as the document
    fig.update_xaxes(title=f"Expected change after {horizon} trading days", ticksuffix="%", zeroline=False)
    fig.update_layout(
        height=max(240, 80 + 46 * len(results)), margin=dict(l=10, r=10, t=10, b=10),
        showlegend=False, hovermode="closest",
    )
    return fig


def overview_table(results: list[Prediction]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Ticker": p.data.ticker,
                "Price date": p.data.close.index[-1],
                "Currency": p.data.currency or "",
                "Current price": p.last_price,
                "Predicted price": float(p.end["forecast"]),
                "Expected change": p.pct(float(p.end["forecast"])),
                "10th pct": float(p.end["p10"]),
                "90th pct": float(p.end["p90"]),
                "Backtest MAPE": p.backtest.mape,
                "Direction accuracy": p.backtest.direction_accuracy,
            }
            for p in results
        ]
    )


# --------------------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Document")
    uploaded = st.file_uploader(
        "Upload a PDF", type=["pdf"], key="doc_upload",
        help="A watchlist, portfolio statement or research note that lists stock tickers.",
    )
    if st.button("Use example PDF", width="stretch"):
        st.session_state["doc_use_example"] = True
    st.download_button(
        "Download example PDF", data=EXAMPLE_PDF.read_bytes(), file_name=EXAMPLE_PDF.name,
        mime="application/pdf", width="stretch",
    )
    st.header("Settings")
    suffix_label = st.selectbox(
        "Symbols without an exchange suffix", list(SUFFIX_OPTIONS), key="doc_suffix",
        help="Indian broker statements often say RELIANCE rather than RELIANCE.NS. Pick NSE or BSE to add the suffix.",
    )
    period = st.selectbox("History period", PERIODS, index=PERIODS.index("2y"), key="doc_period")
    horizon = st.slider("Forecast horizon (trading days)", min_value=5, max_value=60, value=30, key="doc_horizon")
    st.caption("Model: TimesFM 2.5 200M (PyTorch), Apache-2.0 weights.")

# --------------------------------------------------------------------------- page
st.title("📄 Forecast stocks from a document")
st.warning(DISCLAIMER, icon="⚠️")

if uploaded is not None:
    source_name, source_bytes = uploaded.name, uploaded.getvalue()
    st.session_state["doc_use_example"] = False
elif st.session_state.get("doc_use_example"):
    source_name, source_bytes = EXAMPLE_PDF.name, EXAMPLE_PDF.read_bytes()
else:
    st.info(
        "**Upload a PDF** in the sidebar, or click **Use example PDF** to try it. The app finds the stock "
        "tickers in the document, downloads their latest daily prices and forecasts each one."
    )
    st.markdown(
        "**What PDFs work**\n"
        "- A table with a **Ticker** or **Symbol** column (watchlists, portfolio or broker statements)\n"
        "- Tickers with an exchange suffix anywhere in the text: `RELIANCE.NS`, `500325.BO`, `VOD.L`\n"
        "- Labelled tickers such as `Ticker: AAPL`, or cashtags such as `$MSFT`\n"
        "- The PDF needs selectable text; scanned images cannot be read"
    )
    st.stop()

digest = hashlib.sha1(source_bytes).hexdigest()[:12]
try:
    doc = read_document(source_bytes)
except DocumentError as exc:
    st.error(str(exc))
    st.stop()

suffix = SUFFIX_OPTIONS[suffix_label]
found = apply_default_suffix(doc.tickers, suffix)

st.subheader("1. Review the stocks found")
pages = f"{doc.pages} page{'s' if doc.pages != 1 else ''}"
if found:
    st.success(f"Read **{source_name}** ({pages}) and found **{len(found)}** ticker{'s' if len(found) != 1 else ''}.")
else:
    st.warning(f"Read **{source_name}** ({pages}) but found no tickers. Add them below.")
if len(found) > MAX_STOCKS:
    st.info(f"The first {MAX_STOCKS} are selected. Up to {MAX_STOCKS} stocks can be forecast at once.")

selected = st.multiselect(
    "Stocks to forecast", options=found, default=found[:MAX_STOCKS], max_selections=MAX_STOCKS,
    key=f"doc_selected_{digest}{suffix}",
)
extra_text = st.text_input(
    "Add tickers that are not in the document (optional, comma-separated)", key=f"doc_extra_{digest}",
    placeholder="e.g. SBIN.NS, GOOGL",
)
tickers = selected + [t for t in parse_tickers(extra_text, suffix) if t not in selected]
if len(tickers) > MAX_STOCKS:
    st.warning(f"Only the first {MAX_STOCKS} tickers will be forecast.")
    tickers = tickers[:MAX_STOCKS]
with st.expander("Text read from the document"):
    st.text(doc.text[:8000])

inputs = (digest, tuple(tickers), period, horizon)
label = f"Predict {len(tickers)} stock{'s' if len(tickers) != 1 else ''}"
if st.button(label, type="primary", disabled=not tickers):
    results: list[Prediction] = []
    errors: list[tuple[str, str]] = []
    bar = st.progress(0.0)
    for i, ticker in enumerate(tickers):
        bar.progress(i / len(tickers), text=f"Fetching prices and forecasting {ticker} ({i + 1}/{len(tickers)})...")
        try:
            results.append(predict_ticker(ticker, period, horizon))
        except DataError as exc:
            errors.append((ticker, str(exc)))
        except Exception as exc:
            errors.append((ticker, f"Forecast failed: {exc}"))
    bar.empty()
    st.session_state["doc_result"] = {"inputs": inputs, "source": source_name, "results": results, "errors": errors}

res = st.session_state.get("doc_result")
if not res or res["inputs"][0] != digest:
    st.stop()

st.subheader("2. Predictions")
if res["inputs"] != inputs:
    st.info("Settings changed. Click **Predict** to update the results below.")
for ticker, message in res["errors"]:
    st.error(f"**{ticker}** skipped: {message}")
results = res["results"]
if not results:
    st.stop()

r_period, r_horizon = res["inputs"][2], res["inputs"][3]
st.caption(
    f"Stocks from **{res['source']}**, forecast from their latest daily prices on Yahoo Finance "
    f"({r_period} of history, {r_horizon} trading days ahead)."
)

st.markdown(f"**Expected change after {r_horizon} trading days**")
st.caption("Dot: median forecast. Line: 10–90% range. Left of the grey line means a predicted fall.")
st.plotly_chart(overview_figure(results, r_horizon), config=PLOTLY_CONFIG, key="doc_overview")

st.dataframe(
    overview_table(results),
    hide_index=True,
    column_config={
        "Price date": st.column_config.DateColumn("Price date", format="YYYY-MM-DD"),
        **{
            col: st.column_config.NumberColumn(col, format="%.2f")
            for col in ("Current price", "Predicted price", "10th pct", "90th pct")
        },
        "Expected change": st.column_config.NumberColumn("Expected change", format="%+.2f%%"),
        "Backtest MAPE": st.column_config.NumberColumn(
            "Backtest MAPE", format="%.2f%%", help="Average % error when predicting the last 30 known days."
        ),
        "Direction accuracy": st.column_config.NumberColumn(
            "Direction accuracy", format="%.0f%%", help="Up/down days called correctly in the backtest (50% = coin flip)."
        ),
    },
)

st.markdown("**Details for each stock**")
for tab, p in zip(st.tabs([p.data.ticker for p in results]), results):
    with tab:
        key = f"doc_{p.data.ticker}"
        render_summary(p)
        render_forecast(p, r_period, key=key)
        with st.expander("Latest prices"):
            render_latest_prices(p.data)
        render_backtest(p, key=key)
