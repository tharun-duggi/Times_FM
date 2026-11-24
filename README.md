# TimesFM Stock Price Forecast

A Streamlit app that downloads daily closing prices from Yahoo Finance and uses
Google's **TimesFM 2.5** foundation model (`google/timesfm-2.5-200m-pytorch`,
Apache-2.0 weights) to forecast the next 5–60 trading days. It shows a
10–90% uncertainty band and a 30-day backtest of the model's accuracy.

> **Educational tool only. Predictions are not financial advice. Stock prices are
> affected by news and events the model cannot see.**

## Setup

Requires Python 3.10 or newer (tested on 3.11). Runs on CPU; an NVIDIA GPU is
used automatically if PyTorch can see one.

```bash
cd Times_FM
python3.11 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Run

```bash
source .venv/bin/activate
streamlit run app.py
```

The app opens at http://localhost:8501 and has two pages, listed in the
sidebar:

- **Single stock**: enter a ticker (for example `RELIANCE.NS`, `TCS.NS` or
  `AAPL`), pick a history period and horizon, and click **Predict**.
- **Upload document**: upload a PDF that lists stocks, such as a watchlist,
  portfolio or broker statement. The app finds the tickers in it and lets you
  review or edit the list. It then downloads each stock's latest prices and
  shows an overview chart, a table of the predicted numbers, and a tab per
  stock with its forecast chart and accuracy check. Up to 10 stocks at a time.

To try the upload page, click **Use example PDF** in its sidebar, or upload
[examples/sample_watchlist.pdf](examples/sample_watchlist.pdf) (also
available from the **Download example PDF** button).

The first prediction downloads the model weights (about 800 MB) into
`~/.cache/huggingface`. After that the model loads from disk in a few seconds
and stays in memory between predictions.

## What it does

1. **Data** (`src/data.py`): downloads daily OHLCV with yfinance
   (split/dividend adjusted), keeps rows with a Close price, sorts by date, and
   raises a clear error for an unknown ticker or fewer than 100 data points.
2. **Forecast** (`src/model.py`): loads TimesFM 2.5 once (`st.cache_resource`)
   and forecasts from up to the last 1,024 closes. It returns the median
   forecast plus the 10th and 90th percentiles, dated on business days
   (weekends skipped; exchange holidays are not).
3. **Backtest** (`src/backtest.py`): hides the last 30 trading days, forecasts
   them, and reports MAE, MAPE and day-by-day direction accuracy. It also shows
   a naive "price stays flat" baseline for comparison.
4. **Document reading** (`src/document.py`): reads the PDF text with pypdf and
   finds tickers in three ways:
   - a table column headed **Ticker**, **Symbol**, **Scrip** or **Stock code**
   - symbols with an exchange suffix anywhere in the text (`RELIANCE.NS`,
     `500325.BO`, `VOD.L`)
   - labelled symbols (`Ticker: AAPL`) and cashtags (`$MSFT`)

   Words like "NSE" or "IT" in ordinary sentences are ignored. Scanned PDFs
   with no selectable text are rejected with a clear message.
5. **UI** (`app.py` plus `views/`): summary cards, an interactive Plotly chart
   (zoom, hover, 3M/6M/1Y presets), the latest 10 rows of prices, and the
   backtest chart. Market data is cached for 15 minutes, so predictions use
   current prices.

Ticker format: NSE stocks use `.NS` (e.g. `RELIANCE.NS`), BSE stocks use `.BO`,
and US stocks use the plain symbol (`AAPL`). If your PDF lists Indian stocks
without a suffix (`RELIANCE`), set **Symbols without an exchange suffix** to
NSE or BSE on the upload page.

## Project structure

```
app.py                       Streamlit entry point (page navigation)
views/single_stock.py        Page: forecast one ticker
views/document_upload.py     Page: upload a PDF and forecast every stock in it
src/data.py                  Fetching and cleaning stock data
src/model.py                 TimesFM loading and prediction
src/backtest.py              Accuracy metrics
src/document.py              Reading tickers from PDFs
src/ui_common.py             Shared charts, cards and cached loaders
examples/sample_watchlist.pdf    Example document (6 stocks)
examples/make_sample_pdf.py      Regenerates the example PDF
requirements.txt             Pinned dependencies
```

`src/data.py`, `src/model.py`, `src/backtest.py` and `src/document.py` can each
be run on their own as a quick self-test, e.g. `python -m src.document`.

## Notes on versions

- **Model license:** the `timesfm` 3.0.2 package contains both TimesFM 2.5 and
  3.0. This app only loads the 2.5 weights, which are Apache-2.0. The 3.0
  weights have a non-commercial license and are not used.
- **PyTorch 2.2.2** is pinned because it is the last release with wheels for
  Intel Macs. It needs NumPy 1.x, so `numpy==1.26.4` and `pyarrow==24.0.0` (the
  newest pyarrow that works with NumPy 1.x) are pinned too.
- **Context length:** TimesFM pads short inputs, and on PyTorch 2.2.2 that
  padding makes the forecast NaN. The app avoids padding by trimming the
  context to a multiple of 32 points (up to 1,024), so at most 31 of the oldest
  prices are left out.
