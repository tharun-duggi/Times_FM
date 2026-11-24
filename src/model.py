"""TimesFM 2.5 (PyTorch) loading and forecasting.

Uses only the Apache-2.0 TimesFM 2.5 checkpoint (google/timesfm-2.5-200m-pytorch).
"""

from __future__ import annotations

import threading

import numpy as np
import pandas as pd

MODEL_ID = "google/timesfm-2.5-200m-pytorch"
MAX_CONTEXT = 1024
MAX_HORIZON = 256  # compile limit; must be a multiple of the 128-step output patch
PATCH_LEN = 32  # TimesFM 2.5 input patch length

# Index layout of TimesFM 2.5 quantile output: 0 = mean, 1..9 = q10..q90.
Q10_INDEX = 1
Q90_INDEX = 9


class TimesFMForecaster:
    """Thin wrapper around a loaded TimesFM 2.5 model.

    Context length note: when a series is shorter than the compiled context,
    TimesFM left-pads it and masks the padding. With the library's fused
    attention kernel this produces NaN forecasts on some PyTorch versions
    (reproduced on torch 2.2.2, the newest build for Intel Macs). To stay
    correct everywhere we never pad: the context is trimmed to the largest
    multiple of the 32-step patch (at most 1024) and the model is compiled
    for exactly that length. This drops at most 31 of the oldest points.
    """

    def __init__(self, model) -> None:
        self._model = model
        self._compiled_context: int | None = None
        # The model object is shared across Streamlit sessions via
        # st.cache_resource, and compile() mutates it, so serialise access.
        self._lock = threading.Lock()

    @staticmethod
    def context_length(n_points: int) -> int:
        length = min(n_points, MAX_CONTEXT) // PATCH_LEN * PATCH_LEN
        if length < PATCH_LEN:
            raise ValueError(f"Need at least {PATCH_LEN} data points to forecast, got {n_points}.")
        return length

    def _ensure_compiled(self, context: int) -> None:
        if self._compiled_context == context:
            return
        import timesfm

        self._model.compile(
            timesfm.ForecastConfig(
                max_context=context,
                max_horizon=MAX_HORIZON,
                normalize_inputs=True,
                use_continuous_quantile_head=True,
                force_flip_invariance=True,
                infer_is_positive=True,
                fix_quantile_crossing=True,
            )
        )
        self._compiled_context = context

    def predict(self, values: np.ndarray, horizon: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Forecast `horizon` steps after `values`.

        Returns (point, p10, p90) arrays of length `horizon`. The point
        forecast is the model's median (q50).
        """
        values = np.asarray(values, dtype=np.float64)
        if not 1 <= horizon <= MAX_HORIZON:
            raise ValueError(f"Horizon must be between 1 and {MAX_HORIZON}, got {horizon}.")
        if np.isnan(values).any():
            raise ValueError("Input series contains missing values.")

        context = self.context_length(len(values))
        with self._lock:
            self._ensure_compiled(context)
            point, quantiles = self._model.forecast(horizon=horizon, inputs=[values[-context:]])

        point = point[0]
        p10 = quantiles[0, :, Q10_INDEX]
        p90 = quantiles[0, :, Q90_INDEX]
        if not (np.isfinite(point).all() and np.isfinite(p10).all() and np.isfinite(p90).all()):
            raise RuntimeError("TimesFM returned non-finite values for this series.")
        return point, p10, p90


def load_model() -> TimesFMForecaster:
    """Load TimesFM 2.5 200M (PyTorch). Downloads ~800 MB on first use."""
    import timesfm

    # torch.compile (Inductor) needs a working C++ toolchain and gives little
    # benefit for one series on CPU; plain eager mode forecasts in < 1 second.
    model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(MODEL_ID, torch_compile=False)
    return TimesFMForecaster(model)


def future_business_days(last_date: pd.Timestamp, periods: int) -> pd.DatetimeIndex:
    """The next `periods` weekdays after `last_date` (exchange holidays not excluded)."""
    start = pd.Timestamp(last_date).normalize() + pd.offsets.BDay(1)
    return pd.bdate_range(start=start, periods=periods, name="Date")


def forecast_series(forecaster: TimesFMForecaster, close: pd.Series, horizon: int) -> pd.DataFrame:
    """Forecast the next `horizon` business days of a date-indexed close series.

    Returns a frame indexed by future dates with columns: forecast, p10, p90.
    """
    point, p10, p90 = forecaster.predict(close.to_numpy(), horizon)
    dates = future_business_days(close.index[-1], horizon)
    return pd.DataFrame({"forecast": point, "p10": p10, "p90": p90}, index=dates)


if __name__ == "__main__":
    import time

    from src.data import fetch_stock_data

    t0 = time.time()
    fc = load_model()
    print(f"model loaded in {time.time() - t0:.1f}s")

    for symbol, period in (("RELIANCE.NS", "2y"), ("TCS.NS", "1y"), ("AAPL", "5y")):
        close = fetch_stock_data(symbol, period).close
        t0 = time.time()
        out = forecast_series(fc, close, 30)
        ctx = fc.context_length(len(close))
        ok_band = bool((out.p10 <= out.forecast).all() and (out.forecast <= out.p90).all())
        weekdays_only = bool((out.index.dayofweek < 5).all())
        print(
            f"{symbol} {period}: n={len(close)} ctx={ctx} {time.time() - t0:.2f}s | "
            f"last={close.iloc[-1]:.2f} -> day30={out.forecast.iloc[-1]:.2f} "
            f"[{out.p10.iloc[-1]:.2f}, {out.p90.iloc[-1]:.2f}] | "
            f"p10<=pt<=p90={ok_band} weekdays_only={weekdays_only} "
            f"first={out.index[0].date()} ({out.index[0].day_name()}) last_hist={close.index[-1].date()}"
        )

    # Edge cases: minimum data size and horizon bounds.
    short = close.iloc[-100:]
    out = forecast_series(fc, short, 60)
    print(f"100-point series, horizon 60: ctx={fc.context_length(100)} finite={np.isfinite(out.to_numpy()).all()}")
    out = forecast_series(fc, short, 5)
    print(f"horizon 5: rows={len(out)}")
