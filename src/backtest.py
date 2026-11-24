"""Hold-out backtest: hide the last N days, forecast them, and score the result."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.model import TimesFMForecaster

TEST_DAYS = 30


@dataclass
class BacktestResult:
    frame: pd.DataFrame  # index: test dates; columns: actual, predicted, p10, p90
    last_known: float  # last price the model saw before the test window
    mae: float
    mape: float  # percent
    direction_accuracy: float  # percent of days where predicted up/down matched actual
    trend_correct: bool  # did the forecast get the overall direction of the window right
    naive_mae: float  # MAE of "price stays at last_known" baseline


def compute_metrics(actual: np.ndarray, predicted: np.ndarray, last_known: float) -> dict:
    """Error and direction metrics for a multi-step forecast.

    Direction is scored day by day: the move from the previous day to this day
    (the first day is measured from `last_known`), for both the actual and the
    predicted path. Days where either path is exactly flat are excluded.
    """
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    if actual.shape != predicted.shape or actual.size == 0:
        raise ValueError("actual and predicted must be non-empty arrays of equal length")

    errors = predicted - actual
    mae = float(np.mean(np.abs(errors)))
    mape = float(np.mean(np.abs(errors) / np.abs(actual)) * 100)

    actual_moves = np.sign(np.diff(np.concatenate([[last_known], actual])))
    predicted_moves = np.sign(np.diff(np.concatenate([[last_known], predicted])))
    scored = (actual_moves != 0) & (predicted_moves != 0)
    direction_accuracy = (
        float(np.mean(actual_moves[scored] == predicted_moves[scored]) * 100) if scored.any() else float("nan")
    )

    trend_correct = bool(np.sign(predicted[-1] - last_known) == np.sign(actual[-1] - last_known))
    naive_mae = float(np.mean(np.abs(actual - last_known)))

    return {
        "mae": mae,
        "mape": mape,
        "direction_accuracy": direction_accuracy,
        "trend_correct": trend_correct,
        "naive_mae": naive_mae,
    }


def run_backtest(forecaster: TimesFMForecaster, close: pd.Series, test_days: int = TEST_DAYS) -> BacktestResult:
    """Hide the last `test_days` closes, forecast them from the rest, and compare."""
    if len(close) <= test_days + 32:
        raise ValueError(f"Need more than {test_days + 32} points to backtest {test_days} days.")

    train, test = close.iloc[:-test_days], close.iloc[-test_days:]
    point, p10, p90 = forecaster.predict(train.to_numpy(), test_days)
    last_known = float(train.iloc[-1])

    frame = pd.DataFrame(
        {"actual": test.to_numpy(), "predicted": point, "p10": p10, "p90": p90},
        index=test.index,
    )
    metrics = compute_metrics(frame["actual"].to_numpy(), point, last_known)
    return BacktestResult(frame=frame, last_known=last_known, **metrics)


if __name__ == "__main__":
    # Metric sanity checks on hand-computed values.
    m = compute_metrics(np.array([101.0, 103.0, 102.0]), np.array([102.0, 101.0, 102.0]), 100.0)
    # errors 1, 2, 0 -> MAE 1.0; actual moves +,+,- ; predicted moves +,-,+ -> 1/3 correct
    assert abs(m["mae"] - 1.0) < 1e-9, m
    assert abs(m["mape"] - (1 / 101 + 2 / 103 + 0) / 3 * 100) < 1e-9, m
    assert abs(m["direction_accuracy"] - 100 / 3) < 1e-9, m
    assert m["trend_correct"] is True, m  # both end above 100
    assert abs(m["naive_mae"] - 2.0) < 1e-9, m
    print("metric checks passed")

    from src.data import fetch_stock_data
    from src.model import load_model

    fc = load_model()
    for symbol in ("RELIANCE.NS", "TCS.NS", "AAPL"):
        close = fetch_stock_data(symbol, "2y").close
        r = run_backtest(fc, close)
        coverage = float(((r.frame.actual >= r.frame.p10) & (r.frame.actual <= r.frame.p90)).mean() * 100)
        print(
            f"{symbol}: test {r.frame.index[0].date()}..{r.frame.index[-1].date()} ({len(r.frame)} days) | "
            f"MAE={r.mae:.2f} MAPE={r.mape:.2f}% dir={r.direction_accuracy:.1f}% trend_ok={r.trend_correct} | "
            f"naive MAE={r.naive_mae:.2f} | in 10-90 band={coverage:.0f}%"
        )
