from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import MinMaxScaler

from .data import TRADING_DAYS


def rolling_volatility(returns: pd.Series, window_size: int) -> pd.Series:
    return (returns.rolling(window=window_size).std() * np.sqrt(TRADING_DAYS)).dropna()


def _ewma_fallback(volatility: pd.Series, window_size: int, horizon: int, reason: str):
    forecast = np.repeat(volatility.ewm(span=window_size).mean().iloc[-1], horizon)
    return volatility, forecast, {"model": f"EWMA fallback ({reason})", "validation_mae": None}


def forecast_volatility(
    returns: pd.Series,
    window_size: int,
    epochs: int,
    horizon: int = 30,
) -> tuple[pd.Series, np.ndarray, dict[str, float | str | None]]:
    """Forecast volatility and report simple holdout validation diagnostics."""
    volatility = rolling_volatility(returns, window_size)
    if len(volatility) <= window_size + 10:
        return _ewma_fallback(volatility, window_size, horizon, "insufficient history")

    try:
        from tensorflow.keras.layers import LSTM, Dense, Dropout
        from tensorflow.keras.models import Sequential
    except Exception:
        return _ewma_fallback(volatility, window_size, horizon, "TensorFlow unavailable")

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(volatility.to_numpy().reshape(-1, 1))
    x_values, y_values = [], []
    for i in range(len(scaled) - window_size):
        x_values.append(scaled[i : i + window_size])
        y_values.append(scaled[i + window_size])
    x_values = np.asarray(x_values)
    y_values = np.asarray(y_values)

    split = max(1, int(len(x_values) * 0.8))
    x_train, y_train = x_values[:split], y_values[:split]
    x_valid, y_valid = x_values[split:], y_values[split:]

    model = Sequential(
        [
            LSTM(48, activation="tanh", return_sequences=True, input_shape=(window_size, 1)),
            Dropout(0.2),
            LSTM(24, activation="tanh"),
            Dense(1),
        ]
    )
    model.compile(optimizer="adam", loss="mse")
    model.fit(x_train, y_train, epochs=epochs, batch_size=32, verbose=0)

    validation_mae = None
    if len(x_valid):
        valid_pred = model.predict(x_valid, verbose=0)
        validation_mae = float(
            mean_absolute_error(
                scaler.inverse_transform(y_valid),
                scaler.inverse_transform(valid_pred),
            )
        )

    last_sequence = scaled[-window_size:].copy()
    preds = []
    for _ in range(horizon):
        pred = model.predict(last_sequence.reshape(1, window_size, 1), verbose=0)
        preds.append(pred[0, 0])
        last_sequence = np.vstack([last_sequence[1:], pred])
    forecast = scaler.inverse_transform(np.asarray(preds).reshape(-1, 1)).flatten()
    return volatility, forecast, {"model": "LSTM", "validation_mae": validation_mae}
