from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from pandas.tseries.offsets import BDay

from src.data import download_nvda_data
from src.features import compute_rsi, get_feature_columns
from src.train import LR_MODEL_PATH
from src.utils import ensure_directories, write_json


def _engineer_live_features(raw_data: pd.DataFrame) -> pd.DataFrame:
    """Create live-safe feature frame without forward-looking columns."""
    data = raw_data.copy().sort_index()

    if "Close" not in data.columns or "Volume" not in data.columns:
        raise ValueError("Raw data must include 'Close' and 'Volume' columns.")

    close = data["Close"]
    volume = data["Volume"]

    data["log_ret_1d"] = np.log(close / close.shift(1))
    data["ret_3d_back"] = close.pct_change(3)
    data["ret_5d_back"] = close.pct_change(5)
    data["ret_10d_back"] = close.pct_change(10)

    data["sma_9"] = close.rolling(9).mean()
    data["sma_21"] = close.rolling(21).mean()
    data["ema_9"] = close.ewm(span=9, adjust=False).mean()
    data["ema_21"] = close.ewm(span=21, adjust=False).mean()
    data["ema_crossover"] = (data["ema_9"] > data["ema_21"]).astype(int)

    data["rsi_14"] = compute_rsi(close, window=14)
    data["volatility_14"] = data["log_ret_1d"].rolling(14).std()
    data["volatility_21"] = data["log_ret_1d"].rolling(21).std()

    data["volume_mean_14"] = volume.rolling(14).mean()
    volume_std_14 = volume.rolling(14).std()
    data["volume_z_14"] = (volume - data["volume_mean_14"]) / volume_std_14.replace(0, np.nan)

    ema_12 = close.ewm(span=12, adjust=False).mean()
    ema_26 = close.ewm(span=26, adjust=False).mean()
    data["macd"] = ema_12 - ema_26
    data["macd_signal"] = data["macd"].ewm(span=9, adjust=False).mean()

    feature_columns = get_feature_columns()
    live = data.dropna(subset=feature_columns + ["Close"]).copy()
    return live


def _feature_hash(row: pd.Series, feature_columns: list[str]) -> str:
    payload = "|".join(f"{column}:{float(row[column]):.8f}" for column in feature_columns)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def generate_latest_signal(
    raw_data: pd.DataFrame | None = None,
    ticker: str = "NVDA",
    threshold: float = 0.5,
    size_pct: float = 0.10,
    horizon_days: int = 3,
    as_of: str | None = None,
    force_download: bool = False,
    model_path: Path = LR_MODEL_PATH,
) -> dict[str, Any]:
    """Generate a compact, bot-friendly signal payload using the LR model."""
    ensure_directories()

    if not 0.0 < size_pct <= 1.0:
        raise ValueError("size_pct must be in (0, 1].")

    if horizon_days < 1:
        raise ValueError("horizon_days must be at least 1.")

    if not model_path.exists():
        raise FileNotFoundError(
            f"Logistic model not found at {model_path}. Run `python run_all.py` first."
        )

    base_data = raw_data if raw_data is not None else download_nvda_data(
        symbol=ticker,
        force_download=force_download,
    )
    live_features = _engineer_live_features(base_data)

    if live_features.empty:
        raise RuntimeError("Live feature frame is empty after rolling-window NaN drops.")

    selected_row: pd.Series
    if as_of is None:
        selected_row = live_features.iloc[-1]
    else:
        as_of_ts = pd.Timestamp(as_of)
        subset = live_features.loc[live_features.index <= as_of_ts]
        if subset.empty:
            raise ValueError(f"No feature row exists at or before as_of={as_of}.")
        selected_row = subset.iloc[-1]

    feature_columns = get_feature_columns()
    x_row = selected_row[feature_columns].to_frame().T

    lr_model = joblib.load(model_path)
    probability_up = float(lr_model.predict_proba(x_row)[0, 1])
    signal = int(probability_up >= threshold)

    as_of_ts = pd.Timestamp(selected_row.name)
    expires_ts = as_of_ts + BDay(horizon_days)

    payload = {
        "ticker": ticker,
        "asof": as_of_ts.date().isoformat(),
        "model": "logistic_regression",
        "horizon_days": int(horizon_days),
        "p_up_3d": round(probability_up, 6),
        "signal": int(signal),
        "threshold": float(threshold),
        "size_pct": float(size_pct),
        "price": round(float(selected_row["Close"]), 4),
        "expires": expires_ts.date().isoformat(),
        "feature_hash": _feature_hash(selected_row, feature_columns),
        "generated_utc": pd.Timestamp.utcnow().isoformat(),
    }

    return payload


def save_signal_payload(payload: dict[str, Any], output_path: Path) -> None:
    """Persist signal payload as JSON."""
    write_json(payload, output_path)


def main() -> None:
    """CLI entrypoint for ad hoc signal generation."""
    payload = generate_latest_signal()
    print(payload)


if __name__ == "__main__":
    main()
