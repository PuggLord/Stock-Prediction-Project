from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.data import download_nvda_data
from src.utils import PROCESSED_DATA_PATH, ensure_directories

FEATURE_COLUMNS = [
    "log_ret_1d",
    "ret_3d_back",
    "ret_5d_back",
    "ret_10d_back",
    "sma_9",
    "sma_21",
    "ema_9",
    "ema_21",
    "ema_crossover",
    "rsi_14",
    "volatility_14",
    "volatility_21",
    "volume_mean_14",
    "volume_z_14",
    "macd",
    "macd_signal",
]


def get_feature_columns() -> list[str]:
    """Return the ordered list of model feature columns."""
    return FEATURE_COLUMNS.copy()


def compute_rsi(close: pd.Series, window: int = 14) -> pd.Series:
    """Compute Relative Strength Index (RSI)."""
    delta = close.diff()
    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)

    avg_gain = gains.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    avg_loss = losses.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()

    relative_strength = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + relative_strength))
    return rsi


def build_feature_dataset(
    raw_data: pd.DataFrame | None = None,
    save_path: Path = PROCESSED_DATA_PATH,
) -> pd.DataFrame:
    """Build leakage-safe technical features and the 3-day forward target."""
    ensure_directories()

    if raw_data is None:
        raw_data = download_nvda_data()

    data = raw_data.copy().sort_index()

    if "Close" not in data.columns or "Volume" not in data.columns:
        raise ValueError("Input raw data must contain 'Close' and 'Volume' columns.")

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

    data["fwd_ret_3d"] = close.shift(-3) / close - 1.0
    data["y_3d"] = (data["fwd_ret_3d"] > 0).astype(int)

    # Used for the daily signal backtest: signal at t applied to return t->t+1.
    data["next_day_return"] = close.shift(-1) / close - 1.0

    required_columns = FEATURE_COLUMNS + ["y_3d", "fwd_ret_3d", "next_day_return"]
    feature_data = data.dropna(subset=required_columns).copy()
    feature_data["y_3d"] = feature_data["y_3d"].astype(int)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    feature_data.to_csv(save_path, index_label="Date")

    return feature_data


def load_feature_data(path: Path = PROCESSED_DATA_PATH) -> pd.DataFrame:
    """Load processed feature dataset; build it if missing."""
    if path.exists():
        return pd.read_csv(path, parse_dates=["Date"], index_col="Date").sort_index()

    built = build_feature_dataset()
    return built


def main() -> None:
    """CLI entrypoint for feature generation."""
    dataset = build_feature_dataset()
    print(
        f"Saved feature dataset with {len(dataset):,} rows and "
        f"{len(FEATURE_COLUMNS)} features to {PROCESSED_DATA_PATH}."
    )


if __name__ == "__main__":
    main()
