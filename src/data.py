from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.utils import RAW_DATA_PATH, ensure_directories

REQUIRED_COLUMNS = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]


def _validate_ohlcv_columns(df: pd.DataFrame) -> None:
    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Downloaded data is missing required columns: {missing}")


def load_cached_raw_data(path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """Load cached NVDA OHLCV data from disk."""
    if not path.exists():
        raise FileNotFoundError(f"Raw data cache not found at: {path}")

    data = pd.read_csv(path, parse_dates=["Date"], index_col="Date")
    data = data.sort_index()
    _validate_ohlcv_columns(data)
    return data


def download_nvda_data(
    symbol: str = "NVDA",
    start: str = "2000-01-01",
    end: str | None = None,
    force_download: bool = False,
    cache_path: Path = RAW_DATA_PATH,
) -> pd.DataFrame:
    """Download NVDA daily OHLCV data from Yahoo Finance with local caching."""
    ensure_directories()

    if cache_path.exists() and not force_download:
        return load_cached_raw_data(cache_path)

    try:
        import yfinance as yf
    except ImportError as exc:
        raise RuntimeError(
            "yfinance is not installed. Run `pip install -r requirements.txt` first."
        ) from exc

    downloaded = yf.download(
        tickers=symbol,
        start=start,
        end=end,
        interval="1d",
        auto_adjust=False,
        progress=False,
    )

    if downloaded.empty:
        raise RuntimeError(
            "Yahoo Finance returned an empty dataset for NVDA. "
            "Check your internet connection or date range."
        )

    if isinstance(downloaded.columns, pd.MultiIndex):
        downloaded.columns = downloaded.columns.get_level_values(0)

    _validate_ohlcv_columns(downloaded)

    data = downloaded[REQUIRED_COLUMNS].copy()
    data.index = pd.to_datetime(data.index).tz_localize(None)
    data = data.sort_index()

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(cache_path, index_label="Date")
    return data


def main() -> None:
    """CLI entrypoint for downloading data."""
    dataset = download_nvda_data()
    print(
        f"Saved {len(dataset):,} rows to {RAW_DATA_PATH}. "
        f"Range: {dataset.index.min().date()} -> {dataset.index.max().date()}"
    )


if __name__ == "__main__":
    main()
