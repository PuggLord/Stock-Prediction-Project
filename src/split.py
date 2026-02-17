from __future__ import annotations

from dataclasses import dataclass
from typing import Generator

import pandas as pd

from src.features import get_feature_columns


@dataclass
class TimeSplit:
    """Container for time-ordered train/test split outputs."""

    feature_columns: list[str]
    target_column: str
    train_df: pd.DataFrame
    test_df: pd.DataFrame
    X_train: pd.DataFrame
    y_train: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series


def time_train_test_split(
    data: pd.DataFrame,
    feature_columns: list[str] | None = None,
    target_column: str = "y_3d",
    train_ratio: float = 0.8,
) -> TimeSplit:
    """Perform a leakage-safe chronological split (earliest train, latest test)."""
    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be between 0 and 1.")

    ordered = data.sort_index().copy()
    selected_features = feature_columns or get_feature_columns()

    missing = [
        column
        for column in selected_features + [target_column]
        if column not in ordered.columns
    ]
    if missing:
        raise ValueError(f"Missing required columns for split: {missing}")

    split_index = int(len(ordered) * train_ratio)
    if split_index <= 0 or split_index >= len(ordered):
        raise ValueError("Split index is invalid; dataset may be too small.")

    train_df = ordered.iloc[:split_index].copy()
    test_df = ordered.iloc[split_index:].copy()

    X_train = train_df[selected_features].copy()
    y_train = train_df[target_column].copy()
    X_test = test_df[selected_features].copy()
    y_test = test_df[target_column].copy()

    return TimeSplit(
        feature_columns=selected_features,
        target_column=target_column,
        train_df=train_df,
        test_df=test_df,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
    )


def walk_forward_splits(
    data: pd.DataFrame,
    n_splits: int = 5,
    min_train_size: int = 252,
    test_size: int = 63,
) -> Generator[tuple[pd.DataFrame, pd.DataFrame], None, None]:
    """Optional expanding-window walk-forward splits."""
    ordered = data.sort_index()

    if len(ordered) < min_train_size + test_size:
        raise ValueError("Dataset is too small for the requested walk-forward parameters.")

    if n_splits < 1:
        raise ValueError("n_splits must be at least 1.")

    max_start = len(ordered) - test_size
    step = max(1, (max_start - min_train_size) // max(1, n_splits - 1))

    train_end = min_train_size
    for _ in range(n_splits):
        test_end = min(train_end + test_size, len(ordered))
        if test_end - train_end < test_size:
            break

        train_slice = ordered.iloc[:train_end].copy()
        test_slice = ordered.iloc[train_end:test_end].copy()
        yield train_slice, test_slice

        train_end += step
        if train_end >= max_start:
            break
