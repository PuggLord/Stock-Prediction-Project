from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.features import get_feature_columns, load_feature_data
from src.split import TimeSplit, time_train_test_split
from src.utils import MODELS_DIR, REPORTS_DIR, SEED, ensure_directories, set_global_seed, write_json

LR_MODEL_PATH = MODELS_DIR / "logistic_regression.joblib"
RF_MODEL_PATH = MODELS_DIR / "random_forest.joblib"
TRAIN_METADATA_PATH = MODELS_DIR / "train_metadata.json"
TEST_PREDICTIONS_PATH = REPORTS_DIR / "test_predictions.csv"


def _build_logistic_pipeline(seed: int) -> Pipeline:
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=seed,
                ),
            ),
        ]
    )


def _build_random_forest(seed: int) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=500,
        max_depth=6,
        min_samples_leaf=5,
        class_weight="balanced_subsample",
        random_state=seed,
        n_jobs=-1,
    )


def train_models(
    feature_data: pd.DataFrame | None = None,
    seed: int = SEED,
) -> dict[str, Any]:
    """Train Logistic Regression and Random Forest models and save artifacts."""
    ensure_directories()
    set_global_seed(seed)

    data = feature_data if feature_data is not None else load_feature_data()
    split: TimeSplit = time_train_test_split(
        data=data,
        feature_columns=get_feature_columns(),
        target_column="y_3d",
        train_ratio=0.8,
    )

    lr_model = _build_logistic_pipeline(seed)
    rf_model = _build_random_forest(seed)

    lr_model.fit(split.X_train, split.y_train)
    rf_model.fit(split.X_train, split.y_train)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(lr_model, LR_MODEL_PATH)
    joblib.dump(rf_model, RF_MODEL_PATH)

    predictions = pd.DataFrame(index=split.X_test.index)
    predictions["y_true"] = split.y_test.astype(int)
    predictions["lr_pred"] = lr_model.predict(split.X_test).astype(int)
    predictions["lr_proba"] = lr_model.predict_proba(split.X_test)[:, 1]
    predictions["rf_pred"] = rf_model.predict(split.X_test).astype(int)
    predictions["rf_proba"] = rf_model.predict_proba(split.X_test)[:, 1]

    for optional_column in ["next_day_return", "Close", "fwd_ret_3d"]:
        if optional_column in split.test_df.columns:
            predictions[optional_column] = split.test_df[optional_column]

    TEST_PREDICTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(TEST_PREDICTIONS_PATH, index_label="Date")

    class_balance = {
        "train_positive_rate": float(split.y_train.mean()),
        "test_positive_rate": float(split.y_test.mean()),
        "train_size": int(len(split.train_df)),
        "test_size": int(len(split.test_df)),
    }

    metadata = {
        "seed": seed,
        "feature_columns": split.feature_columns,
        "target_column": split.target_column,
        "train_start": str(split.train_df.index.min().date()),
        "train_end": str(split.train_df.index.max().date()),
        "test_start": str(split.test_df.index.min().date()),
        "test_end": str(split.test_df.index.max().date()),
        "class_balance": class_balance,
        "lr_model_path": str(LR_MODEL_PATH),
        "rf_model_path": str(RF_MODEL_PATH),
        "test_predictions_path": str(TEST_PREDICTIONS_PATH),
    }
    write_json(metadata, TRAIN_METADATA_PATH)

    return {
        "lr_model": lr_model,
        "rf_model": rf_model,
        "split": split,
        "predictions": predictions,
        "class_balance": class_balance,
        "metadata": metadata,
    }


def load_trained_models() -> tuple[Pipeline, RandomForestClassifier]:
    """Load trained model artifacts from disk."""
    if not LR_MODEL_PATH.exists() or not RF_MODEL_PATH.exists():
        raise FileNotFoundError(
            "Model artifacts were not found. Run `python run_all.py` or `python -m src.train` first."
        )

    lr_model = joblib.load(LR_MODEL_PATH)
    rf_model = joblib.load(RF_MODEL_PATH)
    return lr_model, rf_model


def main() -> None:
    """CLI entrypoint for training models."""
    output = train_models()
    metadata = output["metadata"]
    print(
        "Training complete. "
        f"Train range: {metadata['train_start']} -> {metadata['train_end']}. "
        f"Test range: {metadata['test_start']} -> {metadata['test_end']}."
    )
    print(f"Saved models to: {Path(metadata['lr_model_path']).parent}")


if __name__ == "__main__":
    main()
