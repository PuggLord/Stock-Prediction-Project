from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from src.features import get_feature_columns, load_feature_data
from src.split import time_train_test_split
from src.train import load_trained_models
from src.utils import FIGURES_DIR, SEED, ensure_directories, set_global_seed

CONFUSION_PLOT_PATH = FIGURES_DIR / "confusion_matrices.png"
ROC_PLOT_PATH = FIGURES_DIR / "roc_curve.png"
RF_IMPORTANCE_PLOT_PATH = FIGURES_DIR / "rf_feature_importance.png"
PERM_IMPORTANCE_PLOT_PATH = FIGURES_DIR / "rf_permutation_importance.png"
LR_COEF_PLOT_PATH = FIGURES_DIR / "lr_coefficients.png"


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
) -> dict[str, Any]:
    """Compute core binary classification metrics."""
    metrics: dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }

    if len(np.unique(y_true)) > 1:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba))
    else:
        metrics["roc_auc"] = float("nan")

    return metrics


def simulate_random_classification_baseline(
    y_true: np.ndarray,
    n_runs: int = 200,
    seed: int = SEED,
) -> dict[str, float]:
    """Random 50/50 classifier baseline (Monte Carlo)."""
    rng = np.random.default_rng(seed)
    accuracies = []

    for _ in range(n_runs):
        random_pred = rng.integers(0, 2, size=len(y_true))
        accuracies.append(float(accuracy_score(y_true, random_pred)))

    return {
        "n_runs": n_runs,
        "accuracy_mean": float(np.mean(accuracies)),
        "accuracy_std": float(np.std(accuracies, ddof=0)),
    }


def _plot_confusion_matrices(cm_lr: np.ndarray, cm_rf: np.ndarray, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)

    for axis, matrix, title in [
        (axes[0], cm_lr, "Logistic Regression"),
        (axes[1], cm_rf, "Random Forest"),
    ]:
        image = axis.imshow(matrix, cmap="Blues")
        axis.set_title(f"Confusion Matrix ({title})")
        axis.set_xlabel("Predicted")
        axis.set_ylabel("Actual")
        axis.set_xticks([0, 1])
        axis.set_yticks([0, 1])
        for row in range(2):
            for col in range(2):
                axis.text(col, row, int(matrix[row, col]), ha="center", va="center")

    fig.colorbar(image, ax=axes.ravel().tolist(), shrink=0.8)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def _plot_roc_curves(
    y_true: np.ndarray,
    lr_proba: np.ndarray,
    rf_proba: np.ndarray,
    path: Path,
) -> None:
    fig, axis = plt.subplots(figsize=(7, 5))

    if len(np.unique(y_true)) > 1:
        lr_fpr, lr_tpr, _ = roc_curve(y_true, lr_proba)
        rf_fpr, rf_tpr, _ = roc_curve(y_true, rf_proba)
        axis.plot(lr_fpr, lr_tpr, label="Logistic Regression", linewidth=2)
        axis.plot(rf_fpr, rf_tpr, label="Random Forest", linewidth=2)

    axis.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random")
    axis.set_title("ROC Curve (Test Set)")
    axis.set_xlabel("False Positive Rate")
    axis.set_ylabel("True Positive Rate")
    axis.legend()
    axis.grid(alpha=0.3)

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def _plot_bar_series(
    series: pd.Series,
    path: Path,
    title: str,
    top_n: int = 15,
    signed: bool = False,
) -> None:
    if signed:
        selected = series.reindex(series.abs().sort_values(ascending=False).head(top_n).index)
        selected = selected.sort_values()
        colors = ["#2ca02c" if value > 0 else "#d62728" for value in selected.values]
    else:
        selected = series.sort_values(ascending=False).head(top_n).sort_values()
        colors = "#1f77b4"

    fig, axis = plt.subplots(figsize=(9, 6))
    axis.barh(selected.index, selected.values, color=colors)
    axis.set_title(title)
    axis.set_xlabel("Importance" if not signed else "Coefficient")
    axis.grid(axis="x", alpha=0.3)

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def _serialize_series(series: pd.Series, top_n: int = 10) -> list[dict[str, float | str]]:
    selected = series.head(top_n)
    return [
        {"feature": str(feature), "value": float(value)}
        for feature, value in selected.items()
    ]


def evaluate_models(
    feature_data: pd.DataFrame | None = None,
    train_output: dict[str, Any] | None = None,
    seed: int = SEED,
    n_random_runs: int = 200,
) -> dict[str, Any]:
    """Evaluate trained models with predictive metrics and feature analyses."""
    ensure_directories()
    set_global_seed(seed)

    data = feature_data if feature_data is not None else load_feature_data()

    if train_output is not None:
        split = train_output["split"]
        lr_model = train_output["lr_model"]
        rf_model = train_output["rf_model"]
    else:
        split = time_train_test_split(
            data=data,
            feature_columns=get_feature_columns(),
            target_column="y_3d",
            train_ratio=0.8,
        )
        lr_model, rf_model = load_trained_models()

    X_test = split.X_test
    y_test = split.y_test.astype(int)
    y_true = y_test.to_numpy()

    lr_pred = lr_model.predict(X_test).astype(int)
    lr_proba = lr_model.predict_proba(X_test)[:, 1]

    rf_pred = rf_model.predict(X_test).astype(int)
    rf_proba = rf_model.predict_proba(X_test)[:, 1]

    lr_metrics = compute_classification_metrics(y_true, lr_pred, lr_proba)
    rf_metrics = compute_classification_metrics(y_true, rf_pred, rf_proba)
    random_baseline = simulate_random_classification_baseline(
        y_true,
        n_runs=n_random_runs,
        seed=seed,
    )

    cm_lr = np.array(lr_metrics["confusion_matrix"])
    cm_rf = np.array(rf_metrics["confusion_matrix"])
    _plot_confusion_matrices(cm_lr, cm_rf, CONFUSION_PLOT_PATH)
    _plot_roc_curves(y_true, lr_proba, rf_proba, ROC_PLOT_PATH)

    feature_names = split.feature_columns
    rf_importance = pd.Series(rf_model.feature_importances_, index=feature_names)

    permutation = permutation_importance(
        rf_model,
        X_test,
        y_test,
        n_repeats=20,
        random_state=seed,
        scoring="f1",
        n_jobs=-1,
    )
    permutation_importance_series = pd.Series(permutation.importances_mean, index=feature_names)

    lr_coefficients = pd.Series(
        lr_model.named_steps["clf"].coef_[0],
        index=feature_names,
    )

    _plot_bar_series(
        rf_importance,
        RF_IMPORTANCE_PLOT_PATH,
        title="Random Forest Feature Importance",
        top_n=15,
        signed=False,
    )
    _plot_bar_series(
        permutation_importance_series,
        PERM_IMPORTANCE_PLOT_PATH,
        title="Random Forest Permutation Importance (F1)",
        top_n=15,
        signed=False,
    )
    _plot_bar_series(
        lr_coefficients,
        LR_COEF_PLOT_PATH,
        title="Logistic Regression Coefficients (Scaled Features)",
        top_n=15,
        signed=True,
    )

    predictions = pd.DataFrame(index=X_test.index)
    predictions["y_true"] = y_test
    predictions["lr_pred"] = lr_pred
    predictions["lr_proba"] = lr_proba
    predictions["rf_pred"] = rf_pred
    predictions["rf_proba"] = rf_proba

    if "next_day_return" in split.test_df.columns:
        predictions["next_day_return"] = split.test_df["next_day_return"]
    if "Close" in split.test_df.columns:
        predictions["Close"] = split.test_df["Close"]

    feature_importance = {
        "rf_importance_top10": _serialize_series(rf_importance.sort_values(ascending=False), top_n=10),
        "rf_permutation_top10": _serialize_series(
            permutation_importance_series.sort_values(ascending=False),
            top_n=10,
        ),
        "lr_top_positive": _serialize_series(lr_coefficients.sort_values(ascending=False), top_n=10),
        "lr_top_negative": _serialize_series(lr_coefficients.sort_values(ascending=True), top_n=10),
    }

    return {
        "classification": {
            "logistic_regression": lr_metrics,
            "random_forest": rf_metrics,
            "random_baseline": random_baseline,
        },
        "feature_importance": feature_importance,
        "predictions": predictions,
        "plot_paths": {
            "confusion_matrices": str(CONFUSION_PLOT_PATH),
            "roc_curve": str(ROC_PLOT_PATH),
            "rf_feature_importance": str(RF_IMPORTANCE_PLOT_PATH),
            "rf_permutation_importance": str(PERM_IMPORTANCE_PLOT_PATH),
            "lr_coefficients": str(LR_COEF_PLOT_PATH),
        },
    }


def main() -> None:
    """CLI entrypoint for model evaluation."""
    evaluation = evaluate_models()
    classification = evaluation["classification"]
    lr_f1 = classification["logistic_regression"]["f1"]
    rf_f1 = classification["random_forest"]["f1"]
    print(f"Evaluation complete. LR F1={lr_f1:.3f}, RF F1={rf_f1:.3f}.")
    print(f"Saved plots to {FIGURES_DIR}")


if __name__ == "__main__":
    main()
