from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from src.backtest import backtest_models
from src.data import download_nvda_data
from src.evaluate import evaluate_models
from src.features import build_feature_dataset
from src.train import train_models
from src.utils import REPORTS_DIR, SEED, ensure_directories, format_pct, set_global_seed, write_json

RESULTS_JSON_PATH = REPORTS_DIR / "results.json"
RESULTS_MD_PATH = REPORTS_DIR / "results.md"


def _fmt_metric(value: float | None) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float) and np.isnan(value):
        return "N/A"
    return f"{value:.3f}"


def _fmt_pct_value(value: float | None) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float) and np.isnan(value):
        return "N/A"
    return format_pct(float(value))


def _best_model_by_metric(classification: dict[str, Any], metric: str) -> str:
    candidates = {
        "Logistic Regression": classification["logistic_regression"].get(metric, float("nan")),
        "Random Forest": classification["random_forest"].get(metric, float("nan")),
    }
    return max(candidates, key=lambda key: candidates[key])


def _build_results_markdown(results: dict[str, Any]) -> str:
    classification = results["classification"]
    backtest = results["backtest"]
    importance = results["feature_importance"]
    class_balance = results["class_balance"]

    best_f1_model = _best_model_by_metric(classification, "f1")

    rf_top = importance["rf_importance_top10"][0]["feature"] if importance["rf_importance_top10"] else "N/A"
    perm_top = (
        importance["rf_permutation_top10"][0]["feature"]
        if importance["rf_permutation_top10"]
        else "N/A"
    )
    lr_pos_top = importance["lr_top_positive"][0]["feature"] if importance["lr_top_positive"] else "N/A"
    lr_neg_top = importance["lr_top_negative"][0]["feature"] if importance["lr_top_negative"] else "N/A"

    random_cls = classification["random_baseline"]
    random_bt = backtest["random_baseline"]

    lines = [
        "# NVDA 3-Day Forward Return Prediction Report",
        "",
        "## Brief Method",
        "- Data: NVDA daily OHLCV from Yahoo Finance via yfinance.",
        "- Target: `y_3d = 1` when `Close[t+3]/Close[t]-1 > 0`, else `0`.",
        "- Features: technical indicators/returns/volume features using only current and past data.",
        "- Split: chronological 80% train / 20% test.",
        "- Models: Logistic Regression (with scaling) and Random Forest.",
        "- Trading simulation: daily signal strategy (`position[t]=prediction[t]`, PnL via `next_day_return[t]`).",
        "",
        "## Classification Metrics (Test Set)",
        "| Model | Accuracy | Precision | Recall | F1 | ROC AUC |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
        (
            f"| Logistic Regression | {_fmt_metric(classification['logistic_regression']['accuracy'])}"
            f" | {_fmt_metric(classification['logistic_regression']['precision'])}"
            f" | {_fmt_metric(classification['logistic_regression']['recall'])}"
            f" | {_fmt_metric(classification['logistic_regression']['f1'])}"
            f" | {_fmt_metric(classification['logistic_regression']['roc_auc'])} |"
        ),
        (
            f"| Random Forest | {_fmt_metric(classification['random_forest']['accuracy'])}"
            f" | {_fmt_metric(classification['random_forest']['precision'])}"
            f" | {_fmt_metric(classification['random_forest']['recall'])}"
            f" | {_fmt_metric(classification['random_forest']['f1'])}"
            f" | {_fmt_metric(classification['random_forest']['roc_auc'])} |"
        ),
        (
            f"| Random Baseline (200 runs) | {_fmt_metric(random_cls['accuracy_mean'])} +/- "
            f"{_fmt_metric(random_cls['accuracy_std'])} | - | - | - | - |"
        ),
        "",
        "## Backtest Summary (Test Set)",
        "| Strategy | Cumulative Return | Annualized Return | Annualized Volatility | Sharpe (rf=0) | Max Drawdown |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
        (
            f"| Logistic Regression Strategy | {_fmt_pct_value(backtest['logistic_regression']['cumulative_return'])}"
            f" | {_fmt_pct_value(backtest['logistic_regression']['annualized_return'])}"
            f" | {_fmt_pct_value(backtest['logistic_regression']['annualized_volatility'])}"
            f" | {_fmt_metric(backtest['logistic_regression']['sharpe'])}"
            f" | {_fmt_pct_value(backtest['logistic_regression']['max_drawdown'])} |"
        ),
        (
            f"| Random Forest Strategy | {_fmt_pct_value(backtest['random_forest']['cumulative_return'])}"
            f" | {_fmt_pct_value(backtest['random_forest']['annualized_return'])}"
            f" | {_fmt_pct_value(backtest['random_forest']['annualized_volatility'])}"
            f" | {_fmt_metric(backtest['random_forest']['sharpe'])}"
            f" | {_fmt_pct_value(backtest['random_forest']['max_drawdown'])} |"
        ),
        (
            f"| Buy and Hold | {_fmt_pct_value(backtest['buy_and_hold']['cumulative_return'])}"
            f" | {_fmt_pct_value(backtest['buy_and_hold']['annualized_return'])}"
            f" | {_fmt_pct_value(backtest['buy_and_hold']['annualized_volatility'])}"
            f" | {_fmt_metric(backtest['buy_and_hold']['sharpe'])}"
            f" | {_fmt_pct_value(backtest['buy_and_hold']['max_drawdown'])} |"
        ),
        (
            f"| Random Baseline (200 runs) | {_fmt_pct_value(random_bt['cumulative_return_mean'])} +/- "
            f"{_fmt_pct_value(random_bt['cumulative_return_std'])}"
            f" | {_fmt_pct_value(random_bt['annualized_return_mean'])}"
            f" | {_fmt_pct_value(random_bt['annualized_volatility_mean'])}"
            f" | {_fmt_metric(random_bt['sharpe_mean'])}"
            f" | {_fmt_pct_value(random_bt['max_drawdown_mean'])} |"
        ),
        "",
        "## 3-5 Key Takeaways",
        f"- Best predictive model by F1 on the test set: **{best_f1_model}**.",
        (
            "- Class balance is moderate: "
            f"train positive rate = {class_balance['train_positive_rate']:.3f}, "
            f"test positive rate = {class_balance['test_positive_rate']:.3f}."
        ),
        (
            "- Random baseline accuracy centers around "
            f"{random_cls['accuracy_mean']:.3f} +/- {random_cls['accuracy_std']:.3f}, "
            "which is a useful sanity check for edge over chance."
        ),
        (
            "- Buy-and-hold remains a strong benchmark for return comparison; "
            "strategy value should be judged against both risk and drawdown."
        ),
        (
            "- Feature importance highlights: "
            f"RF impurity top={rf_top}, RF permutation top={perm_top}, "
            f"LR strongest positive={lr_pos_top}, LR strongest negative={lr_neg_top}."
        ),
        "",
        "## Feature Importance Highlights",
        "### Random Forest (impurity) top 5",
        "| Feature | Importance |",
        "| --- | ---: |",
    ]

    for row in importance["rf_importance_top10"][:5]:
        lines.append(f"| {row['feature']} | {row['value']:.4f} |")

    lines.extend(
        [
            "",
            "### Random Forest (permutation) top 5",
            "| Feature | Importance |",
            "| --- | ---: |",
        ]
    )
    for row in importance["rf_permutation_top10"][:5]:
        lines.append(f"| {row['feature']} | {row['value']:.4f} |")

    lines.extend(
        [
            "",
            "### Logistic Regression coefficients (scaled) top positive 5",
            "| Feature | Coefficient |",
            "| --- | ---: |",
        ]
    )
    for row in importance["lr_top_positive"][:5]:
        lines.append(f"| {row['feature']} | {row['value']:.4f} |")

    lines.extend(
        [
            "",
            "### Logistic Regression coefficients (scaled) top negative 5",
            "| Feature | Coefficient |",
            "| --- | ---: |",
        ]
    )
    for row in importance["lr_top_negative"][:5]:
        lines.append(f"| {row['feature']} | {row['value']:.4f} |")

    lines.extend(
        [
            "",
            "## Artifacts",
            "- JSON summary: `reports/results.json`",
            "- Markdown report: `reports/results.md`",
            "- Figures: `reports/figures/`",
            "- Test predictions: `reports/test_predictions.csv`",
        ]
    )

    return "\n".join(lines) + "\n"


def run_pipeline() -> dict[str, Any]:
    """Run the full reproducible pipeline from data download to reports."""
    ensure_directories()
    set_global_seed(SEED)

    raw_data = download_nvda_data()
    feature_data = build_feature_dataset(raw_data)

    train_output = train_models(feature_data=feature_data, seed=SEED)
    evaluation = evaluate_models(
        feature_data=feature_data,
        train_output=train_output,
        seed=SEED,
        n_random_runs=200,
    )
    backtest = backtest_models(
        predictions=evaluation["predictions"],
        seed=SEED,
        n_random_runs=200,
    )

    split = train_output["split"]

    results = {
        "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "seed": SEED,
        "symbol": "NVDA",
        "data_summary": {
            "raw_rows": int(len(raw_data)),
            "feature_rows": int(len(feature_data)),
            "feature_count": int(len(split.feature_columns)),
            "raw_start": str(raw_data.index.min().date()),
            "raw_end": str(raw_data.index.max().date()),
            "test_start": str(split.test_df.index.min().date()),
            "test_end": str(split.test_df.index.max().date()),
        },
        "class_balance": train_output["class_balance"],
        "classification": evaluation["classification"],
        "backtest": backtest,
        "feature_importance": evaluation["feature_importance"],
        "plots": {
            **evaluation["plot_paths"],
            **backtest["plot_paths"],
        },
    }

    write_json(results, RESULTS_JSON_PATH)
    RESULTS_MD_PATH.write_text(_build_results_markdown(results), encoding="utf-8")

    return results


def main() -> None:
    """CLI entrypoint."""
    results = run_pipeline()

    classification = results["classification"]
    backtest = results["backtest"]

    print("Pipeline completed successfully.")
    print(
        "Test window: "
        f"{results['data_summary']['test_start']} -> {results['data_summary']['test_end']}"
    )
    print(
        "F1 scores: "
        f"LR={classification['logistic_regression']['f1']:.3f}, "
        f"RF={classification['random_forest']['f1']:.3f}"
    )
    print(
        "Strategy cumulative return: "
        f"LR={backtest['logistic_regression']['cumulative_return']:.2%}, "
        f"RF={backtest['random_forest']['cumulative_return']:.2%}, "
        f"BuyHold={backtest['buy_and_hold']['cumulative_return']:.2%}"
    )
    print(f"Results JSON: {RESULTS_JSON_PATH}")
    print(f"Results Markdown: {RESULTS_MD_PATH}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # pragma: no cover
        raise SystemExit(f"Pipeline failed: {exc}") from exc
