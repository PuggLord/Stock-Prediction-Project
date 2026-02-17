from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.utils import FIGURES_DIR, SEED, ensure_directories

EQUITY_CURVE_PATH = FIGURES_DIR / "equity_curve_test.png"


def _equity_curve(daily_returns: pd.Series) -> pd.Series:
    returns = daily_returns.fillna(0.0)
    return (1.0 + returns).cumprod()


def compute_trading_metrics(daily_returns: pd.Series) -> dict[str, float]:
    """Compute standard trading metrics from daily return series."""
    returns = daily_returns.fillna(0.0)

    if len(returns) == 0:
        return {
            "cumulative_return": 0.0,
            "annualized_return": 0.0,
            "annualized_volatility": 0.0,
            "sharpe": 0.0,
            "max_drawdown": 0.0,
        }

    equity = _equity_curve(returns)
    cumulative_return = float(equity.iloc[-1] - 1.0)

    periods = len(returns)
    annualized_return = float((1.0 + cumulative_return) ** (252.0 / periods) - 1.0)
    annualized_volatility = float(returns.std(ddof=0) * np.sqrt(252.0))
    sharpe = float(annualized_return / annualized_volatility) if annualized_volatility > 0 else 0.0

    running_max = equity.cummax()
    drawdown = equity / running_max - 1.0
    max_drawdown = float(drawdown.min())

    return {
        "cumulative_return": cumulative_return,
        "annualized_return": annualized_return,
        "annualized_volatility": annualized_volatility,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
    }


def run_daily_signal_strategy(
    prediction_signal: pd.Series,
    next_day_return: pd.Series,
) -> tuple[pd.Series, dict[str, float]]:
    """Daily strategy: position[t]=prediction[t], pnl[t]=position[t]*next_day_return[t]."""
    aligned_signal = prediction_signal.astype(float).clip(lower=0.0, upper=1.0)
    aligned_returns = next_day_return.astype(float)

    strategy_daily_return = aligned_signal * aligned_returns
    metrics = compute_trading_metrics(strategy_daily_return)
    metrics["average_position"] = float(aligned_signal.mean())

    equity = _equity_curve(strategy_daily_return)
    return equity, metrics


def simulate_random_backtest_baseline(
    next_day_return: pd.Series,
    n_runs: int = 200,
    seed: int = SEED,
) -> dict[str, float]:
    """Random 50/50 signal baseline for strategy return."""
    rng = np.random.default_rng(seed)

    cumulative_returns = []
    annualized_returns = []
    annualized_volatilities = []
    sharpes = []
    max_drawdowns = []

    returns = next_day_return.astype(float)

    for _ in range(n_runs):
        random_signal = pd.Series(rng.integers(0, 2, size=len(returns)), index=returns.index)
        random_daily = random_signal * returns
        stats = compute_trading_metrics(random_daily)

        cumulative_returns.append(stats["cumulative_return"])
        annualized_returns.append(stats["annualized_return"])
        annualized_volatilities.append(stats["annualized_volatility"])
        sharpes.append(stats["sharpe"])
        max_drawdowns.append(stats["max_drawdown"])

    return {
        "n_runs": n_runs,
        "cumulative_return_mean": float(np.mean(cumulative_returns)),
        "cumulative_return_std": float(np.std(cumulative_returns, ddof=0)),
        "annualized_return_mean": float(np.mean(annualized_returns)),
        "annualized_return_std": float(np.std(annualized_returns, ddof=0)),
        "annualized_volatility_mean": float(np.mean(annualized_volatilities)),
        "sharpe_mean": float(np.mean(sharpes)),
        "max_drawdown_mean": float(np.mean(max_drawdowns)),
    }


def _plot_equity_curves(curves: dict[str, pd.Series], path: Path) -> None:
    fig, axis = plt.subplots(figsize=(10, 6))

    for name, curve in curves.items():
        axis.plot(curve.index, curve.values, label=name, linewidth=2)

    axis.set_title("Equity Curve on Test Period")
    axis.set_xlabel("Date")
    axis.set_ylabel("Portfolio Value (Start = 1.0)")
    axis.grid(alpha=0.3)
    axis.legend()

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def backtest_models(
    predictions: pd.DataFrame,
    seed: int = SEED,
    n_random_runs: int = 200,
) -> dict[str, Any]:
    """Backtest LR and RF model signals against buy-and-hold and random baseline."""
    ensure_directories()

    required_columns = ["lr_pred", "rf_pred", "next_day_return"]
    missing = [column for column in required_columns if column not in predictions.columns]
    if missing:
        raise ValueError(
            "Predictions dataframe is missing required columns for backtest: "
            f"{missing}."
        )

    next_day_return = predictions["next_day_return"].astype(float)

    lr_equity, lr_metrics = run_daily_signal_strategy(predictions["lr_pred"], next_day_return)
    rf_equity, rf_metrics = run_daily_signal_strategy(predictions["rf_pred"], next_day_return)

    buy_hold_equity = _equity_curve(next_day_return)
    buy_hold_metrics = compute_trading_metrics(next_day_return)

    random_baseline = simulate_random_backtest_baseline(
        next_day_return=next_day_return,
        n_runs=n_random_runs,
        seed=seed,
    )

    _plot_equity_curves(
        curves={
            "Logistic Strategy": lr_equity,
            "Random Forest Strategy": rf_equity,
            "Buy and Hold": buy_hold_equity,
        },
        path=EQUITY_CURVE_PATH,
    )

    return {
        "strategy_definition": (
            "Daily signal strategy: position[t] = model prediction at t (0 or 1), "
            "and daily PnL uses next_day_return[t]."
        ),
        "logistic_regression": lr_metrics,
        "random_forest": rf_metrics,
        "buy_and_hold": buy_hold_metrics,
        "random_baseline": random_baseline,
        "plot_paths": {
            "equity_curve": str(EQUITY_CURVE_PATH),
        },
    }


def main() -> None:
    """CLI entrypoint for backtesting from saved test predictions."""
    predictions_path = Path("reports/test_predictions.csv")
    if not predictions_path.exists():
        raise FileNotFoundError(
            "reports/test_predictions.csv not found. Run training/evaluation first."
        )

    predictions = pd.read_csv(predictions_path, parse_dates=["Date"], index_col="Date")
    output = backtest_models(predictions)

    print("Backtest complete.")
    print(f"LR cumulative return: {output['logistic_regression']['cumulative_return']:.2%}")
    print(f"RF cumulative return: {output['random_forest']['cumulative_return']:.2%}")
    print(f"Buy-and-hold cumulative return: {output['buy_and_hold']['cumulative_return']:.2%}")


if __name__ == "__main__":
    main()
