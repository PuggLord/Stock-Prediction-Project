NVDA 3-Day Return Direction Prediction

End-to-end data science workflow: data acquisition, EDA, modeling, and backtesting.

This project builds a reproducible pipeline to predict whether NVDA (NVIDIA Corporation) will experience a positive 3-day forward return using historical market data and engineered technical indicators. Two classifiers (Logistic Regression and Random Forest) are trained, evaluated, and run through a daily-signal backtest against a buy-and-hold benchmark and a 200-run random baseline.

Headline Results (Test Set, Chronological Split)

| Strategy | Cumulative Return | Sharpe (rf=0) | Max Drawdown |
| --- | ---: | ---: | ---: |
| Logistic Regression | 707.39% | 1.042 | -64.22% |
| Random Forest | 53.37% | 1.311 | -4.94% |
| Buy and Hold | 1346.35% | 1.307 | -66.36% |
| Random Baseline (200 runs) | 324.22% ± 284.16% | 0.773 | -50.40% |

Key takeaways:
  • Random Forest delivers the best risk-adjusted return (Sharpe 1.311) with a -4.94% max drawdown by trading rarely (≈3.7% of days) but with high precision (0.755).
  • Logistic Regression captures most up-moves (recall 0.852) and produces a high cumulative return, but its ROC AUC (0.494) shows weak ranking power — its profit comes largely from class imbalance and a long bias during a strong bull period.
  • Buy-and-hold remains the cumulative-return benchmark; both strategies should be judged against it on risk-adjusted terms, not raw return.

⸻

1. Objective

The goal is to classify whether the 3-day forward return of NVDA is positive.

Target Definition:

y_{3d} = 1 \quad \text{if} \quad \left(\frac{Close_{t+3}}{Close_t} - 1\right) > 0, \text{ else } 0

This transforms a financial time series into a supervised classification problem.

⸻

2. Data Acquisition
	•	Source: Yahoo Finance
	•	Access Method: yfinance Python library
	•	Ticker: NVDA
	•	Frequency: Daily OHLCV
	•	Time Range: 2015–Present

The dataset includes:
	•	Open
	•	High
	•	Low
	•	Close
	•	Adjusted Close
	•	Volume

Data is downloaded and cached locally for reproducibility.

⸻

3. Data Preparation

Feature Engineering

Technical indicators were constructed using only historical data available at time t, including:
	•	Daily returns
	•	Rolling moving averages
	•	Volatility measures
	•	Momentum-based features

Forward returns were created using explicit shifting to prevent data leakage.

⸻

Handling Missing Values
	•	NaN values introduced from rolling windows were removed.
	•	Forward-shift rows with undefined targets were dropped.
	•	The percentage of removed rows was minimal and did not distort the dataset.

⸻

Data Leakage Controls
	•	Chronological train/test split (80% train, 20% test)
	•	No shuffling
	•	Scaling fit only on training data
	•	All forward-looking variables explicitly shifted

⸻

4. Exploratory Data Analysis (EDA)

EDA focused on understanding class balance, distribution shape, and feature relationships.

Target Distribution

The dataset is relatively balanced between positive and negative 3-day returns, reducing classification bias risk.

Returns Distribution
	•	Fat-tailed distribution
	•	Evidence of volatility clustering
	•	Non-normal characteristics

This suggests linear models may struggle without feature transformation.

⸻

Correlation Analysis

Correlation matrices revealed:
	•	High multicollinearity among OHLC price columns
	•	Weak direct linear correlation between raw volume and forward return
	•	Stronger relationships between momentum-based indicators and target direction

⸻

Key Insight

Momentum-based features show more predictive structure than raw trading volume.
This supports using nonlinear models capable of capturing interaction effects.

⸻

5. Modeling and Evaluation

Two classifiers were trained on the chronological 80% train split and evaluated on the held-out 20% test split (1,309 days, test period 2020-11-23 to 2026-02-10).

Classification Metrics (Test Set)

| Model | Accuracy | Precision | Recall | F1 | ROC AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| Logistic Regression | 0.544 | 0.568 | 0.852 | 0.681 | 0.494 |
| Random Forest | 0.447 | 0.755 | 0.049 | 0.093 | 0.517 |
| Random Baseline (200 runs) | 0.498 ± 0.014 | – | – | – | – |

Trading Simulation

A daily-signal strategy (`position[t] = prediction[t]`, PnL via `next_day_return[t]`) is run on the test set against buy-and-hold and a 200-run random baseline. See the Headline Results table above and `reports/results.md` for the full backtest summary.

Feature Importance Highlights
	•	Random Forest (impurity & permutation): `ema_21` ranks #1 in both views, followed by `ema_9`, `sma_9`.
	•	Logistic Regression (scaled coefficients): strongest positive `sma_9` (+0.531), strongest negative `sma_21` (-0.507).

Artifacts (committed under `reports/`)
	•	`results.md` — full metrics report
	•	`results.json` — machine-readable run summary
	•	`figures/` — confusion matrices, ROC curve, equity curve, feature-importance plots
	•	`test_predictions.csv` — per-day test-set predictions

⸻

6. Reproducibility

To run the full pipeline:

pip install -r requirements.txt
python run_all.py

All results, plots, and metrics are automatically generated in the reports/ directory.

⸻

Repository Structure

.
├── README.md
├── requirements.txt
├── run_all.py
├── data/
├── models/
├── notebooks/
├── reports/
└── src/

⸻
