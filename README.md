NVDA 3-Day Return Direction Prediction

Review 1 – Data Acquisition, Preparation, and Exploratory Data Analysis

This project builds a reproducible data science workflow to predict whether NVDA (NVIDIA Corporation) will experience a positive 3-day forward return using historical market data and engineered technical indicators.

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

5. Modeling Plan (Next Phase)

Planned models:
	•	Logistic Regression (baseline linear classifier)
	•	Random Forest (nonlinear ensemble model)

Performance will be evaluated using:
	•	Accuracy
	•	F1 Score
	•	ROC Curve
	•	Strategy-level cumulative returns

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
