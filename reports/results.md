# NVDA 3-Day Forward Return Prediction Report

## Brief Method
- Data: NVDA daily OHLCV from Yahoo Finance via yfinance.
- Target: `y_3d = 1` when `Close[t+3]/Close[t]-1 > 0`, else `0`.
- Features: technical indicators/returns/volume features using only current and past data.
- Split: chronological 80% train / 20% test.
- Models: Logistic Regression (with scaling) and Random Forest.
- Trading simulation: daily signal strategy (`position[t]=prediction[t]`, PnL via `next_day_return[t]`).

## Classification Metrics (Test Set)
| Model | Accuracy | Precision | Recall | F1 | ROC AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| Logistic Regression | 0.544 | 0.568 | 0.852 | 0.681 | 0.494 |
| Random Forest | 0.447 | 0.755 | 0.049 | 0.093 | 0.517 |
| Random Baseline (200 runs) | 0.498 +/- 0.014 | - | - | - | - |

## Backtest Summary (Test Set)
| Strategy | Cumulative Return | Annualized Return | Annualized Volatility | Sharpe (rf=0) | Max Drawdown |
| --- | ---: | ---: | ---: | ---: | ---: |
| Logistic Regression Strategy | 707.39% | 49.49% | 47.51% | 1.042 | -64.22% |
| Random Forest Strategy | 53.37% | 8.58% | 6.54% | 1.311 | -4.94% |
| Buy and Hold | 1346.35% | 67.25% | 51.46% | 1.307 | -66.36% |
| Random Baseline (200 runs) | 324.22% +/- 284.16% | 28.28% | 36.45% | 0.773 | -50.40% |

## 3-5 Key Takeaways
- Best predictive model by F1 on the test set: **Logistic Regression**.
- Class balance is moderate: train positive rate = 0.523, test positive rate = 0.572.
- Random baseline accuracy centers around 0.498 +/- 0.014, which is a useful sanity check for edge over chance.
- Buy-and-hold remains a strong benchmark for return comparison; strategy value should be judged against both risk and drawdown.
- Feature importance highlights: RF impurity top=ema_21, RF permutation top=ema_21, LR strongest positive=sma_9, LR strongest negative=sma_21.

## Feature Importance Highlights
### Random Forest (impurity) top 5
| Feature | Importance |
| --- | ---: |
| ema_21 | 0.0888 |
| volume_mean_14 | 0.0871 |
| ema_9 | 0.0827 |
| macd | 0.0781 |
| sma_9 | 0.0781 |

### Random Forest (permutation) top 5
| Feature | Importance |
| --- | ---: |
| ema_21 | 0.0848 |
| ema_9 | 0.0680 |
| sma_9 | 0.0563 |
| rsi_14 | 0.0177 |
| sma_21 | 0.0172 |

### Logistic Regression coefficients (scaled) top positive 5
| Feature | Coefficient |
| --- | ---: |
| sma_9 | 0.5309 |
| volatility_14 | 0.2338 |
| macd_signal | 0.2242 |
| rsi_14 | 0.0904 |
| ret_5d_back | 0.0580 |

### Logistic Regression coefficients (scaled) top negative 5
| Feature | Coefficient |
| --- | ---: |
| sma_21 | -0.5067 |
| macd | -0.2324 |
| ret_10d_back | -0.1445 |
| volatility_21 | -0.1410 |
| volume_mean_14 | -0.1240 |

## Artifacts
- JSON summary: `reports/results.json`
- Markdown report: `reports/results.md`
- Figures: `reports/figures/`
- Test predictions: `reports/test_predictions.csv`
