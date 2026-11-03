# Volatility Forecasting: Rolling Window vs EWMA vs GARCH

Compare one-day-ahead NIFTY 50 variance forecasts from a rolling historical variance, RiskMetrics-style EWMA, and GARCH(1,1), then evaluate them out of sample and compare India VIX with realized volatility.

## Run on Windows

Install Python 3.14 (64-bit), then open PowerShell in this folder and run:

```powershell
py -3.14 -m venv .venv314
.\.venv314\Scripts\python.exe -m pip install -r requirements.txt
.\.venv314\Scripts\python.exe main.py
```

The script downloads daily NIFTY 50 (`^NSEI`) and India VIX (`^INDIAVIX`) close data using yfinance. An internet connection is required. To select dates:

```powershell
.\.venv314\Scripts\python.exe main.py --start 2015-01-01 --end 2025-01-01
```

Use `--train-fraction 0.8`, `--window 21`, `--ewma-lambda 0.94`, and `--output outputs` to adjust the experiment. End date is exclusive. Yahoo Finance data can change and may occasionally be unavailable or rate limited.

## Method

- Compute log returns from NIFTY 50 daily close prices.
- Split returns chronologically. No random shuffle is used.
- Rolling forecast: variance of the previous 21 returns.
- EWMA forecast: update variance with lambda 0.94 after each observed return.
- GARCH forecast: estimate GARCH(1,1) parameters once on training returns, then keep parameters fixed and update the one-step variance recursion as each out-of-sample return becomes available.
- Score daily variance forecasts against squared next-day returns with MSE and QLIKE. Lower values are better.
- Compare India VIX with 21-trading-day trailing annualized realized volatility.

Squared daily returns are a noisy proxy for latent variance, so the scores are comparative rather than a definitive measure of the true variance. India VIX is an implied-volatility index with an approximately 30-day horizon, while the realized-volatility series uses a trailing 21-trading-day window; the plot is a contextual comparison, not a like-for-like forecast test. The script reports parameters and writes the full forecasts and scores for inspection.

## Outputs

Running the script creates:

- `outputs/model_scores.csv`
- `outputs/oos_forecasts.csv`
- `outputs/oos_volatility_forecasts.png`
- `outputs/vix_vs_realized_volatility.csv`
- `outputs/india_vix_vs_realized_volatility.png`

## Data and package references

Market data is retrieved via yfinance from Yahoo Finance. See the [yfinance download API](https://ranaroussi.github.io/yfinance/reference/api/yfinance.download.html) and [arch forecasting documentation](https://arch.readthedocs.io/en/stable/univariate/forecasting.html).

