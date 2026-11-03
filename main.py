from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
from arch import arch_model

TRADING_DAYS = 252


def close_series(ticker: str, start: str, end: str) -> pd.Series:
    """Download daily close prices, handling yfinance's single-ticker column layouts."""
    frame = yf.download(
        ticker, start=start, end=end, interval="1d", auto_adjust=True,
        progress=False, multi_level_index=False,
    )
    if frame is None or frame.empty:
        raise RuntimeError(f"No data returned for {ticker}. Check the ticker, dates, and internet connection.")
    close = frame["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    close = close.dropna().astype(float)
    close.name = ticker
    return close


def forecast_variances(returns: pd.Series, train_fraction: float = 0.8,
                       rolling_window: int = 21, ewma_lambda: float = 0.94) -> tuple[pd.DataFrame, int, object]:
    """Forecast next-day return variance out of sample using three models.

    GARCH parameters are estimated once on the training set. Its variance recursion
    updates using each newly observed return, with no parameter re-estimation.
    """
    returns = returns.dropna().astype(float)
    if not 0.5 <= train_fraction < 1:
        raise ValueError("train_fraction must be at least 0.5 and less than 1.")
    if rolling_window < 2:
        raise ValueError("rolling_window must be at least 2.")
    if not 0 < ewma_lambda < 1:
        raise ValueError("ewma_lambda must be between 0 and 1.")

    split = int(len(returns) * train_fraction)
    if split <= rolling_window + 10 or len(returns) - split < 20:
        raise ValueError("Not enough observations. Use a longer date range or adjust the training fraction.")

    scaled = returns * 100.0  # arch works more reliably with percentage-point returns.
    train = scaled.iloc[:split]
    model = arch_model(train, mean="Zero", vol="GARCH", p=1, q=1, dist="normal", rescale=False)
    fitted = model.fit(disp="off")
    params = fitted.params
    omega, alpha, beta = float(params["omega"]), float(params["alpha[1]"]), float(params["beta[1]"])

    # Conditional variance at the final training observation, in percent squared.
    garch_variance = float(fitted.conditional_volatility.iloc[-1] ** 2)
    previous_scaled_return = float(train.iloc[-1])
    ewma_variance = float(train.var(ddof=0))
    for observed in train.iloc[1:]:
        ewma_variance = ewma_lambda * ewma_variance + (1.0 - ewma_lambda) * float(observed) ** 2

    records: list[dict[str, float | pd.Timestamp]] = []
    for i in range(split, len(returns)):
        index = returns.index[i]
        history = scaled.iloc[max(0, i - rolling_window):i]
        rolling_var = float(history.var(ddof=0))
        garch_variance = omega + alpha * previous_scaled_return ** 2 + beta * garch_variance

        records.append({
            "Date": index,
            "Actual_Return": float(returns.iloc[i]),
            "Actual_Variance_Proxy": float(returns.iloc[i] ** 2),
            "Rolling_Variance": rolling_var / 10_000.0,
            "EWMA_Variance": ewma_variance / 10_000.0,
            "GARCH_Variance": garch_variance / 10_000.0,
        })

        # Update the variance states only after this day's actual return is observed.
        observed_scaled = float(scaled.iloc[i])
        ewma_variance = ewma_lambda * ewma_variance + (1.0 - ewma_lambda) * observed_scaled ** 2
        previous_scaled_return = observed_scaled

    forecasts = pd.DataFrame.from_records(records).set_index("Date")
    return forecasts, split, fitted


def score_forecasts(forecasts: pd.DataFrame) -> pd.DataFrame:
    """Compare variance forecasts to squared returns using MSE and QLIKE."""
    actual = forecasts["Actual_Variance_Proxy"].clip(lower=1e-12)
    rows = []
    for column in ["Rolling_Variance", "EWMA_Variance", "GARCH_Variance"]:
        predicted = forecasts[column].clip(lower=1e-12)
        qlike = np.log(predicted / actual) + actual / predicted - 1.0
        rows.append({"Model": column.replace("_Variance", ""),
                     "MSE": float(np.mean((predicted - actual) ** 2)),
                     "Mean_QLIKE": float(qlike.mean())})
    return pd.DataFrame(rows).sort_values("Mean_QLIKE").reset_index(drop=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare rolling, EWMA, and GARCH(1,1) volatility forecasts for NIFTY 50.")
    parser.add_argument("--start", default="2012-01-01", help="Start date (YYYY-MM-DD). Default: 2012-01-01")
    parser.add_argument("--end", default=None, help="End date (exclusive, YYYY-MM-DD). Default: latest available")
    parser.add_argument("--train-fraction", type=float, default=0.8, help="Fraction of return observations used for training")
    parser.add_argument("--window", type=int, default=21, help="Rolling variance lookback in trading days")
    parser.add_argument("--ewma-lambda", type=float, default=0.94, help="EWMA decay parameter")
    parser.add_argument("--output", default="outputs", help="Directory for CSVs and charts")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    end = args.end or (pd.Timestamp.today().normalize() + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Downloading NIFTY 50 and India VIX daily data from Yahoo Finance...")
    nifty = close_series("^NSEI", args.start, end)
    vix = close_series("^INDIAVIX", args.start, end)
    returns = np.log(nifty).diff().dropna()
    forecasts, split, fitted = forecast_variances(returns, args.train_fraction, args.window, args.ewma_lambda)
    scores = score_forecasts(forecasts)

    forecasts.to_csv(output_dir / "oos_forecasts.csv", float_format="%.10g")
    scores.to_csv(output_dir / "model_scores.csv", index=False, float_format="%.10g")

    train_end = returns.index[split - 1]
    print(f"Training observations: {split:,} (through {train_end.date()})")
    print(f"Out-of-sample observations: {len(forecasts):,}")
    print("\nOut-of-sample scores (lower is better):")
    print(scores.to_string(index=False, formatters={"MSE": "{:.8g}".format, "Mean_QLIKE": "{:.6f}".format}))
    print("\nGARCH(1,1) parameters estimated on training data only:")
    print(fitted.params.to_string())

    fig, ax = plt.subplots(figsize=(12, 6))
    (forecasts[["Rolling_Variance", "EWMA_Variance", "GARCH_Variance"]] * TRADING_DAYS * 10_000).pow(0.5).plot(ax=ax)
    (forecasts["Actual_Variance_Proxy"] * TRADING_DAYS * 10_000).pow(0.5).rename("Realized proxy (daily return)").plot(ax=ax, alpha=0.45, linewidth=0.7)
    ax.set(title="NIFTY 50: out-of-sample annualized volatility forecasts", ylabel="Percent", xlabel="Date")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "oos_volatility_forecasts.png", dpi=160)
    plt.close(fig)

    comparison = pd.DataFrame({"India_VIX": vix, "NIFTY_Return": returns}).dropna()
    comparison["Realized_Vol_21d"] = comparison["NIFTY_Return"].rolling(21).std(ddof=1) * np.sqrt(TRADING_DAYS) * 100.0
    comparison = comparison.dropna()
    comparison.to_csv(output_dir / "vix_vs_realized_volatility.csv", float_format="%.8g")
    fig, ax = plt.subplots(figsize=(12, 6))
    comparison[["India_VIX", "Realized_Vol_21d"]].plot(ax=ax)
    ax.set(title="India VIX and trailing 21-trading-day NIFTY 50 realized volatility", ylabel="Annualized volatility (%)", xlabel="Date")
    ax.legend(["India VIX (implied, approx. 30-day)", "NIFTY realized volatility (trailing 21 days)"])
    fig.tight_layout()
    fig.savefig(output_dir / "india_vix_vs_realized_volatility.png", dpi=160)
    plt.close(fig)
    print(f"\nSaved results to: {output_dir.resolve()}")
    print("Note: squared daily returns are a noisy proxy for latent daily variance; VIX and trailing realized volatility use different horizons.")


if __name__ == "__main__":
    main()
