"""
Comprehensive Baseline Pipeline for BIS XTD Forecasting
========================================================
Runs Seasonal Naive, Prophet, Auto-SARIMA, ETS on all series.
Evaluates on test set (2026-01 onwards) with proper metrics.
"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from datasets import load_dataset
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
import pmdarima as pm
from tqdm import tqdm
import json

# ─── Load Data ──────────────────────────────────────────────────────────
print("📊 Loading dataset...")
ds = load_dataset("HugoxBaca/bis-xtd-deriv")
train_df = ds["train"].to_pandas()
test_df = ds["test"].to_pandas()

full_df = pd.concat([train_df, test_df]).sort_values(["currency", "risk_category", "instrument", "period"])

print(f"Train: {len(train_df):,} rows, Test: {len(test_df):,} rows")
print(f"Test period range: {test_df['period'].min()} to {test_df['period'].max()}")

# ─── Configuration ──────────────────────────────────────────────────────
HORIZON = 5  # Test only has 5 months (2026-02 to 2026-06)
SEASONALITY = 12

# ─── Baseline Functions ─────────────────────────────────────────────────

def seasonal_naive(series, horizon=HORIZON, seasonality=SEASONALITY):
    last_season = series[-seasonality:]
    reps = int(np.ceil(horizon / seasonality))
    return np.tile(last_season, reps)[:horizon]

def naive_forecast(series, horizon=HORIZON):
    return np.full(horizon, series[-1])

def prophet_forecast(series, horizon=HORIZON):
    dates = pd.date_range(end="2026-01-01", periods=len(series), freq="MS")
    df = pd.DataFrame({"ds": dates, "y": series})
    try:
        m = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
        m.fit(df)
        future = m.make_future_dataframe(periods=horizon, freq="MS")
        fc = m.predict(future)
        return fc["yhat"].values[-horizon:]
    except Exception as e:
        return np.full(horizon, np.nan)

def auto_sarima_forecast(series, horizon=HORIZON, seasonality=SEASONALITY):
    try:
        model = pm.auto_arima(
            series, 
            seasonal=True, m=seasonality,
            stepwise=True, suppress_warnings=True,
            error_action="ignore", trace=False,
            n_jobs=1,
            max_p=3, max_q=3, max_P=2, max_Q=2,
            max_order=8, max_d=2, max_D=1,
        )
        return model.predict(n_periods=horizon)
    except Exception as e:
        return np.full(horizon, np.nan)

def ets_forecast(series, horizon=HORIZON, seasonality=SEASONALITY):
    try:
        model = ExponentialSmoothing(
            series, trend="add", seasonal="add", seasonal_periods=seasonality
        ).fit()
        return model.forecast(horizon)
    except Exception as e:
        return np.full(horizon, np.nan)

# ─── Run Evaluation ─────────────────────────────────────────────────────
models = {
    "SeasonalNaive": seasonal_naive,
    "Naive": naive_forecast,
    "Prophet": prophet_forecast,
    "AutoSARIMA": auto_sarima_forecast,
    "ETS": ets_forecast,
}

results = []
series_groups = list(full_df.groupby(["currency", "risk_category", "instrument"]))

print(f"\n🏃 Running baselines on {len(series_groups)} series...")
for (currency, risk_cat, instrument), group in tqdm(series_groups, desc="Series"):
    group = group.sort_values("period")
    
    train = group[group["period"] < "2026-01-01"]["value"].values.astype(float)
    true = group[group["period"] >= "2026-01-01"]["value"].values.astype(float)
    
    if len(train) < 24 or len(true) < HORIZON:
        continue
    
    for model_name, model_fn in models.items():
        try:
            pred = model_fn(train)
            if np.any(np.isnan(pred)):
                continue
            
            # Align lengths
            min_len = min(len(true), len(pred))
            if min_len == 0:
                continue
                
            mae = mean_absolute_error(true[:min_len], pred[:min_len])
            mape = mean_absolute_percentage_error(true[:min_len] + 1e-8, pred[:min_len] + 1e-8) * 100
            
            results.append({
                "currency": currency,
                "risk_category": risk_cat,
                "instrument": instrument,
                "model": model_name,
                "mae": mae,
                "mape": mape,
            })
        except Exception as e:
            pass  # Silently skip failures

results_df = pd.DataFrame(results)
results_df.to_parquet("baseline_results.parquet", index=False)
print(f"\n✅ Saved {len(results_df)} results → baseline_results.parquet")

# ─── Summary ────────────────────────────────────────────────────────────
if len(results_df) > 0:
    print("\n📊 Overall MAE by Model:")
    summary = results_df.groupby("model")[["mae", "mape"]].agg(["mean", "median", "std"])
    print(summary.to_string())

    print("\n📊 MAE by Risk Category:")
    print(results_df.groupby(["risk_category", "model"])["mae"].mean().unstack().to_string())

    print("\n📊 MAE by Instrument:")
    print(results_df.groupby(["instrument", "model"])["mae"].mean().unstack().to_string())

    print("\n🏆 Best Model Distribution:")
    best_per_series = results_df.loc[results_df.groupby(["currency", "risk_category", "instrument"])["mae"].idxmin()]
    print(best_per_series["model"].value_counts())

    results_df.to_csv("baseline_detailed_results.csv", index=False)
    best_per_series.to_csv("best_model_per_series.csv", index=False)
else:
    print("❌ No results generated - all models failed")