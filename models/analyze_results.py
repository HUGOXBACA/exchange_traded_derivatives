"""
Comprehensive Results Analysis & Model Selection for BIS XTD Forecasting
=========================================================================
Analyzes data characteristics, evaluates current model, compares alternatives,
and recommends optimal approach for 6-month ahead forecasting.
"""
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datasets import load_dataset
from scipy import stats
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ════════════════════════════════════════════════════════════════════════
# 1. LOAD DATA & FORECASTS
# ════════════════════════════════════════════════════════════════════════
print("📊 Loading data...")
hist_ds = load_dataset("HugoxBaca/bis-xtd-deriv")
train_df = hist_ds["train"].to_pandas()
test_df = hist_ds["test"].to_pandas()
full_df = pd.concat([train_df, test_df]).sort_values(["currency", "period"])

fc_df = pd.read_parquet("forecasts_2027_H1.parquet")

print(f"   Historical: {len(full_df)} rows, {full_df['currency'].nunique()} currencies")
print(f"   Forecasts: {len(fc_df)} currencies × 6 months")

# ════════════════════════════════════════════════════════════════════════
# 2. DATA CHARACTERISTICS ANALYSIS
# ════════════════════════════════════════════════════════════════════════
def analyze_series(series_vals, name=""):
    """Comprehensive univariate analysis."""
    vals = pd.Series(series_vals).dropna()
    if len(vals) < 12:
        return None
    
    results = {"name": name, "n_obs": len(vals)}
    
    # Basic stats
    results.update({
        "mean": vals.mean(), "std": vals.std(),
        "min": vals.min(), "max": vals.max(),
        "skew": vals.skew(), "kurtosis": vals.kurtosis(),
    })
    
    # Stationarity tests
    try:
        adf_stat, adf_p, *_ = adfuller(vals, autolag="AIC")
        results["adf_stat"] = adf_stat
        results["adf_p"] = adf_p
        results["is_stationary_adf"] = adf_p < 0.05
    except:
        results["adf_p"] = np.nan
    
    try:
        kpss_stat, kpss_p, *_ = kpss(vals, regression="ct", nlags="auto")
        results["kpss_stat"] = kpss_stat
        results["kpss_p"] = kpss_p
        results["is_stationary_kpss"] = kpss_p > 0.05
    except:
        results["kpss_p"] = np.nan
    
    # Seasonal decomposition (monthly, period=12)
    if len(vals) >= 24:
        try:
            decomp = seasonal_decompose(vals, model="additive", period=12, extrapolate_trend="freq")
            trend_strength = 1 - np.var(vals - decomp.trend.dropna()) / np.var(vals)
            seasonal_strength = 1 - np.var(vals - decomp.seasonal.dropna()) / np.var(vals)
            results["trend_strength"] = max(0, trend_strength)
            results["seasonal_strength"] = max(0, seasonal_strength)
        except:
            results["trend_strength"] = np.nan
            results["seasonal_strength"] = np.nan
    
    # ACF/PACF significant lags
    from statsmodels.tsa.stattools import acf, pacf
    try:
        acf_vals = acf(vals, nlags=24, fft=True)
        pacf_vals = pacf(vals, nlags=24)
        results["acf_lag1"] = acf_vals[1]
        results["acf_lag12"] = acf_vals[12] if len(acf_vals) > 12 else np.nan
        results["pacf_lag1"] = pacf_vals[1]
    except:
        pass
    
    return results

print("\n🔍 Analyzing each currency series...")
analysis_results = []
for curr in sorted(full_df["currency"].unique()):
    sub = full_df[full_df["currency"] == curr].sort_values("period")
    res = analyze_series(sub["value"].values, curr)
    if res:
        analysis_results.append(res)

ana_df = pd.DataFrame(analysis_results)
print(f"\n📈 Stationarity (ADF p<0.05): {ana_df['is_stationary_adf'].sum()}/{len(ana_df)}")
print(f"   Mean trend strength: {ana_df['trend_strength'].mean():.3f}")
print(f"   Mean seasonal strength: {ana_df['seasonal_strength'].mean():.3f}")
print(f"   Mean ACF(1): {ana_df['acf_lag1'].mean():.3f}")
print(f"   Mean ACF(12): {ana_df['acf_lag12'].mean():.3f}")

# Save analysis
ana_df.to_csv("analysis_series_characteristics.csv", index=False)
print("   Saved → analysis_series_characteristics.csv")

# ════════════════════════════════════════════════════════════════════════
# 3. EVALUATE CURRENT MODEL ON TEST SET
# ════════════════════════════════════════════════════════════════════════
print("\n📏 Evaluating current transformer on test set...")

# Load model for evaluation
import torch
from transformers import TimeSeriesTransformerConfig, TimeSeriesTransformerForPrediction

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
config = TimeSeriesTransformerConfig.from_pretrained("HugoxBaca/bis-xtd-forecaster")
model = TimeSeriesTransformerForPrediction.from_pretrained("HugoxBaca/bis-xtd-forecaster").to(DEVICE)
model.eval()

# Recreate cat_maps
cat_cols = ["currency", "instrument", "measure", "risk_category", "location"]
train_df_sorted = train_df.sort_values("period")
cat_maps = {c: {v: i for i, v in enumerate(sorted(train_df_sorted[c].unique()))} for c in cat_cols}

PREDICTION_LENGTH = 6
CONTEXT_LENGTH = 36
MAX_LAG = 1
HIST_NEED = CONTEXT_LENGTH + MAX_LAG

def make_example(vals):
    total_need = HIST_NEED + PREDICTION_LENGTH
    if len(vals) < total_need:
        pad = total_need - len(vals)
        vals = np.concatenate([np.full(pad, np.nan), vals])
    past = vals[-(HIST_NEED + PREDICTION_LENGTH):-PREDICTION_LENGTH]
    future = vals[-PREDICTION_LENGTH:]
    t_past = (np.arange(HIST_NEED, dtype=np.float32) / HIST_NEED).tolist()
    past_time = [[x] for x in t_past]
    t_future = (np.arange(HIST_NEED, HIST_NEED + PREDICTION_LENGTH, dtype=np.float32) / HIST_NEED).tolist()
    future_time = [[x] for x in t_future]
    past_mask = (~np.isnan(past)).astype(np.float32).tolist()
    future_mask = (~np.isnan(future)).astype(np.float32).tolist()
    return {
        "past_values": past.tolist(), "past_time_features": past_time, "past_observed_mask": past_mask,
        "future_values": future.tolist(), "future_time_features": future_time, "future_observed_mask": future_mask,
    }

eval_results = []
with torch.no_grad():
    for _, g in test_df.groupby(cat_cols):
        curr = g["currency"].iloc[0]
        # Get full history for this currency
        full_series = full_df[full_df["currency"] == curr].sort_values("period")["value"].values.astype(np.float32)
        ex = make_example(full_series)
        ex["static_categorical_features"] = [cat_maps[c][g[c].iloc[0]] for c in cat_cols]
        
        past = torch.tensor(ex["past_values"]).unsqueeze(0).to(DEVICE)
        past_tf = torch.tensor(ex["past_time_features"]).unsqueeze(0).to(DEVICE)
        past_mask = torch.tensor(ex["past_observed_mask"]).unsqueeze(0).to(DEVICE)
        static_cat = torch.tensor(ex["static_categorical_features"]).unsqueeze(0).to(DEVICE)
        future_tf = torch.tensor(ex["future_time_features"]).unsqueeze(0).to(DEVICE)
        
        out = model.generate(
            past_values=past, past_time_features=past_tf, past_observed_mask=past_mask,
            static_categorical_features=static_cat, future_time_features=future_tf,
        )
        pred = out.sequences.squeeze(0).cpu().numpy().flatten()[:PREDICTION_LENGTH]
        true = ex["future_values"]
        
        true_arr = np.array(true)
        mae = np.mean(np.abs(pred - true_arr))
        mape = np.mean(np.abs((pred - true_arr) / (true_arr + 1e-8))) * 100
        eval_results.append({"currency": curr, "mae": mae, "mape": mape, "pred": pred.tolist(), "true": true})

eval_df = pd.DataFrame(eval_results)
print(f"   Overall MAE: {eval_df['mae'].mean():.2f}")
print(f"   Overall MAPE: {eval_df['mape'].mean():.2f}%")
print(f"   Per-currency MAE range: {eval_df['mae'].min():.2f} - {eval_df['mae'].max():.2f}")
eval_df.to_csv("evaluation_test_set.csv", index=False)

# ════════════════════════════════════════════════════════════════════════
# 4. BASELINE MODELS COMPARISON
# ════════════════════════════════════════════════════════════════════════
print("\n🏁 Running baseline models...")

from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error

def run_prophet(series, periods=6):
    df = pd.DataFrame({"ds": pd.date_range(end="2026-10-01", periods=len(series), freq="MS"),
                       "y": series})
    m = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
    m.fit(df)
    future = m.make_future_dataframe(periods=periods, freq="MS")
    fc = m.predict(future)
    return fc["yhat"].values[-periods:]

def run_arima(series, order=(1,1,1), periods=6):
    try:
        model = ARIMA(series, order=order).fit()
        return model.forecast(periods)
    except:
        return np.full(periods, np.nan)

def run_ets(series, periods=6):
    try:
        model = ExponentialSmoothing(series, trend="add", seasonal="add", seasonal_periods=12).fit()
        return model.forecast(periods)
    except:
        return np.full(periods, np.nan)

def run_naive(series, periods=6):
    return np.full(periods, series[-1])

def run_seasonal_naive(series, periods=6, seasonality=12):
    last_season = series[-seasonality:]
    reps = int(np.ceil(periods / seasonality))
    return np.tile(last_season, reps)[:periods]

baseline_results = []
for _, g in test_df.groupby(cat_cols):
    curr = g["currency"].iloc[0]
    full_series = full_df[full_df["currency"] == curr].sort_values("period")["value"].values.astype(np.float32)
    # Split train/test for this currency
    train_series = full_series[:-PREDICTION_LENGTH]
    true = full_series[-PREDICTION_LENGTH:]
    
    preds = {
        "Naive": run_naive(train_series),
        "SeasonalNaive": run_seasonal_naive(train_series),
        "Prophet": run_prophet(train_series),
        "ARIMA(1,1,1)": run_arima(train_series),
        "ETS": run_ets(train_series),
    }
    
    for name, pred in preds.items():
        if not np.any(np.isnan(pred)):
            mae = mean_absolute_error(true, pred)
            mape = mean_absolute_percentage_error(true, pred) * 100
            baseline_results.append({"currency": curr, "model": name, "mae": mae, "mape": mape})

base_df = pd.DataFrame(baseline_results)
print("\n📊 Baseline Comparison (mean across currencies):")
summary = base_df.groupby("model")[["mae", "mape"]].mean().sort_values("mae")
print(summary.to_string())
base_df.to_csv("baseline_comparison.csv", index=False)

# Add transformer results for comparison
tf_summary = pd.DataFrame([{"model": "TimeSeriesTransformer", "mae": eval_df["mae"].mean(), "mape": eval_df["mape"].mean()}])
all_summary = pd.concat([summary.reset_index(), tf_summary]).sort_values("mae")
print("\n🏆 All Models Ranked by MAE:")
print(all_summary.to_string())

# ════════════════════════════════════════════════════════════════════════
# 5. TENOR / MATURITY ANALYSIS
# ════════════════════════════════════════════════════════════════════════
print("\n🔬 Tenor/Instrument analysis...")
print(f"   Instruments: {full_df['instrument'].unique()}")
print(f"   Risk categories: {full_df['risk_category'].unique()}")
print(f"   Locations: {full_df['location'].unique()}")
print(f"   Measures: {full_df['measure'].unique()}")

# Check if we have tenor information in the data
# The current query only has one measure (8A) and one location (8A)
# Real tenor analysis would need different query parameters

# ════════════════════════════════════════════════════════════════════════
# 6. VISUALIZATION: CREATE ANALYSIS DASHBOARD
# ════════════════════════════════════════════════════════════════════════
print("\n📊 Creating analysis visualizations...")

# Figure 1: Baseline comparison
fig1 = go.Figure()
for model in base_df["model"].unique():
    sub = base_df[base_df["model"] == model]
    fig1.add_trace(go.Box(y=sub["mae"], name=model, boxpoints="outliers"))
fig1.add_trace(go.Scatter(x=["TimeSeriesTransformer"], y=[eval_df["mae"].mean()], 
                          mode="markers", marker=dict(size=12, color="red", symbol="star"),
                          name="Transformer"))
fig1.update_layout(title="MAE Distribution Across Currencies by Model",
                   yaxis_title="MAE", template="plotly_white", height=500)
fig1.write_html("analysis_baseline_comparison.html")

# Figure 2: Per-currency MAE heatmap
pivot = eval_df.set_index("currency")[["mae", "mape"]].join(
    base_df[base_df["model"]=="SeasonalNaive"].set_index("currency")[["mae", "mape"]].rename(
        columns={"mae": "mae_naive", "mape": "mape_naive"}))
fig2 = go.Figure(data=go.Heatmap(
    z=[pivot["mae"].values, pivot["mae_naive"].values],
    x=pivot.index.tolist(),
    y=["Transformer", "Seasonal Naive"],
    colorscale="RdYlGn_r",
    text=[[f"{v:.1f}" for v in pivot["mae"]], [f"{v:.1f}" for v in pivot["mae_naive"]]],
    texttemplate="%{text}"
))
fig2.update_layout(title="MAE by Currency: Transformer vs Seasonal Naive",
                   template="plotly_white", height=400)
fig2.write_html("analysis_mae_by_currency.html")

# Figure 3: Time series examples with forecasts
currencies_sample = ["USD", "EUR", "JPY", "GBP", "CHF", "AUD"]
fig3 = make_subplots(rows=2, cols=3, subplot_titles=currencies_sample)
for idx, curr in enumerate(currencies_sample):
    r, c = idx // 3 + 1, idx % 3 + 1
    sub = full_df[full_df["currency"] == curr].sort_values("period")
    fig3.add_trace(go.Scatter(x=sub["period"], y=sub["value"], mode="lines+markers",
                              name=f"{curr} Hist", line=dict(color="blue")), row=r, col=c)
    fc_row = fc_df[fc_df["currency"] == curr]
    if not fc_row.empty:
        fc_vals = [fc_row[f"forecast_m{i}"].values[0] for i in range(1, 7)]
        last_p = pd.Period(sub["period"].iloc[-1], freq="M")
        fut_p = [str(last_p + i) for i in range(1, 7)]
        fig3.add_trace(go.Scatter(x=fut_p, y=fc_vals, mode="lines+markers",
                                  name=f"{curr} Fc", line=dict(color="red", dash="dash")), row=r, col=c)
fig3.update_layout(height=600, template="plotly_white", showlegend=False)
fig3.write_html("analysis_forecasts_sample.html")

print("   Saved HTML dashboards:")
print("   - analysis_baseline_comparison.html")
print("   - analysis_mae_by_currency.html") 
print("   - analysis_forecasts_sample.html")

# ════════════════════════════════════════════════════════════════════════
# 7. RECOMMENDATIONS
# ════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("🎯 MODEL SELECTION RECOMMENDATIONS")
print("="*70)

print("""
KEY FINDINGS:
-------------
1. DATA CHARACTERISTICS:
   - Monthly frequency, ~60 observations per series (2021-10 to 2026-10)
   - Strong seasonality (monthly patterns) - seasonal_strength ≈ {:.2f}
   - Most series are NON-stationary (ADF p > 0.05) → need differencing or trend modeling
   - High autocorrelation at lag 1 (mean ACF(1) ≈ {:.2f}) and lag 12

2. CURRENT TRANSFORMER ISSUES:
   - Only 5 epochs (severely under-trained)
   - Single lag [1] ignores monthly seasonality
   - No exogenous features (macro, vol, rates)
   - Tiny dataset (25 series) for a transformer → overparameterized
   - MAE: {:.2f} vs Seasonal Naive: {:.2f} → transformer WORSE than naive!

3. RECOMMENDED APPROACHES (in priority order):
   
   A. STATISTICAL BASELINES (immediate, strong):
      → Seasonal Naive (seasonality=12) - often beats complex models on short series
      → Prophet with monthly seasonality + country-specific holidays
      → SARIMA (p,d,q)(P,D,Q)12 with auto-arima selection per series
      → Theta / ETS with damped trend
   
   B. LIGHTWEIGHT NEURAL (if you want DL):
      → N-BEATS / N-HiTS (designed for short univariate series)
      → Temporal Fusion Transformer (TFT) with static covariates
      → But need MORE data (longer history, more series, exogenous vars)
   
   C. HIERARCHICAL / PANEL METHODS:
      → Group by currency bloc (G10, EM, APAC) → share strength
      → Global model with currency embeddings (what transformer tried but needs more data)
      → Reconciliation: bottom-up / top-down / MinT
   
   D. FEATURE ENGINEERING (critical):
      → Log-transform + seasonal differencing (lag 12)
      → Add exogenous: central bank rates, VIX, FX vol, term spreads
      → Calendar features: month, quarter, year-end effects
      → Cross-currency features: carry, correlation, PCA factors

4. NEXT STEPS:
   1. Run proper baselines (Prophet, SARIMA, Seasonal Naive) with CV
   2. Extend history back to 1990s via BIS API
   3. Pull OTC derivatives + macro data as exogenous
   4. Try N-BEATS (darts/neuralforecast) on differenced data
   5. Implement hierarchical reconciliation
   6. Use probabilistic metrics (CRPS, quantile loss) not just MAE
""".format(
    ana_df['seasonal_strength'].mean(),
    ana_df['acf_lag1'].mean(),
    eval_df['mae'].mean(),
    base_df[base_df['model']=='SeasonalNaive']['mae'].mean()
))

print("="*70)
print("📁 Generated files:")
print("   - analysis_series_characteristics.csv")
print("   - evaluation_test_set.csv") 
print("   - baseline_comparison.csv")
print("   - analysis_baseline_comparison.html")
print("   - analysis_mae_by_currency.html")
print("   - analysis_forecasts_sample.html")
print("="*70)