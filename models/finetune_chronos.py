"""
Evaluate Amazon Chronos Foundation Model on BIS XTD Data
==========================================================
Chronos is pre-trained on 100B+ time series points.
Evaluate zero-shot performance on our 252 series (FX + IR) for 5-month horizon.
"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from chronos import ChronosPipeline
from tqdm import tqdm

# ─── Config ──────────────────────────────────────────────────────────────
PREDICTION_LENGTH = 5  # Test has 5 months (2026-02 to 2026-06)

# ─── Load Data ──────────────────────────────────────────────────────────
print("📊 Loading dataset...")
ds = load_dataset("HugoxBaca/bis-xtd-deriv")
train_df = ds["train"].to_pandas()
test_df = ds["test"].to_pandas()
full_df = pd.concat([train_df, test_df]).sort_values(["currency", "risk_category", "instrument", "period"])

print(f"Test period range: {test_df['period'].min()} to {test_df['period'].max()}")
print(f"Test shape: {test_df.shape}")

# ─── Load Chronos Pipeline ──────────────────────────────────────────────
print("🔄 Loading amazon/chronos-t5-small on CPU...")
pipeline = ChronosPipeline.from_pretrained(
    "amazon/chronos-t5-small",
    device_map="cpu",
)

# ─── MASE Function ──────────────────────────────────────────────────────
def mase(y_true, y_pred, y_train, seasonality=12):
    naive_errors = np.abs(y_train[seasonality:] - y_train[:-seasonality])
    scale = np.mean(naive_errors) if len(naive_errors) > 0 else np.mean(np.abs(np.diff(y_train)))
    if scale == 0:
        return np.nan
    return np.mean(np.abs(y_true - y_pred)) / scale

# ─── Evaluation Function ────────────────────────────────────────────────
def evaluate_chronos(pipeline, test_df, full_df, regime_name="FX", prediction_length=5):
    """Evaluate Chronos on test set."""
    results = []
    
    for keys, group in tqdm(test_df.groupby(["currency", "risk_category", "instrument"]), desc=f"Chronos {regime_name}"):
        currency, risk_cat, inst = keys
        
        group = group.sort_values("period")
        test_series = group[group["period"] >= "2026-01-01"]
        if len(test_series) < 1:
            continue
        
        true = test_series["value"].values[:prediction_length].astype(float)
        if len(true) < prediction_length:
            continue
        
        # Get training data
        train_series = full_df[
            (full_df["currency"]==currency) & 
            (full_df["risk_category"]==risk_cat) & 
            (full_df["instrument"]==inst)
        ].sort_values("period")
        train_vals = train_series[train_series["period"] < "2026-01-01"]["value"].values.astype(float)
        
        if len(train_vals) < 12:
            continue
        
        # Handle zeros
        train_vals = np.where(train_vals == 0, np.nan, train_vals)
        train_vals = pd.Series(train_vals).ffill().bfill().values
        
        # Seasonal Naive baseline
        seasonal_naive = np.tile(train_vals[-12:], int(np.ceil(prediction_length/12)))[:prediction_length]
        sn_mase = mase(true, seasonal_naive, train_vals)
        
        # Chronos prediction (zero-shot)
        try:
            context = torch.tensor(np.log(train_vals[-128:]), dtype=torch.float32).unsqueeze(0)
            forecast = pipeline.predict(
                context,
                prediction_length=prediction_length,
                num_samples=20,
                temperature=1.0,
                top_k=50,
                top_p=1.0,
            )
            # forecast shape: (num_samples, prediction_length)
            median_forecast = np.median(forecast[0].cpu().numpy(), axis=0)
            pred = np.exp(median_forecast)  # inverse log
            
            chronos_mase = mase(true, pred, train_vals)
            
            results.append({
                "series_id": f"{currency}_{risk_cat}_{inst}",
                "currency": currency,
                "risk_category": risk_cat,
                "instrument": inst,
                "SeasonalNaive_MASE": mase(true, seasonal_naive, train_vals),
                "Chronos_MASE": chronos_mase,
                "true_mean": np.mean(true),
            })
        except Exception as e:
            print(f"  Error on {currency}_{risk_cat}_{inst}: {e}")
            continue
    
    return pd.DataFrame(results)

# ─── MASE Function ──────────────────────────────────────────────────────
def mase(y_true, y_pred, y_train, seasonality=12):
    naive_errors = np.abs(y_train[seasonality:] - y_train[:-seasonality])
    scale = np.mean(naive_errors) if len(naive_errors) > 0 else np.mean(np.abs(np.diff(y_train)))
    if scale == 0:
        return np.nan
    return np.mean(np.abs(y_true - y_pred)) / scale

# ─── Load Data ──────────────────────────────────────────────────────────
print("📊 Loading dataset...")
ds = load_dataset("HugoxBaca/bis-xtd-deriv")
train_df = ds["train"].to_pandas()
test_df = ds["test"].to_pandas()
full_df = pd.concat([train_df, test_df]).sort_values(["currency", "risk_category", "instrument", "period"])

print(f"Test period range: {test_df['period'].min()} to {test_df['period'].max()}")
print(f"Test shape: {test_df.shape}")

# ─── Load Chronos Pipeline ──────────────────────────────────────────────
print("🔄 Loading amazon/chronos-t5-small on CPU...")
pipeline = ChronosPipeline.from_pretrained(
    "amazon/chronos-t5-small",
    device_map="cpu",
)

# ─── Run Evaluation ─────────────────────────────────────────────────────
print("\n🔍 Evaluating on FX...")
fx_results = evaluate_chronos(pipeline, test_df, full_df, "FX")
print(f"FX Series evaluated: {len(fx_results)}")
if len(fx_results) > 0:
    print(f"  Seasonal Naive MASE: {fx_results['SeasonalNaive_MASE'].mean():.3f}")
    print(f"  Chronos MASE: {fx_results['Chronos_MASE'].mean():.3f}")

print("\n🔍 Evaluating on IR...")
ir_results = evaluate_chronos(pipeline, test_df, full_df, "IR")
print(f"IR Series evaluated: {len(ir_results)}")
if len(ir_results) > 0:
    print(f"  Seasonal Naive MASE: {ir_results['SeasonalNaive_MASE'].mean():.3f}")
    print(f"  Chronos MASE: {ir_results['Chronos_MASE'].mean():.3f}")

# ─── Save Results ───────────────────────────────────────────────────────
all_results = pd.concat([fx_results, ir_results])
all_results.to_csv("chronos_evaluation.csv", index=False)
fx_results.to_csv("chronos_fx_results.csv", index=False)
ir_results.to_csv("chronos_ir_results.csv", index=False)
print("\n✅ Saved → chronos_evaluation.csv")

# ─── Summary ────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("🎯 CHRONOS EVALUATION SUMMARY")
print("="*60)
if len(fx_results) > 0:
    print(f"\nFX: Chronos MASE {fx_results['Chronos_MASE'].mean():.3f} vs Seasonal Naive {fx_results['SeasonalNaive_MASE'].mean():.3f}")
    better = (fx_results['Chronos_MASE'] < fx_results['SeasonalNaive_MASE']).sum()
    print(f"  Chronos beats Naive on {better}/{len(fx_results)} series")

if len(ir_results) > 0:
    print(f"\nIR: Chronos MASE {ir_results['Chronos_MASE'].mean():.3f} vs Seasonal Naive {ir_results['SeasonalNaive_MASE'].mean():.3f}")
    better = (ir_results['Chronos_MASE'] < ir_results['SeasonalNaive_MASE']).sum()
    print(f"  Chronos beats Naive on {better}/{len(ir_results)} series")

# Save
fx_results.to_csv("chronos_fx_results.csv", index=False)
ir_results.to_csv("chronos_ir_results.csv", index=False)
print("\n✅ Saved → chronos_evaluation.csv")

print("\n" + "="*60)