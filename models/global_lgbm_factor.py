"""
Global LightGBM + Factor Model for BIS XTD
===========================================
Implements the "Statistical Ceiling" from hybrid plan:
1. Per-series log-return transformation + standardization
2. Panel PCA → forecast factors (USD, Carry, Vol)
3. Global LightGBM with rich features + currency target encoding
4. MASE metric (scale-independent)
5. Separate models for FX vs IR regimes
"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from datasets import load_dataset
from sklearn.metrics import mean_absolute_error
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
from tqdm import tqdm
import json

# ─── Load Data ──────────────────────────────────────────────────────────
print("📊 Loading dataset...")
ds = load_dataset("HugoxBaca/bis-xtd-deriv")
train_df = ds["train"].to_pandas()
test_df = ds["test"].to_pandas()
full_df = pd.concat([train_df, test_df]).sort_values(["currency", "risk_category", "instrument", "period"])

# ─── Regime Split ───────────────────────────────────────────────────────
# FX (risk_category B) vs Interest Rate (C, I, J)
fx_df = full_df[full_df["risk_category"] == "B"].copy()
ir_df = full_df[full_df["risk_category"].isin(["C", "I", "J"])].copy()

print(f"FX series: {fx_df.groupby(['currency','instrument']).ngroup().nunique()}")
print(f"IR series: {ir_df.groupby(['currency','risk_category','instrument']).ngroup().nunique()}")

HORIZON = 5  # Test has 5 months (2026-02 to 2026-06)
SEASONALITY = 12

# ─── MASE Metric ────────────────────────────────────────────────────────
def mase(y_true, y_pred, y_train, seasonality=12):
    """Mean Absolute Scaled Error - scale independent."""
    # In-sample seasonal naive MAE
    naive_errors = np.abs(y_train[seasonality:] - y_train[:-seasonality])
    scale = np.mean(naive_errors) if len(naive_errors) > 0 else np.mean(np.abs(np.diff(y_train)))
    if scale == 0:
        return np.nan
    return np.mean(np.abs(y_true - y_pred)) / scale

# ─── Seasonal Naive Baseline (per series) ──────────────────────────────
def seasonal_naive_mase(series_train, series_test, seasonality=12):
    if len(series_train) < seasonality:
        return np.nan
    pred = np.tile(series_train[-seasonality:], int(np.ceil(len(series_test)/seasonality)))[:len(series_test)]
    return mase(series_test, pred, series_train, seasonality)

# ─── Build Panel for PCA (log returns) ─────────────────────────────────
def build_panel(df, regime_name, min_obs=100):
    """Create wide panel: rows=time, cols=series."""
    # Pivot
    panel = df.pivot_table(index="period", columns=["currency", "risk_category", "instrument"], 
                           values="value", aggfunc="first").sort_index()
    panel.columns = [f"{c[0]}_{c[1]}_{c[2]}" for c in panel.columns]
    
    # Keep only series with sufficient non-zero observations
    nonzero_counts = (panel != 0).sum()
    valid_cols = nonzero_counts[nonzero_counts >= min_obs].index
    panel = panel[valid_cols]
    print(f"  Kept {len(valid_cols)} / {len(nonzero_counts)} series with >= {min_obs} non-zero obs")
    
    # Forward fill zeros
    panel_ff = panel.replace(0, np.nan).ffill().bfill()
    
    # Log returns
    log_ret = np.log(panel_ff).diff().dropna()
    
    # Standardize per series - drop columns with any NaN
    log_ret_clean = log_ret.dropna(axis=1, how='any')
    if log_ret_clean.shape[1] == 0:
        raise ValueError("No valid series after cleaning")
    
    scaler = StandardScaler()
    scaled = pd.DataFrame(scaler.fit_transform(log_ret_clean), index=log_ret_clean.index, columns=log_ret_clean.columns)
    
    return panel, log_ret_clean, scaled, scaler

print("\n📊 Building FX panel...")
fx_panel, fx_logret, fx_scaled, fx_scaler = build_panel(fx_df, "FX")
print(f"FX panel: {fx_panel.shape[0]} months × {fx_panel.shape[1]} series")

print("\n📊 Building IR panel...")
ir_panel, ir_logret, ir_scaled, ir_scaler = build_panel(ir_df, "IR")
print(f"IR panel: {ir_panel.shape[0]} months × {ir_panel.shape[1]} series")

# ─── Factor Model (PCA) ─────────────────────────────────────────────────
def fit_factor_model(scaled_panel, n_factors=3):
    """PCA on standardized log returns."""
    # Drop columns with any NaN
    clean = scaled_panel.dropna(axis=1)
    print(f"  Clean series for PCA: {clean.shape[1]} / {scaled_panel.shape[1]}")
    
    pca = PCA(n_components=min(n_factors, clean.shape[1]))
    factors = pca.fit_transform(clean)
    
    print(f"  Explained variance: {pca.explained_variance_ratio_}")
    print(f"  Cumulative: {np.cumsum(pca.explained_variance_ratio_)}")
    
    factor_df = pd.DataFrame(factors, index=clean.index, 
                             columns=[f"PC{i+1}" for i in range(pca.n_components_)])
    return pca, factor_df, clean.columns.tolist()

print("\n🔬 Fitting FX Factor Model...")
fx_pca, fx_factors, fx_series = fit_factor_model(fx_scaled, n_factors=3)

print("\n🔬 Fitting IR Factor Model...")
ir_pca, ir_factors, ir_series = fit_factor_model(ir_scaled, n_factors=5)

# ─── Forecast Factors (SARIMA/ETS) ──────────────────────────────────────
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing

def forecast_factor(factor_series, horizon=HORIZON):
    """Forecast a single factor using best of ARIMA/ETS."""
    best_mae = np.inf
    best_pred = None
    
    # ARIMA
    try:
        model = ARIMA(factor_series, order=(1,1,1)).fit()
        pred = model.forecast(horizon)
        mae = np.mean(np.abs(pred - factor_series[-horizon:].values)) if len(factor_series) >= horizon else np.inf
        if mae < best_mae:
            best_mae, best_pred = mae, pred
    except:
        pass
    
    # ETS
    try:
        model = ExponentialSmoothing(factor_series, trend="add", seasonal="add", seasonal_periods=12).fit()
        pred = model.forecast(horizon)
        mae = np.mean(np.abs(pred - factor_series[-horizon:].values)) if len(factor_series) >= horizon else np.inf
        if mae < best_mae:
            best_mae, best_pred = mae, pred
    except:
        pass
    
    # Naive fallback
    if best_pred is None:
        best_pred = np.full(horizon, factor_series.iloc[-1])
    
    return best_pred

print("\n📈 Forecasting FX factors...")
fx_factor_preds = {}
for col in fx_factors.columns:
    fx_factor_preds[col] = forecast_factor(fx_factors[col])

print("\n📈 Forecasting IR factors...")
ir_factor_preds = {}
for col in ir_factors.columns:
    ir_factor_preds[col] = forecast_factor(ir_factors[col])

# ─── Project Factor Forecasts Back to Series ────────────────────────────
def project_factors_to_series(pca, factor_preds, series_names, scaler, last_levels, regime_name="FX"):
    """Inverse transform: factors → scaled returns → log returns → levels."""
    # Reconstruct scaled returns
    factor_matrix = np.column_stack([factor_preds[c] for c in factor_preds.keys()])
    # pca.components_ shape: (n_components, n_features)
    scaled_preds = factor_matrix @ pca.components_
    
    # Inverse scale
    logret_preds = scaler.inverse_transform(scaled_preds)
    
    # Convert to levels: last_known * exp(cumsum(logret))
    preds = {}
    for i, name in enumerate(series_names):
        last = last_levels.get(name, 1.0)
        if i < logret_preds.shape[1]:
            cumret = np.exp(np.cumsum(logret_preds[:, i])) - 1
            level_pred = last * (1 + cumret)
            preds[name] = level_pred
            if regime_name == "FX":
                # Add tuple key for FX: (currency, instrument)
                parts = name.split('_')
                preds[(parts[0], parts[2])] = level_pred
            else:
                # Add tuple key for IR: (currency, risk_category, instrument)
                parts = name.split('_')
                preds[(parts[0], parts[1], parts[2])] = level_pred
    return preds

# Get last known levels for each series
fx_last = fx_panel.iloc[-1].to_dict()
ir_last = ir_panel.iloc[-1].to_dict()

print("\n🔄 Projecting FX factor forecasts...")
fx_proj = project_factors_to_series(fx_pca, fx_factor_preds, fx_series, fx_scaler, fx_last, regime_name="FX")

print("\n🔄 Projecting IR factor forecasts...")
ir_proj = project_factors_to_series(ir_pca, ir_factor_preds, ir_series, ir_scaler, ir_last, regime_name="IR")

# ─── Global LightGBM Features ───────────────────────────────────────────
def make_lgb_features(df, regime_name):
    """Create rich feature set for LightGBM."""
    df = df.sort_values(["currency", "risk_category", "instrument", "period"]).copy()
    
    # Target: log return
    df["log_val"] = np.log(df["value"].replace(0, np.nan))
    df["log_ret"] = df.groupby(["currency", "risk_category", "instrument"])["log_val"].diff()
    
    # Drop NaN
    df = df.dropna(subset=["log_ret"]).copy()
    
    # Lags
    for lag in [1, 2, 3, 6, 12, 13, 24]:
        df[f"lag_{lag}"] = df.groupby(["currency", "risk_category", "instrument"])["log_ret"].shift(lag)
    
    # Rolling stats
    for window in [3, 6, 12]:
        df[f"roll_mean_{window}"] = df.groupby(["currency", "risk_category", "instrument"])["log_ret"].transform(lambda x: x.rolling(window).mean())
        df[f"roll_std_{window}"] = df.groupby(["currency", "risk_category", "instrument"])["log_ret"].transform(lambda x: x.rolling(window).std())
    
    # Calendar
    df["month"] = pd.to_datetime(df["period"]).dt.month
    df["quarter"] = pd.to_datetime(df["period"]).dt.quarter
    df["year"] = pd.to_datetime(df["period"]).dt.year
    df["sin_month"] = np.sin(2 * np.pi * df["month"] / 12)
    df["cos_month"] = np.cos(2 * np.pi * df["month"] / 12)
    
    # Currency target encoding (mean log_ret per currency)
    curr_mean = df.groupby("currency")["log_ret"].mean()
    df["curr_te"] = df["currency"].map(curr_mean)
    
    # Risk category encoding - use integer codes
    df["risk_cat_code"] = df["risk_category"].astype("category").cat.codes
    df["instrument_code"] = df["instrument"].astype("category").cat.codes
    
    # Fill NaN in features with 0
    feat_cols_all = [c for c in df.columns if c.startswith(("lag_", "roll_", "sin_", "cos_", "curr_", "risk_", "instrument_", "month", "quarter", "year"))]
    df[feat_cols_all] = df[feat_cols_all].fillna(0)
    
    return df

print("\n🔧 Building LightGBM features for FX...")
fx_lgb = make_lgb_features(fx_df, "FX")

print("\n🔧 Building LightGBM features for IR...")
ir_lgb = make_lgb_features(ir_df, "IR")

# Feature columns (exclude target, identifiers, and non-numeric)
exclude_cols = {"value", "log_val", "log_ret", "period", "currency", "risk_category", "instrument", "location", "frequency", "measure"}
feat_cols = [c for c in fx_lgb.columns if c not in exclude_cols]
# Ensure all feature columns are numeric
feat_cols = [c for c in feat_cols if fx_lgb[c].dtype in [np.float64, np.float32, np.int64, np.int32, np.bool_]]

# Keep identifier columns for grouping
id_cols = ["currency", "risk_category", "instrument"]

# Train/test split - keep id cols
fx_train = fx_lgb[fx_lgb["period"] < "2026-01-01"]
fx_test = fx_lgb[fx_lgb["period"] >= "2026-01-01"]

ir_train = ir_lgb[ir_lgb["period"] < "2026-01-01"]
ir_test = ir_lgb[ir_lgb["period"] >= "2026-01-01"]

print(f"\nFX Train: {len(fx_train):,}, Test: {len(fx_test):,}")
print(f"IR Train: {len(ir_train):,}, Test: {len(ir_test):,}")

# ─── Train LightGBM (Direct Multi-Horizon) ──────────────────────────────
def train_lgb_multih(X_train, y_train, X_val, y_val, id_train, id_val, params=None):
    """Train one model per horizon step."""
    if params is None:
        params = {
            "objective": "regression",
            "metric": "mae",
            "learning_rate": 0.05,
            "num_leaves": 63,
            "max_depth": 8,
            "min_child_samples": 20,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.1,
            "reg_lambda": 0.1,
            "n_estimators": 500,
            "random_state": 42,
            "verbosity": -1,
            "n_jobs": -1,
        }
    
    # Reset indices once
    X_train_r = X_train.reset_index(drop=True)
    y_train_r = y_train.reset_index(drop=True)
    id_train_r = id_train.reset_index(drop=True)
    X_val_r = X_val.reset_index(drop=True)
    y_val_r = y_val.reset_index(drop=True)
    id_val_r = id_val.reset_index(drop=True)
    
    models = []
    for h in range(HORIZON):
        # Create horizon-specific target using id columns for grouping
        train_df = pd.concat([id_train_r, y_train_r], axis=1)
        y_h = train_df.groupby(id_cols)["log_ret"].shift(-h)
        valid = ~y_h.isna()
        
        dtrain = lgb.Dataset(X_train_r[valid], label=y_h[valid])
        
        val_df = pd.concat([id_val_r, y_val_r], axis=1)
        y_h_val = val_df.groupby(id_cols)["log_ret"].shift(-h)
        valid_val = ~y_h_val.isna()
        
        dval = lgb.Dataset(X_val_r[valid_val], label=y_h_val[valid_val], reference=dtrain)
        
        model = lgb.train(
            params, dtrain, 
            valid_sets=[dval], 
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
        )
        models.append(model)
    return models

# Prepare data
fx_X_train = fx_train[feat_cols].fillna(0)
fx_y_train = fx_train["log_ret"]
fx_id_train = fx_train[id_cols]
fx_X_test = fx_test[feat_cols].fillna(0)
fx_y_test = fx_test["log_ret"]
fx_id_test = fx_test[id_cols]

print("\n🌲 Training FX LightGBM...")
fx_models = train_lgb_multih(fx_X_train, fx_y_train, fx_X_test, fx_y_test, fx_id_train, fx_id_test)

# Predict
fx_lgb_preds = np.column_stack([m.predict(fx_X_test, num_iteration=m.best_iteration) for m in fx_models])

# Convert log_ret predictions back to levels
# For each series, start from last known value
fx_lgb_level_preds = {}
for (curr, inst), group in fx_test.groupby(["currency", "instrument"]):
    idx = group.index
    last_val = fx_panel.loc["2026-01", (curr, "B", inst)] if (curr, "B", inst) in fx_panel.columns else 1.0
    logret_pred = fx_lgb_preds[fx_test.index.get_indexer(idx)]
    # Cumulative
    level_pred = last_val * np.exp(np.cumsum(logret_pred, axis=1))
    fx_lgb_level_preds[(curr, inst)] = level_pred

print("✅ FX LightGBM done")

# ─── Prepare IR LightGBM ────────────────────────────────────────────────
ir_X_train = ir_train[feat_cols].fillna(0)
ir_y_train = ir_train["log_ret"]
ir_id_train = ir_train[id_cols]
ir_X_test = ir_test[feat_cols].fillna(0)
ir_y_test = ir_test["log_ret"]
ir_id_test = ir_test[id_cols]

print("\n🌲 Training IR LightGBM...")
ir_models = train_lgb_multih(ir_X_train, ir_y_train, ir_X_test, ir_y_test, ir_id_train, ir_id_test)

# Predict
ir_lgb_preds = np.column_stack([m.predict(ir_X_test, num_iteration=m.best_iteration) for m in ir_models])

# Convert log_ret predictions back to levels for IR
ir_lgb_level_preds = {}
for (curr, risk_cat, inst), group in ir_test.groupby(["currency", "risk_category", "instrument"]):
    idx = group.index
    key = (curr, risk_cat, inst)
    last_val = ir_panel.loc["2026-01", key] if key in ir_panel.columns else 1.0
    logret_pred = ir_lgb_preds[ir_test.index.get_indexer(idx)]
    level_pred = last_val * np.exp(np.cumsum(logret_pred, axis=1))
    ir_lgb_level_preds[key] = level_pred

print("✅ IR LightGBM done")

# ─── Evaluate All Methods ───────────────────────────────────────────────
def evaluate_regime(test_df, proj_preds, lgb_level_preds, regime_name):
    """Compute MASE for all methods."""
    results = []
    
    for (currency, risk_cat, inst), group in test_df.groupby(["currency", "risk_category", "instrument"]):
        if regime_name == "FX":
            key = (currency, inst)
        else:  # IR
            key = (currency, risk_cat, inst)
            
        true = group.sort_values("period")["value"].values[:HORIZON]
        if len(true) < HORIZON:
            continue
        
        # Get training data for this series
        train_series = full_df[(full_df["currency"]==currency) & (full_df["risk_category"]==risk_cat) & (full_df["instrument"]==inst)].sort_values("period")
        train_vals = train_series[train_series["period"] < "2026-01-01"]["value"].values
        if len(train_vals) < SEASONALITY:
            continue
        
        # Seasonal Naive
        sn_mase = seasonal_naive_mase(train_vals, true)
        
        # Factor model
        if key in proj_preds:
            fm_pred = proj_preds[key][:HORIZON]
            fm_mase = mase(true, fm_pred, train_vals)
        else:
            fm_mase = np.nan
        
        # LightGBM
        if key in lgb_level_preds:
            lgb_pred = lgb_level_preds[key][0][:HORIZON]
            lgb_mase = mase(true, lgb_pred, train_vals)
        else:
            lgb_mase = np.nan
        
        result = {
            "currency": currency,
            "instrument": inst,
            "regime": regime_name,
            "SeasonalNaive_MASE": sn_mase,
            "FactorModel_MASE": fm_mase,
            "LightGBM_MASE": lgb_mase,
            "true_mean": np.mean(true),
        }
        if regime_name == "IR":
            result["risk_category"] = risk_cat
        results.append(result)
    
    return pd.DataFrame(results)

print("\n📊 Evaluating FX...")
fx_eval = evaluate_regime(fx_test, fx_proj, fx_lgb_level_preds, "FX")

print(f"\n📊 FX Results (MASE - lower is better):")
print(fx_eval.columns.tolist())
print(fx_eval[["SeasonalNaive_MASE", "FactorModel_MASE", "LightGBM_MASE"]].mean())
print(fx_eval.groupby("instrument")[["SeasonalNaive_MASE", "FactorModel_MASE", "LightGBM_MASE"]].mean())

# Save
fx_eval.to_csv("fx_global_model_results.csv", index=False)
print("\n✅ Saved → fx_global_model_results.csv")

# ─── Evaluate IR Regime ─────────────────────────────────────────────────
print("\n📊 Evaluating IR...")

# Need to create IR LightGBM level predictions
ir_lgb_preds = np.column_stack([m.predict(ir_X_test, num_iteration=m.best_iteration) for m in ir_models])

# Convert log_ret predictions back to levels for IR
ir_lgb_level_preds = {}
for (curr, risk_cat, inst), group in ir_test.groupby(["currency", "risk_category", "instrument"]):
    idx = group.index
    key = (curr, risk_cat, inst)
    last_val = ir_panel.loc["2026-01", key] if key in ir_panel.columns else 1.0
    logret_pred = ir_lgb_preds[ir_test.index.get_indexer(idx)]
    level_pred = last_val * np.exp(np.cumsum(logret_pred, axis=1))
    ir_lgb_level_preds[key] = level_pred

print("\n🌲 Training IR LightGBM...")
ir_models = train_lgb_multih(ir_X_train, ir_y_train, ir_X_test, ir_y_test, ir_id_train, ir_id_test)

print("✅ IR LightGBM done")

# Evaluate IR
ir_eval = evaluate_regime(ir_test, ir_proj, ir_lgb_level_preds, "IR")

print(f"\n📊 IR Results (MASE - lower is better):")
print(ir_eval.columns.tolist())
print(ir_eval[["SeasonalNaive_MASE", "FactorModel_MASE", "LightGBM_MASE"]].mean())
print(ir_eval.groupby("risk_category")[["SeasonalNaive_MASE", "FactorModel_MASE", "LightGBM_MASE"]].mean())

# Save
ir_eval.to_csv("ir_global_model_results.csv", index=False)
print("\n✅ Saved → ir_global_model_results.csv")

# ─── Combined Summary ───────────────────────────────────────────────────
print("\n" + "="*60)
print("🎯 GLOBAL MODEL SUMMARY")
print("="*60)
print(f"FX Series evaluated: {len(fx_eval)}")
print(f"  Mean MASE - SeasonalNaive: {fx_eval['SeasonalNaive_MASE'].mean():.3f}")
print(f"  Mean MASE - FactorModel:   {fx_eval['FactorModel_MASE'].mean():.3f}")
print(f"  Mean MASE - LightGBM:      {fx_eval['LightGBM_MASE'].mean():.3f}")

print(f"\nIR Series evaluated: {len(ir_eval)}")
print(f"  Mean MASE - SeasonalNaive: {ir_eval['SeasonalNaive_MASE'].mean():.3f}")
print(f"  Mean MASE - FactorModel:   {ir_eval['FactorModel_MASE'].mean():.3f}")
print(f"  Mean MASE - LightGBM:      {ir_eval['LightGBM_MASE'].mean():.3f}")

# Best model distribution
all_eval = pd.concat([fx_eval, ir_eval])
best_all = all_eval[["SeasonalNaive_MASE", "FactorModel_MASE", "LightGBM_MASE"]].idxmin(axis=1).value_counts()
print(f"\nBest model per series (all):\n{best_all}")

# Save combined
all_eval.to_csv("global_model_results.csv", index=False)
print("\n✅ Saved → global_model_results.csv")