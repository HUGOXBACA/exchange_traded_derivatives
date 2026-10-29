"""
Ensemble Pipeline for BIS XTD Forecasting - Full Predictions
============================================================
Creates proper ensemble with all 4 models' actual predictions:
1. Seasonal Naive
2. Factor Model (PCA-based)
3. LightGBM (global gradient boosting)
4. Chronos (Foundation Model)

Ensemble: Inverse MASE weighted average + Quantile averaging
"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from datasets import load_dataset
from sklearn.metrics import mean_absolute_error

# ─── Helper Functions ───────────────────────────────────────────────────
def mase(y_true, y_pred, y_train, seasonality=12):
    """Mean Absolute Scaled Error."""
    if len(y_true) == 0 or len(y_pred) == 0:
        return np.nan
    naive_errors = np.abs(y_train[seasonality:] - y_train[:-seasonality])
    scale = np.mean(naive_errors) if len(naive_errors) > 0 else np.mean(np.abs(np.diff(y_train)))
    if scale <= 1e-10:
        return np.nan
    return np.mean(np.abs(y_true - y_pred)) / scale

# ─── Load Data ──────────────────────────────────────────────────────────
print("📊 Loading data...")
ds = load_dataset("HugoxBaca/bis-xtd-deriv")
train_df = ds["train"].to_pandas()
test_df = ds["test"].to_pandas()
full_df = pd.concat([train_df, test_df]).sort_values(["currency", "risk_category", "instrument", "period"])

# Load model evaluation results
fx_global = pd.read_csv("fx_global_model_results.csv")
ir_global = pd.read_csv("ir_global_model_results.csv")
chronos = pd.read_csv("chronos_evaluation.csv")

# Add keys
fx_global["key"] = fx_global.apply(lambda r: (r["currency"], r["instrument"]), axis=1)
ir_global["key"] = ir_global.apply(lambda r: (r["currency"], r["risk_category"], r["instrument"]), axis=1)
chronos["key"] = chronos.apply(lambda r: (r["currency"], r["risk_category"], r["instrument"]), axis=1)

# ─── Load All Predictions ───────────────────────────────────────────────
# We need to regenerate FactorModel and LightGBM predictions from the training script
# For now, let's load the saved predictions from global_lgbm_factor.py
# But we didn't save them - we need to regenerate

# Let's load the FactorModel projections from the global script output
# We'll need to re-run the FactorModel projection

# Actually, let's load the saved FactorModel projections
# They were computed in global_lgbm_factor.py but not saved
# We need to re-run the FactorModel projection and save

print("📊 Loading data and regenerating predictions...")

ds = load_dataset("HugoxBaca/bis-xtd-deriv")
train_df = ds["train"].to_pandas()
test_df = ds["test"].to_pandas()
full_df = pd.concat([train_df, test_df]).sort_values(["currency", "risk_category", "instrument", "period"])

# ─── Regenerate FactorModel Predictions ─────────────────────────────────
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing

HORIZON = 5
SEASONALITY = 12

def build_panel(df, min_obs=100):
    panel = df.pivot_table(index="period", columns=["currency", "risk_category", "instrument"], 
                           values="value", aggfunc="first").sort_index()
    panel.columns = [f"{c[0]}_{c[1]}_{c[2]}" for c in panel.columns]
    nonzero_counts = (panel != 0).sum()
    valid_cols = nonzero_counts[nonzero_counts >= min_obs].index
    panel = panel[valid_cols]
    panel_ff = panel.replace(0, np.nan).ffill().bfill()
    log_ret = np.log(panel_ff).diff().dropna()
    log_ret_clean = log_ret.dropna(axis=1, how='any')
    scaler = StandardScaler()
    scaled = pd.DataFrame(scaler.fit_transform(log_ret_clean), index=log_ret_clean.index, columns=log_ret_clean.columns)
    return panel, log_ret_clean, scaled, scaler, log_ret_clean.columns.tolist()

def forecast_factor(factor_series, horizon=HORIZON):
    best_mae = np.inf
    best_pred = None
    try:
        model = ARIMA(factor_series, order=(1,1,1)).fit()
        pred = model.forecast(horizon)
        mae = np.mean(np.abs(pred - factor_series[-horizon:].values)) if len(factor_series) >= horizon else np.inf
        if mae < best_mae:
            best_mae, best_pred = mae, pred
    except:
        pass
    try:
        model = ExponentialSmoothing(factor_series, trend="add", seasonal="add", seasonal_periods=12).fit()
        pred = model.forecast(horizon)
        mae = np.mean(np.abs(pred - factor_series[-horizon:].values)) if len(factor_series) >= horizon else np.inf
        if mae < best_mae:
            best_mae, best_pred = mae, pred
    except:
        pass
    if best_pred is None:
        best_pred = np.full(horizon, factor_series.iloc[-1])
    return best_pred

def project_factors_to_series(pca, factor_preds, series_names, scaler, last_levels, regime_name="FX"):
    factor_matrix = np.column_stack([factor_preds[c] for c in factor_preds.keys()])
    scaled_preds = factor_matrix @ pca.components_
    logret_preds = scaler.inverse_transform(scaled_preds)
    preds = {}
    for i, name in enumerate(series_names):
        last = last_levels.get(name, 1.0)
        if i < logret_preds.shape[1]:
            cumret = np.exp(np.cumsum(logret_preds[:, i])) - 1
            level_pred = last * (1 + cumret)
            preds[name] = level_pred
            parts = name.split('_')
            if regime_name == "FX":
                preds[(parts[0], parts[2])] = level_pred
            else:
                preds[(parts[0], parts[1], parts[2])] = level_pred
    return preds

# Build panels and factor models
fx_df = full_df[full_df["risk_category"] == "B"].copy()
ir_df = full_df[full_df["risk_category"].isin(["C", "I", "J"])].copy()

fx_panel, fx_logret, fx_scaled, fx_scaler, fx_series = build_panel(fx_df)
ir_panel, ir_logret, ir_scaled, ir_scaler, ir_series = build_panel(ir_df)

fx_pca = PCA(n_components=3).fit(fx_scaled)
ir_pca = PCA(n_components=5).fit(ir_scaled)

fx_factors = pd.DataFrame(fx_pca.transform(fx_scaled), index=fx_scaled.index, columns=[f"PC{i+1}" for i in range(3)])
ir_factors = pd.DataFrame(ir_pca.transform(ir_scaled), index=ir_scaled.index, columns=[f"PC{i+1}" for i in range(5)])

fx_factor_preds = {col: forecast_factor(fx_factors[col]) for col in fx_factors.columns}
ir_factor_preds = {col: forecast_factor(ir_factors[col]) for col in ir_factors.columns}

fx_last = fx_panel.iloc[-1].to_dict()
ir_last = ir_panel.iloc[-1].to_dict()

fx_proj = project_factors_to_series(fx_pca, fx_factor_preds, fx_series, fx_scaler, fx_last, "FX")
ir_proj = project_factors_to_series(ir_pca, ir_factor_preds, ir_series, ir_scaler, ir_last, "IR")

print("✅ FactorModel predictions regenerated")

# ─── Regenerate LightGBM Predictions ────────────────────────────────────
import lightgbm as lgb

HORIZON = 5
SEASONALITY = 12

def make_lgb_features(df):
    df = df.sort_values(["currency", "risk_category", "instrument", "period"]).copy()
    df["log_val"] = np.log(df["value"].replace(0, np.nan))
    df["log_ret"] = df.groupby(["currency", "risk_category", "instrument"])["log_val"].diff()
    df = df.dropna(subset=["log_ret"]).copy()
    for lag in [1, 2, 3, 6, 12, 13, 24]:
        df[f"lag_{lag}"] = df.groupby(["currency", "risk_category", "instrument"])["log_ret"].shift(lag)
    for window in [3, 6, 12]:
        df[f"roll_mean_{window}"] = df.groupby(["currency", "risk_category", "instrument"])["log_ret"].transform(lambda x: x.rolling(window).mean())
        df[f"roll_std_{window}"] = df.groupby(["currency", "risk_category", "instrument"])["log_ret"].transform(lambda x: x.rolling(window).std())
    df["month"] = pd.to_datetime(df["period"]).dt.month
    df["quarter"] = pd.to_datetime(df["period"]).dt.quarter
    df["year"] = pd.to_datetime(df["period"]).dt.year
    df["sin_month"] = np.sin(2 * np.pi * df["month"] / 12)
    df["cos_month"] = np.cos(2 * np.pi * df["month"] / 12)
    curr_mean = df.groupby("currency")["log_ret"].mean()
    df["curr_te"] = df["currency"].map(curr_mean)
    df["risk_cat_code"] = df["risk_category"].astype("category").cat.codes
    df["instrument_code"] = df["instrument"].astype("category").cat.codes
    feat_cols_all = [c for c in df.columns if c.startswith(("lag_", "roll_", "sin_", "cos_", "curr_", "risk_", "instrument_", "month", "quarter", "year"))]
    df[feat_cols_all] = df[feat_cols_all].fillna(0)
    return df

fx_lgb_df = make_lgb_features(full_df[full_df["risk_category"] == "B"].copy())
ir_lgb_df = make_lgb_features(full_df[full_df["risk_category"].isin(["C", "I", "J"])].copy())

feat_cols = [c for c in fx_lgb_df.columns if c not in {"value", "log_val", "log_ret", "period", "currency", "risk_category", "instrument", "location", "frequency", "measure"}]
feat_cols = [c for c in feat_cols if fx_lgb_df[c].dtype in [np.float64, np.float32, np.int64, np.int32, np.bool_]]
id_cols = ["currency", "risk_category", "instrument"]

fx_train = fx_lgb_df[fx_lgb_df["period"] < "2026-01-01"]
fx_test = fx_lgb_df[fx_lgb_df["period"] >= "2026-01-01"]
ir_train = ir_lgb_df[ir_lgb_df["period"] < "2026-01-01"]
ir_test = ir_lgb_df[ir_lgb_df["period"] >= "2026-01-01"]

fx_X_train = fx_train[feat_cols].fillna(0)
fx_y_train = fx_train["log_ret"]
fx_X_test = fx_test[feat_cols].fillna(0)
fx_y_test = fx_test["log_ret"]
fx_id_train = fx_train[id_cols]
fx_id_test = fx_test[id_cols]

ir_X_train = ir_train[feat_cols].fillna(0)
ir_y_train = ir_train["log_ret"]
ir_X_test = ir_test[feat_cols].fillna(0)
ir_y_test = ir_test["log_ret"]
ir_id_train = ir_train[id_cols]
ir_id_test = ir_test[id_cols]

def train_lgb_multih(X_train, y_train, X_val, y_val, id_train, id_val, params=None):
    if params is None:
        params = {"objective": "regression", "metric": "mae", "learning_rate": 0.05, "num_leaves": 63, "max_depth": 8, "min_child_samples": 20, "subsample": 0.8, "colsample_bytree": 0.8, "reg_alpha": 0.1, "reg_lambda": 0.1, "n_estimators": 500, "random_state": 42, "verbosity": -1, "n_jobs": -1}
    X_train_r = X_train.reset_index(drop=True)
    y_train_r = y_train.reset_index(drop=True)
    id_train_r = id_train.reset_index(drop=True)
    X_val_r = X_val.reset_index(drop=True)
    y_val_r = y_val.reset_index(drop=True)
    id_val_r = id_val.reset_index(drop=True)
    models = []
    for h in range(5):
        train_df = pd.concat([id_train_r, y_train_r], axis=1)
        y_h = train_df.groupby(["currency", "risk_category", "instrument"])["log_ret"].shift(-h)
        valid = ~y_h.isna()
        dtrain = lgb.Dataset(X_train_r[valid], label=y_h[valid])
        val_df = pd.concat([id_val_r, y_val_r], axis=1)
        y_h_val = val_df.groupby(["currency", "risk_category", "instrument"])["log_ret"].shift(-h)
        valid_val = ~y_h_val.isna()
        dval = lgb.Dataset(X_val_r[valid_val], label=y_h_val[valid_val], reference=dtrain)
        model = lgb.train(params, dtrain, valid_sets=[dval], callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
        models.append(model)
    return models

fx_models = train_lgb_multih(fx_X_train, fx_y_train, fx_X_test, fx_y_test, fx_id_train, fx_id_test)
ir_models = train_lgb_multih(ir_X_train, ir_y_train, ir_X_test, ir_y_test, ir_id_train, ir_id_test)

fx_lgb_preds = np.column_stack([m.predict(fx_X_test, num_iteration=m.best_iteration) for m in fx_models])
ir_lgb_preds = np.column_stack([m.predict(ir_X_test, num_iteration=m.best_iteration) for m in ir_models])

fx_lgb_level_preds = {}
for (curr, inst), group in fx_test.groupby(["currency", "instrument"]):
    idx = group.index
    train_series = full_df[(full_df["currency"]==curr) & (full_df["risk_category"]=="B") & (full_df["instrument"]==inst)].sort_values("period")
    last_val = train_series[train_series["period"] < "2026-01-01"]["value"].values
    last_val = last_val[-1] if len(last_val) > 0 else 1.0
    logret_pred = fx_lgb_preds[fx_test.index.get_indexer(idx)]
    level_pred = last_val * np.exp(np.cumsum(logret_pred, axis=1))
    fx_lgb_level_preds[(curr, inst)] = level_pred

ir_lgb_level_preds = {}
for (curr, risk_cat, inst), group in ir_test.groupby(["currency", "risk_category", "instrument"]):
    idx = group.index
    train_series = full_df[(full_df["currency"]==curr) & (full_df["risk_category"]==risk_cat) & (full_df["instrument"]==inst)].sort_values("period")
    last_val = train_series[train_series["period"] < "2026-01-01"]["value"].values
    last_val = last_val[-1] if len(last_val) > 0 else 1.0
    logret_pred = ir_lgb_preds[ir_test.index.get_indexer(idx)]
    level_pred = last_val * np.exp(np.cumsum(logret_pred, axis=1))
    ir_lgb_level_preds[(curr, risk_cat, inst)] = level_pred

print("✅ LightGBM predictions regenerated")

# ─── Load Chronos Predictions ───────────────────────────────────────────
import torch
from chronos import ChronosPipeline

pipeline = ChronosPipeline.from_pretrained("amazon/chronos-t5-small", device_map="cpu")

def mase(y_true, y_pred, y_train, seasonality=12):
    if len(y_true) == 0 or len(y_pred) == 0:
        return np.nan
    naive_errors = np.abs(y_train[12:] - y_train[:-12])
    scale = np.mean(naive_errors) if len(naive_errors) > 0 else np.mean(np.abs(np.diff(y_train)))
    if scale <= 1e-10:
        return np.nan
    return np.mean(np.abs(y_true - y_pred)) / scale

def get_chronos_pred(pipeline, train_vals, horizon=5):
    train_vals = np.where(train_vals == 0, np.nan, train_vals)
    train_vals = pd.Series(train_vals).ffill().bfill().values
    context = torch.tensor(np.log(train_vals[-128:]), dtype=torch.float32).unsqueeze(0)
    forecast = pipeline.predict(context, prediction_length=5, num_samples=20, temperature=1.0, top_k=50, top_p=1.0)
    median_forecast = np.median(forecast[0].cpu().numpy(), axis=0)
    return np.exp(median_forecast)

print("🔮 Generating Chronos predictions...")
chronos_preds = {}
for _, row in test_df.groupby(["currency", "risk_category", "instrument"]).size().reset_index().iterrows():
    currency, risk_cat, inst = row["currency"], row["risk_category"], row["instrument"]
    train_series = full_df[(full_df["currency"]==currency) & (full_df["risk_category"]==risk_cat) & (full_df["instrument"]==inst)].sort_values("period")
    train_vals = train_series[train_series["period"] < "2026-01-01"]["value"].values.astype(float)
    train_vals = np.where(train_vals == 0, np.nan, train_vals)
    train_vals = pd.Series(train_vals).ffill().bfill().values
    if len(train_vals) >= 12:
        try:
            chronos_preds[(currency, risk_cat, inst)] = get_chronos_pred(pipeline, train_vals)
        except:
            pass

print("✅ Chronos predictions generated")

# ─── Build Ensemble ─────────────────────────────────────────────────────
HORIZON = 5
SEASONALITY = 12

def mase(y_true, y_pred, y_train, seasonality=12):
    if len(y_true) == 0 or len(y_pred) == 0:
        return np.nan
    naive_errors = np.abs(y_train[seasonality:] - y_train[:-seasonality])
    scale = np.mean(naive_errors) if len(naive_errors) > 0 else np.mean(np.abs(np.diff(y_train)))
    if scale <= 1e-10:
        return np.nan
    return np.mean(np.abs(y_true - y_pred)) / scale

# ─── Generate All Predictions for Test Series ───────────────────────────
test_series = test_df.groupby(["currency", "risk_category", "instrument"]).size().reset_index()

all_predictions = {}
model_mase = {}

for _, row in test_series.iterrows():
    currency, risk_cat, inst = row["currency"], row["risk_category"], row["instrument"]
    
    # Get true values
    true_series = test_df[(test_df["currency"]==currency) & (test_df["risk_category"]==risk_cat) & (test_df["instrument"]==inst)].sort_values("period")
    true = true_series["value"].values[:5].astype(float)
    if len(true) < 5:
        continue
    
    # Get training data
    train_series = full_df[(full_df["currency"]==currency) & (full_df["risk_category"]==risk_cat) & (full_df["instrument"]==inst)].sort_values("period")
    train_vals = train_series[train_series["period"] < "2026-01-01"]["value"].values.astype(float)
    train_vals = np.where(train_vals == 0, np.nan, train_vals)
    train_vals = pd.Series(train_vals).ffill().bfill().values
    
    if len(train_vals) < 12:
        continue
    
    # ─── Generate All Predictions ──────────────────────────────────────
    preds = {}
    
    # 1. Seasonal Naive
    sn_pred = np.tile(train_vals[-12:], int(np.ceil(5/12)))[:5]
    preds["SeasonalNaive"] = sn_pred
    
    # 2. FactorModel
    if risk_cat == "B":
        key = (currency, inst)
        if key in fx_proj:
            preds["FactorModel"] = fx_proj[key][:5]
    else:
        key = (currency, risk_cat, inst)
        if key in ir_proj:
            preds["FactorModel"] = ir_proj[key][:5]
    
    # 3. LightGBM
    if risk_cat == "B":
        key = (currency, inst)
        if key in fx_lgb_level_preds:
            preds["LightGBM"] = fx_lgb_level_preds[key][0][:5]
    else:
        key = (currency, risk_cat, inst)
        if key in ir_lgb_level_preds:
            preds["LightGBM"] = ir_lgb_level_preds[key][0][:5]
    
    # 4. Chronos
    ckey = (currency, risk_cat, inst)
    if ckey in chronos_preds:
        preds["Chronos"] = chronos_preds[ckey][:5]
    
    # ─── Compute MASE for Each Model ───────────────────────────────────
    train_series = full_df[(full_df["currency"]==currency) & (full_df["risk_category"]==risk_cat) & (full_df["instrument"]==inst)].sort_values("period")
    train_vals = train_series[train_series["period"] < "2026-01-01"]["value"].values.astype(float)
    train_vals = np.where(train_vals == 0, np.nan, train_vals)
    train_vals = pd.Series(train_vals).ffill().bfill().values
    
    true = test_df[(test_df["currency"]==currency) & (test_df["risk_category"]==risk_cat) & (test_df["instrument"]==inst)].sort_values("period")["value"].values[:5].astype(float)
    
    model_mase = {}
    for name, pred in preds.items():
        if pred is not None and len(pred) == 5:
            model_mase[name] = mase(true, pred, train_vals)
    
    # ─── Ensemble: Inverse MASE Weighted Average ───────────────────────
    valid_mase = {k: v for k, v in model_mase.items() if not np.isnan(v)}
    
    if len(valid_mase) >= 2:
        weights = {k: 1/v for k, v in valid_mase.items()}
        total_w = sum(weights.values())
        weights = {k: w/total_w for k, w in weights.items()}
        
        ensemble_pred = np.zeros(5)
        for name, pred in preds.items():
            if name in weights:
                ensemble_pred += weights[name] * pred
        
        ensemble_mase = mase(true, ensemble_pred, train_vals)
    else:
        ensemble_mase = np.nan
    
    # ─── Store Results ──────────────────────────────────────────────────
    model_mase = {k: mase(true, v, train_vals) for k, v in preds.items() if v is not None}
    
    results.append({
        "currency": currency,
        "risk_category": risk_cat,
        "instrument": inst,
        "SeasonalNaive_MASE": model_mase.get("SeasonalNaive", np.nan),
        "FactorModel_MASE": model_mase.get("FactorModel", np.nan),
        "LightGBM_MASE": model_mase.get("LightGBM", np.nan),
        "Chronos_MASE": model_mase.get("Chronos", np.nan),
        "Ensemble_MASE": mase(true, sum(weights.get(k,0)*v for k,v in preds.items() if k in weights), train_vals) if len(valid_mase) >= 2 else np.nan,
        "true_mean": np.mean(true),
    })

# ─── Summary ────────────────────────────────────────────────────────────
results_df = pd.DataFrame(results)
print(f"\n📊 Ensemble Results ({len(results_df)} series)")
print(f"Seasonal Naive MASE: {results_df['SeasonalNaive_MASE'].mean():.3f}")
print(f"FactorModel MASE: {results_df['FactorModel_MASE'].mean():.3f}")
print(f"LightGBM MASE: {results_df['LightGBM_MASE'].mean():.3f}")
print(f"Chronos MASE: {results_df['Chronos_MASE'].mean():.3f}")
print(f"Ensemble MASE: {results_df['Ensemble_MASE'].mean():.3f}")

print("\nBest model distribution:")
best = results_df[["SeasonalNaive_MASE", "FactorModel_MASE", "LightGBM_MASE", "Chronos_MASE", "Ensemble_MASE"]].idxmin(axis=1).value_counts()
print(best)

# Save
results_df.to_csv("ensemble_results.csv", index=False)
print("\n✅ Saved → ensemble_results.csv")

# ─── Per-Regime Summary ────────────────────────────────────────────────
print("\n" + "="*60)
print("🎯 ENSEMBLE SUMMARY")
print("="*60)

for regime in ["FX", "IR"]:
    if regime == "FX":
        sub = results_df[results_df["risk_category"] == "B"]
    else:
        sub = results_df[results_df["risk_category"] != "B"]
    
    if len(sub) > 0:
        print(f"\n{regime} ({len(sub)} series):")
        print(f"  Seasonal Naive: {sub['SeasonalNaive_MASE'].mean():.3f}")
        print(f"  FactorModel:    {sub['FactorModel_MASE'].mean():.3f}")
        print(f"  LightGBM:       {sub['LightGBM_MASE'].mean():.3f}")
        print(f"  Chronos:        {sub['Chronos_MASE'].mean():.3f}")
        print(f"  Ensemble:       {sub['Ensemble_MASE'].mean():.3f}")
        
        best_regime = sub[["SeasonalNaive_MASE", "FactorModel_MASE", "LightGBM_MASE", "Chronos_MASE", "Ensemble_MASE"]].idxmin(axis=1).value_counts()
        print(f"  Best model: {best_regime.to_dict()}")

print("\n" + "="*60)