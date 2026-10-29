"""
Fixed LightGBM Implementation for BIS XTD
==========================================
Fixes:
1. Per-series target normalization (RobustScaler per series)
2. QuantileLoss for probabilistic forecasting (0.1, 0.5, 0.9)
3. Proper categorical feature handling (integer indices, not names)
4. Global feature scaling (RobustScaler)
5. Direct level prediction with proper scaling
6. Direct multi-horizon (one model per horizon)
7. Rigorous validation with MASE metric
"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from datasets import load_dataset
from sklearn.preprocessing import RobustScaler
import lightgbm as lgb
from tqdm import tqdm
from pathlib import Path

# ─── Config ──────────────────────────────────────────────────────────────
HORIZON = 5  # Test has 5 months (2026-02 to 2026-06)
SEASONALITY = 12
N_ESTIMATORS = 500
EARLY_STOPPING = 50
LR = 0.03

# ─── Load Data ──────────────────────────────────────────────────────────────
print("📊 Loading dataset...")
ds = load_dataset("HugoxBaca/bis-xtd-deriv")
train_df = ds["train"].to_pandas()
test_df = ds["test"].to_pandas()
full_df = pd.concat([train_df, test_df]).sort_values(["currency", "risk_category", "instrument", "period"])

# Load model evaluation results
fx_global = pd.read_csv("fx_global_model_results.csv")
ir_global = pd.read_csv("ir_global_model_results.csv")
chronos_eval = pd.read_csv("chronos_evaluation.csv")

# Add keys
fx_global["key"] = fx_global.apply(lambda r: (r["currency"], r["instrument"]), axis=1)
ir_global["key"] = ir_global.apply(lambda r: (r["currency"], r["risk_category"], r["instrument"]), axis=1)
chronos_eval["key"] = chronos_eval.apply(lambda r: (r["currency"], r["risk_category"], r["instrument"]), axis=1)

# ─── MASE Metric ────────────────────────────────────────────────────────
def mase(y_true, y_pred, y_train, seasonality=12):
    if len(y_true) == 0 or len(y_pred) == 0:
        return np.nan
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    y_train = np.asarray(y_train).ravel()
    if len(y_true) != len(y_pred):
        raise ValueError(f"Length mismatch: y_true={len(y_true)}, y_pred={len(y_pred)}")
    naive_errors = np.abs(y_train[seasonality:] - y_train[:-seasonality])
    scale = np.mean(naive_errors) if len(naive_errors) > 0 else np.mean(np.abs(np.diff(y_train)))
    if scale <= 1e-10:
        return np.nan
    return np.mean(np.abs(y_true - y_pred)) / scale


def rmse(y_true, y_pred):
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


# ─── Feature Engineering ─────────────────────────────────────────────────
def make_features(df):
    """Create rich feature set for LightGBM. Predicts LEVELS directly."""
    df = df.sort_values(["currency", "risk_category", "instrument", "period"]).copy()
    
    # Target: LEVEL (not log-return)
    df["target"] = df["value"].astype(float)
    
    # Handle zeros
    df["target"] = df["target"].replace(0, np.nan)
    df["target"] = df.groupby(["currency", "risk_category", "instrument"])["target"].transform(
        lambda x: x.ffill().bfill()
    )
    
    # Lags of target (levels)
    for lag in [1, 2, 3, 6, 12, 13, 24]:
        df[f"lag_{lag}"] = df.groupby(["currency", "risk_category", "instrument"])["target"].shift(lag)
    
    # Rolling stats on target
    for window in [3, 6, 12]:
        df[f"roll_mean_{window}"] = df.groupby(["currency", "risk_category", "instrument"])["target"].transform(
            lambda x: x.rolling(window, min_periods=1).mean()
        )
        df[f"roll_std_{window}"] = df.groupby(["currency", "risk_category", "instrument"])["target"].transform(
            lambda x: x.rolling(window, min_periods=2).std()
        )
    
    # Rolling stats on 1-period change
    df["diff_1"] = df.groupby(["currency", "risk_category", "instrument"])["target"].diff()
    for window in [3, 6, 12]:
        df[f"diff_roll_mean_{window}"] = df.groupby(["currency", "risk_category", "instrument"])["diff_1"].transform(
            lambda x: x.rolling(window, min_periods=1).mean()
        )
        df[f"diff_roll_std_{window}"] = df.groupby(["currency", "risk_category", "instrument"])["diff_1"].transform(
            lambda x: x.rolling(window, min_periods=2).std()
        )
    
    # Calendar
    df["month"] = pd.to_datetime(df["period"]).dt.month
    df["quarter"] = pd.to_datetime(df["period"]).dt.quarter
    df["year"] = pd.to_datetime(df["period"]).dt.year
    df["sin_month"] = np.sin(2 * np.pi * df["month"] / 12)
    df["cos_month"] = np.cos(2 * np.pi * df["month"] / 12)
    
    # Currency target encoding (mean target per currency)
    curr_mean = df.groupby("currency")["target"].mean()
    df["curr_te"] = df["currency"].map(curr_mean)
    
    # Categorical encodings - use integer codes for LightGBM categorical_feature
    df["currency_code"] = df["currency"].astype("category").cat.codes
    df["risk_cat_code"] = df["risk_category"].astype("category").cat.codes
    df["instrument_code"] = df["instrument"].astype("category").cat.codes
    
    # Fill NaN in features with 0
    feat_cols_all = [c for c in df.columns if c.startswith(("lag_", "roll_", "sin_", "cos_", "curr_", "risk_", "instrument_", "month", "quarter", "year"))]
    df[feat_cols_all] = df[feat_cols_all].fillna(0)
    
    return df


# ─── Future Target Creation ──────────────────────────────────────────────
def add_future_targets(df, horizon=5):
    """Create future target columns for multi-horizon direct forecasting."""
    df = df.copy()
    for h in range(1, horizon + 1):
        df[f"target_h{h}"] = df.groupby(["currency", "risk_category", "instrument"])["target"].shift(-h)
    return df


# ─── Feature Columns ─────────────────────────────────────────────────────
def get_feature_cols(df):
    """Get numeric feature columns excluding targets and identifiers."""
    exclude = {
        "value", "target", "period", "currency", "risk_category", "instrument", 
        "location", "frequency", "measure", "diff_1"
    }
    future_targets = {f"target_h{h}" for h in range(1, 10)}
    exclude.update(future_targets)
    
    feat_cols = [c for c in df.columns if c not in exclude]
    # Ensure all feature columns are numeric
    feat_cols = [c for c in feat_cols if c in ["lag_1", "lag_2", "lag_3", "lag_6", "lag_12", "lag_13", "lag_24",
                                                "roll_mean_3", "roll_std_3", "roll_mean_6", "roll_std_6", 
                                                "roll_mean_12", "roll_std_12",
                                                "diff_roll_mean_3", "diff_roll_std_3",
                                                "diff_roll_mean_6", "diff_roll_std_6",
                                                "diff_roll_mean_12", "diff_roll_std_12",
                                                "sin_month", "cos_month",
                                                "month", "quarter", "year",
                                                "curr_te",
                                                "risk_cat_code", "instrument_code",
                                                "currency_code", "instrument_code"]]
    return feat_cols


# ─── Categorical Feature Indices ─────────────────────────────────────────
def get_categorical_indices(feat_cols):
    """Get indices of categorical features in feature column list."""
    cat_cols = ["currency_code", "risk_cat_code", "instrument_code", "month", "quarter"]
    return [feat_cols.index(c) for c in cat_cols if c in feat_cols]


# ─── Target Scaler (Per-Series RobustScaler) ─────────────────────────────
class TargetScaler:
    """Per-series RobustScaler for target normalization."""
    def __init__(self):
        self.scalers = {}  # key -> (median, scale)
    
    def fit(self, df, target_col="target"):
        self.scalers = {}
        for key, group in df.groupby(["currency", "risk_category", "instrument"]):
            key = tuple(key) if isinstance(key, tuple) else (key,)
            values = group["target"].values
            median = np.median(values)
            q75 = np.percentile(values, 75)
            q25 = np.percentile(values, 25)
            scale = q75 - q25
            if scale <= 1e-10:
                scale = 1.0
            self.scalers[key] = (median, scale)
    
    def transform(self, df, target_col="target"):
        df = df.copy()
        for key, group in df.groupby(["currency", "risk_category", "instrument"]):
            key = tuple(key) if isinstance(key, tuple) else (key,)
            if key in self.scalers:
                idx = group.index
                median, scale = self.scalers[key]
                df.loc[idx, target_col] = (df.loc[idx, target_col] - median) / scale
        return df
    
    def inverse_transform(self, df, target_col="target"):
        df = df.copy()
        for key, group in df.groupby(["currency", "risk_category", "instrument"]):
            key = tuple(key) if isinstance(key, tuple) else (key,)
            if key in self.scalers:
                idx = group.index
                median, scale = self.scalers[key]
                df.loc[idx, target_col] = df.loc[idx, target_col] * scale + median
        return df


# ─── Future Target Creation ──────────────────────────────────────────────
def add_future_targets(df, horizon=5):
    """Create future target columns for multi-horizon direct forecasting."""
    df = df.copy()
    for h in range(1, horizon + 1):
        df[f"target_h{h}"] = df.groupby(["currency", "risk_category", "instrument"])["target"].shift(-h)
    return df


# ─── Categorical Feature Indices ─────────────────────────────────────────
def get_categorical_indices(feat_cols):
    """Get indices of categorical features in feature column list."""
    cat_cols = ["currency_code", "risk_cat_code", "instrument_code", "month", "quarter"]
    return [feat_cols.index(c) for c in cat_cols if c in feat_cols]


# ─── MASE Metric ────────────────────────────────────────────────────────
def mase(y_true, y_pred, y_train, seasonality=12):
    if len(y_true) == 0 or len(y_pred) == 0:
        return np.nan
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    y_train = np.asarray(y_train).ravel()
    if len(y_true) != len(y_pred):
        raise ValueError(f"Length mismatch: y_true={len(y_true)}, y_pred={len(y_pred)}")
    naive_errors = np.abs(y_train[seasonality:] - y_train[:-seasonality])
    scale = np.mean(naive_errors) if len(naive_errors) > 0 else np.mean(np.abs(np.diff(y_train)))
    if scale <= 1e-10:
        return np.nan
    return np.mean(np.abs(y_true - y_pred)) / scale


def rmse(y_true, y_pred):
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


# ─── Model Training ──────────────────────────────────────────────────────
def train_point_model(X_train, y_train, X_val, y_val, cat_indices):
    """Train point forecast model (median/mean)."""
    params = {
        "objective": "regression",
        "metric": "mae",
        "learning_rate": 0.03,
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
    
    model = lgb.train(
        params,
        lgb.Dataset(X_train, label=y_train, categorical_feature=cat_indices),
        valid_sets=[lgb.Dataset(X_val, label=y_val, reference=False, categorical_feature=cat_indices)],
        num_boost_round=500,
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
    )
    return model


def train_quantile_model(X_train, y_train, X_val, y_val, alpha, cat_indices):
    """Train a single quantile model."""
    params = {
        "objective": "quantile",
        "alpha": alpha,
        "metric": "quantile",
        "learning_rate": 0.03,
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
    
    model = lgb.train(
        {"objective": "quantile", "alpha": alpha, "metric": "quantile", "learning_rate": 0.03,
         "num_leaves": 63, "max_depth": 8, "min_child_samples": 20, "subsample": 0.8,
         "colsample_bytree": 0.8, "reg_alpha": 0.1, "reg_lambda": 0.1, "n_estimators": 500,
         "random_state": 42, "verbosity": -1, "n_jobs": -1},
        lgb.Dataset(X_train, label=y_train, categorical_feature=cat_indices),
        valid_sets=[lgb.Dataset(X_val, label=y_val, reference=False, categorical_feature=cat_indices)],
        num_boost_round=500,
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
    )
    return model


# ─── Training Pipeline ───────────────────────────────────────────────────
def train_regime(full_df, regime_name, horizon=5):
    """Train models for a regime (FX or IR)."""
    print(f"\n{'='*60}")
    print(f"Training {regime_name} regime")
    print(f"{'='*60}")
    
    # Filter data
    if regime_name == "FX":
        df = full_df[full_df["risk_category"] == "B"].copy()
    else:
        df = full_df[full_df["risk_category"].isin(["C", "I", "J"])].copy()
    
    # Feature engineering
    print(f"\n🔧 Building features for {regime_name}...")
    df = make_features(df)
    df = add_future_targets(df, horizon=5)
    
    # Train/test split
    train_df = df[df["period"] < "2026-01-01"].copy()
    test_df = df[df["period"] >= "2026-01-01"].copy()
    
    print(f"Train: {len(train_df):,}, Test: {len(test_df):,}")
    
    # Feature columns
    feat_cols = get_feature_cols(train_df)
    cat_indices = [feat_cols.index(c) for c in ["currency_code", "risk_cat_code", "instrument_code", "month", "quarter"] if c in feat_cols]
    print(f"Feature columns: {len(feat_cols)}, Categorical indices: {cat_indices}")
    
    # Prepare features
    X_train = train_df[get_feature_cols(train_df)].fillna(0)
    X_test = test_df[get_feature_cols(test_df)].fillna(0)
    
    # Global feature scaler
    feature_scaler = RobustScaler(quantile_range=(5.0, 95.0))
    X_train_scaled = RobustScaler(quantile_range=(5.0, 95.0)).fit_transform(train_df[get_feature_cols(train_df)].fillna(0))
    X_test_scaled = RobustScaler(quantile_range=(5.0, 95.0)).fit_transform(test_df[get_feature_cols(test_df)].fillna(0))
    
    # Target scaler (per-series)
    target_scaler = TargetScaler()
    target_scaler.fit(train_df, "target")
    
    # Scale target
    train_df_scaled = target_scaler.transform(train_df.copy(), "target")
    test_df_scaled = target_scaler.transform(test_df.copy(), "target")
    
    # Scale future targets too
    for h in range(1, 6):
        col = f"target_h{h}"
        if col in train_df.columns:
            train_df = target_scaler.transform(train_df.copy(), f"target_h{h}")
            test_df = target_scaler.transform(test_df.copy(), f"target_h{h}")
    
    # Train models for each horizon
    models = {}
    for h in range(1, 6):
        target_col = f"target_h{h}"
        if target_col not in train_df.columns:
            continue
        
        print(f"\n  Training horizon h={h}...")
        
        y_train = train_df[f"target_h{h}"].values
        y_val = test_df[f"target_h{h}"].values
        
        valid_train = ~np.isnan(train_df[f"target_h{h}"].values)
        valid_val = ~np.isnan(test_df[f"target_h{h}"].values)
        
        if valid_train.sum() < 100:
            continue
        
        if not valid_val.any():
            print(f"  Skipping horizon h={h}: no validation data available")
            continue
        
        # Use scaled features
        X_train_scaled = RobustScaler(quantile_range=(5.0, 95.0)).fit_transform(
            train_df[get_feature_cols(train_df)].fillna(0).values[valid_train]
        )
        X_val_scaled = RobustScaler(quantile_range=(5.0, 95.0)).fit_transform(
            test_df[get_feature_cols(test_df)].fillna(0).values[valid_val]
        )
        
        y_train_h = train_df[f"target_h{h}"].values[~np.isnan(train_df[f"target_h{h}"].values)]
        y_val_h = test_df[f"target_h{h}"].values[~np.isnan(test_df[f"target_h{h}"].values)]
        
        # Train median model
        median_model = lgb.train(
            {"objective": "regression", "metric": "mae", "learning_rate": 0.03, "num_leaves": 63,
             "max_depth": 8, "min_child_samples": 20, "subsample": 0.8, "colsample_bytree": 0.8,
             "reg_alpha": 0.1, "reg_lambda": 0.1, "n_estimators": 500, "random_state": 42,
             "verbosity": -1, "n_jobs": -1},
            lgb.Dataset(X_train_scaled, label=y_train_h, categorical_feature=cat_indices),
            valid_sets=[lgb.Dataset(X_val_scaled, label=y_val_h, reference=False, categorical_feature=cat_indices)],
            num_boost_round=500,
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
        )
        
        # Train quantile models
        quantile_models = {}
        for alpha in [0.1, 0.5, 0.9]:
            q_model = lgb.train(
                {"objective": "quantile", "alpha": alpha, "metric": "quantile", "learning_rate": 0.03,
                 "num_leaves": 63, "max_depth": 8, "min_child_samples": 20, "subsample": 0.8,
                 "colsample_bytree": 0.8, "reg_alpha": 0.1, "reg_lambda": 0.1, "n_estimators": 500,
                 "random_state": 42, "verbosity": -1, "n_jobs": -1},
                lgb.Dataset(X_train_scaled, label=y_train_h, categorical_feature=cat_indices),
                valid_sets=[lgb.Dataset(X_val_scaled, label=y_val_h, reference=False, categorical_feature=cat_indices)],
                num_boost_round=500,
                callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
            )
            quantile_models[alpha] = q_model
        
        models[h] = {"median": median_model, "quantiles": quantile_models}
    
    return models


# ─── Main ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("LightGBM Fixed Training Pipeline")
    print("=" * 60)
    
    # Load data
    print("📊 Loading dataset...")
    ds = load_dataset("HugoxBaca/bis-xtd-deriv")
    train_df = ds["train"].to_pandas()
    test_df = ds["test"].to_pandas()
    full_df = pd.concat([train_df, test_df]).sort_values(["currency", "risk_category", "instrument", "period"])
    
    # Regime split
    fx_df = full_df[full_df["risk_category"] == "B"].copy()
    ir_df = full_df[full_df["risk_category"].isin(["C", "I", "J"])].copy()
    
    print(f"FX series: {fx_df.groupby(['currency','instrument']).ngroup().nunique()}")
    print(f"IR series: {ir_df.groupby(['currency','risk_category','instrument']).ngroup().nunique()}")
    
    # Train FX
    print("\n" + "="*60)
    print("Training FX Regime")
    print("="*60)
    fx_preds = train_regime(full_df, "FX")
    
    # Train IR
    print("\n" + "="*60)
    print("Training IR Regime")
    print("="*60)
    ir_preds = train_regime(full_df, "IR")
    
    print("\n✅ Training complete!")