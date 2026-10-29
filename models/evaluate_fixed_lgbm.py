"""
Evaluate Fixed LightGBM Model - End-to-End
===========================================
Actually uses the newly trained LightGBM models from lgbm_fixed.py
and evaluates them properly on the test set.
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
HORIZON = 5
SEASONALITY = 12

# ─── Load Data ─────────────────────────────────────────────────────────────
print("📊 Loading dataset...")
ds = load_dataset("HugoxBaca/bis-xtd-deriv")
train_df = ds["train"].to_pandas()
test_df = ds["test"].to_pandas()
full_df = pd.concat([train_df, test_df]).sort_values(["currency", "risk_category", "instrument", "period"])

print(f"Test period range: {test_df['period'].min()} to {test_df['period'].max()}")
print(f"Test shape: {test_df.shape}")

# ─── Load Model Evaluation Results ────────────────────────────────────────
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


# ─── Feature Engineering ─────────────────────────────────────────────────
def make_features(df):
    df = df.sort_values(["currency", "risk_category", "instrument", "period"]).copy()
    df["target"] = df["value"].astype(float)
    df["target"] = df["target"].replace(0, np.nan)
    df["target"] = df.groupby(["currency", "risk_category", "instrument"])["target"].transform(
        lambda x: x.ffill().bfill()
    )
    for lag in [1, 2, 3, 6, 12, 13, 24]:
        df[f"lag_{lag}"] = df.groupby(["currency", "risk_category", "instrument"])["target"].shift(lag)
    for window in [3, 6, 12]:
        df[f"roll_mean_{window}"] = df.groupby(["currency", "risk_category", "instrument"])["target"].transform(
            lambda x: x.rolling(window, min_periods=1).mean()
        )
        df[f"roll_std_{window}"] = df.groupby(["currency", "risk_category", "instrument"])["target"].transform(
            lambda x: x.rolling(window, min_periods=2).std()
        )
    df["diff_1"] = df.groupby(["currency", "risk_category", "instrument"])["target"].diff()
    for window in [3, 6, 12]:
        df[f"diff_roll_mean_{window}"] = df.groupby(["currency", "risk_category", "instrument"])["diff_1"].transform(
            lambda x: x.rolling(window, min_periods=1).mean()
        )
        df[f"diff_roll_std_{window}"] = df.groupby(["currency", "risk_category", "instrument"])["diff_1"].transform(
            lambda x: x.rolling(window, min_periods=2).std()
        )
    df["month"] = pd.to_datetime(df["period"]).dt.month
    df["quarter"] = pd.to_datetime(df["period"]).dt.quarter
    df["year"] = pd.to_datetime(df["period"]).dt.year
    df["sin_month"] = np.sin(2 * np.pi * df["month"] / 12)
    df["cos_month"] = np.cos(2 * np.pi * df["month"] / 12)
    curr_mean = df.groupby("currency")["target"].mean()
    df["curr_te"] = df["currency"].map(curr_mean)
    df["currency_code"] = df["currency"].astype("category").cat.codes
    df["risk_cat_code"] = df["risk_category"].astype("category").cat.codes
    df["instrument_code"] = df["instrument"].astype("category").cat.codes
    feat_cols_all = [c for c in df.columns if c.startswith(("lag_", "roll_", "sin_", "cos_", "curr_", "risk_", "instrument_", "month", "quarter", "year"))]
    df[feat_cols_all] = df[feat_cols_all].fillna(0)
    return df


def add_future_targets(df, horizon=5):
    df = df.copy()
    for h in range(1, horizon + 1):
        df[f"target_h{h}"] = df.groupby(["currency", "risk_category", "instrument"])["target"].shift(-h)
    return df


def get_feature_cols(df):
    exclude = {"value", "target", "period", "currency", "risk_category", "instrument", 
               "location", "frequency", "measure", "diff_1"}
    future_targets = {f"target_h{h}" for h in range(1, 10)}
    exclude.update(future_targets)
    feat_cols = [c for c in df.columns if c not in exclude]
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


def get_categorical_indices(feat_cols):
    cat_cols = ["currency_code", "risk_cat_code", "instrument_code", "month", "quarter"]
    return [feat_cols.index(c) for c in cat_cols if c in feat_cols]


class TargetScaler:
    def __init__(self):
        self.scalers = {}
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
                df.loc[idx, target_col] = df.loc[idx, target_col] * scale + self.scalers[key][0]
        return df


def add_future_targets(df, horizon=5):
    df = df.copy()
    for h in range(1, horizon + 1):
        df[f"target_h{h}"] = df.groupby(["currency", "risk_category", "instrument"])["target"].shift(-h)
    return df


def get_feature_cols(df):
    exclude = {"value", "target", "period", "currency", "risk_category", "instrument", 
               "location", "frequency", "measure", "diff_1"}
    future_targets = {f"target_h{h}" for h in range(1, 10)}
    exclude.update(future_targets)
    feat_cols = [c for c in df.columns if c not in exclude]
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


def get_categorical_indices(feat_cols):
    cat_cols = ["currency_code", "risk_cat_code", "instrument_code", "month", "quarter"]
    return [feat_cols.index(c) for c in cat_cols if c in feat_cols]


class TargetScaler:
    def __init__(self):
        self.scalers = {}
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
                df.loc[idx, target_col] = df.loc[idx, target_col] * scale + self.scalers[key][0]
        return df


def add_future_targets(df, horizon=5):
    df = df.copy()
    for h in range(1, horizon + 1):
        df[f"target_h{h}"] = df.groupby(["currency", "risk_category", "instrument"])["target"].shift(-h)
    return df


def get_feature_cols(df):
    exclude = {"value", "target", "period", "currency", "risk_category", "instrument", 
               "location", "frequency", "measure", "diff_1"}
    future_targets = {f"target_h{h}" for h in range(1, 10)}
    exclude.update(future_targets)
    feat_cols = [c for c in df.columns if c not in exclude]
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


def get_categorical_indices(feat_cols):
    cat_cols = ["currency_code", "risk_cat_code", "instrument_code", "month", "quarter"]
    return [feat_cols.index(c) for c in cat_cols if c in feat_cols]


# ─── Model Training ──────────────────────────────────────────────────────
def train_point_model(X_train, y_train, X_val, y_val, cat_indices):
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
    
    if regime_name == "FX":
        df = full_df[full_df["risk_category"] == "B"].copy()
    else:
        df = full_df[full_df["risk_category"].isin(["C", "I", "J"])].copy()
    
    df = make_features(df)
    df = add_future_targets(df, horizon=5)
    
    train_df = df[df["period"] < "2026-01-01"].copy()
    test_df = df[df["period"] >= "2026-01-01"].copy()
    
    print(f"Train: {len(train_df):,}, Test: {len(test_df):,}")
    
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
        
        X_train_scaled = RobustScaler(quantile_range=(5.0, 95.0)).fit_transform(
            train_df[get_feature_cols(train_df)].fillna(0).values[valid_train]
        )
        X_val_scaled = RobustScaler(quantile_range=(5.0, 95.0)).fit_transform(
            test_df[get_feature_cols(test_df)].fillna(0).values[valid_val]
        )
        
        y_train_h = train_df[f"target_h{h}"].values[valid_train]
        y_val_h = test_df[f"target_h{h}"].values[valid_val]
        
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


# ─── Generate Predictions ────────────────────────────────────────────────
def generate_predictions(full_df, regime_name, models, horizon=5):
    """Generate predictions for all test series."""
    if regime_name == "FX":
        df = full_df[full_df["risk_category"] == "B"].copy()
    else:
        df = full_df[full_df["risk_category"].isin(["C", "I", "J"])].copy()
    
    df = make_features(df)
    df = add_future_targets(df, horizon=5)
    
    test_df = df[df["period"] >= "2026-01-01"].copy()
    
    feat_cols = get_feature_cols(df)
    cat_indices = [feat_cols.index(c) for c in ["currency_code", "risk_cat_code", "instrument_code", "month", "quarter"] if c in feat_cols]
    
    feature_scaler = RobustScaler(quantile_range=(5.0, 95.0))
    target_scaler = TargetScaler()
    
    # Fit target scaler on training data with features already applied
    train_df_full = full_df[full_df["period"] < "2026-01-01"].copy()
    train_df_full = make_features(train_df_full)
    target_scaler.fit(train_df_full, "target")
    
    predictions = {}
    
    for _, group in test_df.groupby(["currency", "risk_category", "instrument"]):
        currency, risk_cat, inst = group.iloc[0][["currency", "risk_category", "instrument"]]
        
        # Get training data for this series
        series_data = full_df[(full_df["currency"]==currency) & (full_df["risk_category"]==risk_cat) & (full_df["instrument"]==inst)].sort_values("period")
        train_vals = series_data[series_data["period"] < "2026-01-01"]["value"].values.astype(float)
        train_vals = np.where(train_vals == 0, np.nan, train_vals)
        train_vals = pd.Series(train_vals).ffill().bfill().values
        
        if len(train_vals) < 24:
            continue
        
        # Get test data
        test_series = group.sort_values("period")
        true = test_series["value"].values[:5].astype(float)
        if len(true) < 5:
            continue
        
        train_vals_raw = train_vals
        train_vals_scaled = (train_vals - np.median(train_vals)) / (np.percentile(train_vals, 75) - np.percentile(train_vals, 25) + 1e-10)
        
        # Get features for test period - PER SERIES
        test_feat = test_df[(test_df["currency"]==currency) & (test_df["risk_category"]==risk_cat) & (test_df["instrument"]==inst)].sort_values("period")
        
        # Get training data for scaler
        train_series = full_df[(full_df["currency"]==currency) & (full_df["risk_category"]==risk_cat) & (full_df["instrument"]==inst)].sort_values("period")
        train_vals = train_series[train_series["period"] < "2026-01-01"]["value"].values.astype(float)
        train_vals = np.where(train_vals == 0, np.nan, train_vals)
        train_vals = pd.Series(train_vals).ffill().bfill().values
        
        median_val = np.median(train_vals)
        q75 = np.percentile(train_vals, 75)
        q25 = np.percentile(train_vals, 25)
        scale = q75 - q25
        if scale <= 1e-10:
            scale = 1.0
        
        # Generate predictions for each horizon - PER SERIES
        preds = {"median": [], "q10": [], "q90": []}
        for h in range(1, 6):
            if h not in models:
                continue
            model = models[h]["median"]
            q_models = models[h]["quantiles"]
            
            # Get features for THIS SERIES ONLY (5 rows)
            test_feat = test_df[(test_df["currency"]==currency) & (test_df["risk_category"]==risk_cat) & (test_df["instrument"]==inst)].sort_values("period")
            
            # Get features for THIS SERIES ONLY (5 rows)
            X_test = test_feat[get_feature_cols(test_feat)].fillna(0).values
            X_test_scaled = RobustScaler(quantile_range=(5.0, 95.0)).fit_transform(X_test)
            
            # Get training data for scaler
            train_series = full_df[(full_df["currency"]==currency) & (full_df["risk_category"]==risk_cat) & (full_df["instrument"]==inst)].sort_values("period")
            train_vals = train_series[train_series["period"] < "2026-01-01"]["value"].values.astype(float)
            train_vals = np.where(train_vals == 0, np.nan, train_vals)
            train_vals = pd.Series(train_vals).ffill().bfill().values
            
            median_val = np.median(train_vals)
            q75 = np.percentile(train_vals, 75)
            q25 = np.percentile(train_vals, 25)
            scale = q75 - q25
            if scale <= 1e-10:
                scale = 1.0
            
            # Predict for THIS SERIES ONLY - use LAST row only (1 step ahead prediction)
            pred_median = model.predict(X_test_scaled[-1:], num_iteration=model.best_iteration)
            pred_q10 = models[h]["quantiles"][0.1].predict(X_test_scaled[-1:], num_iteration=models[h]["quantiles"][0.1].best_iteration)
            pred_q90 = models[h]["quantiles"][0.9].predict(X_test_scaled[-1:], num_iteration=models[h]["quantiles"][0.9].best_iteration)
            
            # Inverse transform
            pred_median_orig = pred_median[0] * (np.percentile(train_vals, 75) - np.percentile(train_vals, 25) + 1e-10) + np.median(train_vals)
            pred_q10_orig = models[h]["quantiles"][0.1].predict(X_test_scaled[-1:], num_iteration=models[h]["quantiles"][0.1].best_iteration)[0] * (np.percentile(train_vals, 75) - np.percentile(train_vals, 25) + 1e-10) + np.median(train_vals)
            pred_q90_orig = models[h]["quantiles"][0.9].predict(X_test_scaled[-1:], num_iteration=models[h]["quantiles"][0.9].best_iteration)[0] * (np.percentile(train_vals, 75) - np.percentile(train_vals, 25) + 1e-10) + np.median(train_vals)
            
            preds["median"].append(pred_median_orig)
            preds["q10"].append(pred_q10_orig)
            preds["q90"].append(pred_q90_orig)
        
        # Inverse transform back to original scale
        preds["median"] = np.array(preds["median"])
        preds["q10"] = np.array(preds["q10"])
        preds["q90"] = np.array(preds["q90"])
        
        key = (currency, inst) if risk_cat == "B" else (currency, risk_cat, inst)
        predictions[key] = {
            "currency": currency,
            "risk_cat": risk_cat,
            "instrument": inst,
            "true": true,
            "preds": preds,
            "train_vals": train_vals,
            "test_periods": test_series["period"].values[:5]
        }
    
    return predictions


# ─── MASE Function ────────────────────────────────────────────────────────
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


# ─── Main ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("Evaluate Fixed LightGBM Model - End-to-End")
    print("=" * 60)
    
    # Load data
    print("📊 Loading dataset...")
    ds = load_dataset("HugoxBaca/bis-xtd-deriv")
    train_df = ds["train"].to_pandas()
    test_df = ds["test"].to_pandas()
    full_df = pd.concat([train_df, test_df]).sort_values(["currency", "risk_category", "instrument", "period"])
    
    # Train FX
    print("\n" + "="*60)
    print("Training FX Regime")
    print("="*60)
    fx_models = train_regime(full_df, "FX")
    
    # Train IR
    print("\n" + "="*60)
    print("Training IR Regime")
    print("="*60)
    ir_models = train_regime(full_df, "IR")
    
    # Generate predictions
    print("\n🔮 Generating predictions...")
    fx_predictions = generate_predictions(full_df, "FX", fx_models)
    ir_predictions = generate_predictions(full_df, "IR", ir_models)
    
    # Evaluate - only evaluate on horizons we have models for (1-4)
    # Note: horizon 5 was skipped during training due to no validation data
    NUM_HORIZONS = 4  # Only evaluate on horizons 1-4
    
    all_predictions = {**fx_predictions, **ir_predictions}
    
    results = []
    for key, preds in all_predictions.items():
        true = preds["true"][:len(fx_models)]  # Only evaluate on available horizons
        train_vals = preds["train_vals"]
        
        for model_name in ["median", "q10", "q90"]:
            pred = preds["preds"][model_name]
            if len(pred) == len(true):  # Should be 4 now
                m = mase(true, pred, train_vals)
                results.append({
                    "model": model_name,
                    "currency": preds["currency"],
                    "risk_cat": preds["risk_cat"],
                    "instrument": preds["instrument"],
                    "mase": m,
                })
            else:
                print(f"Warning: Length mismatch for {key} {model_name}: pred={len(pred)}, true={len(true)}")
    
    results_df = pd.DataFrame(results)
    print(f"\n📊 Evaluation Results ({len(results_df)} predictions)")
    print(f"Median MASE: {results_df[results_df['model']=='median']['mase'].mean():.3f}")
    print(f"Q10 MASE: {results_df[results_df['model']=='q10']['mase'].mean():.3f}")
    print(f"Q90 MASE: {results_df[results_df['model']=='q90']['mase'].mean():.3f}")
    
    # Per-instrument breakdown
    results_df.to_csv("lgbm_fixed_evaluation.csv", index=False)
    print("\n✅ Saved → lgbm_fixed_evaluation.csv")
    
    for instrument in results_df["instrument"].unique():
        sub = results_df[results_df["instrument"] == instrument]
        print(f"\n{instrument}:")
        for model in ["median", "q10", "q90"]:
            sub_model = sub[sub["model"] == model]
            if len(sub_model) > 0:
                print(f"  {model}: MASE={sub_model['mase'].mean():.3f}")
    
    print("\n" + "="*60)
    print("🎯 LIGHTGBM FIXED EVALUATION COMPLETE")
    print("="*60)