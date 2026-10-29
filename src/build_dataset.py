import pandas as pd
from datasets import Dataset, DatasetDict, Features, Value, Sequence
import tempfile
import os

df = pd.read_parquet("data/raw/bis_xtd_deriv.parquet")

# Keep all dimensions - they're useful for hierarchical modeling
# Filter to main measure (U = Turnover) and all-exchanges (8A) for primary analysis
# But keep everything for hierarchical/panel models
df_main = df[(df["measure"] == "U") & (df["location"] == "8A")].copy()

print(f"Main analysis set (measure=U, location=8A): {len(df_main):,} rows")
print(f"Currencies: {df_main['currency'].nunique()}")
print(f"Risk categories: {df_main['risk_category'].unique()}")
print(f"Instruments: {df_main['instrument'].unique()}")
print(f"Period range: {df_main['period'].min()} to {df_main['period'].max()}")

# For each currency-risk-instrument combo, check series length
series_lens = df_main.groupby(["currency", "risk_category", "instrument"]).size()
print(f"\nSeries lengths: min={series_lens.min()}, max={series_lens.max()}, median={series_lens.median()}")
print(f"Series with >200 obs: {(series_lens > 200).sum()} / {len(series_lens)}")

# Time-based split: everything before 2026 is train, after is test
cutoff = "2026-01-01"
train_df = df_main[df_main["period"] < cutoff]
test_df = df_main[df_main["period"] >= cutoff]

print(f"\nTrain: {len(train_df):,} rows, Test: {len(test_df):,} rows")

# Save full dataset (all dimensions) for hierarchical modeling
full_features = Features({
    "frequency": Value("string"),
    "measure": Value("string"),
    "risk_category": Value("string"),
    "instrument": Value("string"),
    "currency": Value("string"),
    "location": Value("string"),
    "period": Value("string"),
    "value": Value("float64"),
})

# Save main analysis set
main_features = Features({
    "frequency": Value("string"),
    "measure": Value("string"),
    "risk_category": Value("string"),
    "instrument": Value("string"),
    "currency": Value("string"),
    "location": Value("string"),
    "period": Value("string"),
    "value": Value("float64"),
})

# Write to temp parquet and load via datasets (avoids pickling issues)
def df_to_dataset(df, features):
    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as tmp:
        df.to_parquet(tmp.name, index=False)
        tmp_path = tmp.name
    loaded = load_dataset("parquet", data_files=tmp_path, split="train")
    os.unlink(tmp_path)
    return loaded.cast(features)

from datasets import load_dataset

train_ds = df_to_dataset(train_df, main_features)
test_ds = df_to_dataset(test_df, main_features)

ds_dict = DatasetDict({"train": train_ds, "test": test_ds})

# Push to Hub
ds_dict.push_to_hub("HugoxBaca/bis-xtd-deriv")
print("✅ Dataset pushed to HugoxBaca/bis-xtd-deriv")

# Also save the full dataset (all locations, measures) for hierarchical work
full_train = df[df["period"] < cutoff]
full_test = df[df["period"] >= cutoff]

full_train_ds = df_to_dataset(full_train, full_features)
full_test_ds = df_to_dataset(full_test, full_features)

full_dict = DatasetDict({"train": full_train_ds, "test": full_test_ds})
full_dict.push_to_hub("HugoxBaca/bis-xtd-deriv-full")
print("✅ Full dataset pushed to HugoxBaca/bis-xtd-deriv-full")