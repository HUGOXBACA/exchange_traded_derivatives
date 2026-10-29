"""
Train a Hugging Face TimeSeriesTransformer on BIS XTD data.
Multi-series (25 currencies), monthly frequency, 6-step forecast horizon (2027-H1).
"""
import os
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset, Features, Value, Sequence, Dataset
from transformers import (
    TimeSeriesTransformerConfig,
    TimeSeriesTransformerForPrediction,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)
from huggingface_hub import HfApi, create_repo, upload_folder

# ─── Config ──────────────────────────────────────────────────────────────
REPO_ID = "HugoxBaca/bis-xtd-forecaster"
PREDICTION_LENGTH = 6          # 6 months = H1 2027
CONTEXT_LENGTH = 36            # encoder context length
MAX_LAG = 1                    # max lag in lags_sequence
LAGS_SEQUENCE = [1]
BATCH_SIZE = 32
EPOCHS = 30
LR = 5e-4
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

# ─── Load dataset from Hub ──────────────────────────────────────────────
ds = load_dataset("HugoxBaca/bis-xtd-deriv")
train_df = ds["train"].to_pandas()
test_df  = ds["test"].to_pandas()

# ─── Build static categorical vocabularies ──────────────────────────────
cat_cols = ["currency", "instrument", "measure", "risk_category", "location"]
cat_maps = {c: {v: i for i, v in enumerate(sorted(train_df[c].unique()))} for c in cat_cols}
cardinalities = [len(cat_maps[c]) for c in cat_cols]

HIST_NEED = CONTEXT_LENGTH + MAX_LAG   # length of past_values fed to model

# ─── Convert each series to a fixed-length context + future ─────────────
def make_example(vals, is_train):
    """vals: full series values (float32 array) for a single series."""
    total_need = HIST_NEED + (PREDICTION_LENGTH if not is_train else PREDICTION_LENGTH)
    if len(vals) < total_need:
        pad = total_need - len(vals)
        vals = np.concatenate([np.full(pad, np.nan), vals])

    if is_train:
        # Use the last HIST_NEED+PREDICTION_LENGTH points from train period
        window = vals[-total_need:]
        past = window[:HIST_NEED]
        future = window[HIST_NEED:]
    else:
        # For test/forecast: past = last HIST_NEED before forecast origin, future = next PREDICTION_LENGTH (ground truth in test)
        past = vals[-(HIST_NEED + PREDICTION_LENGTH):-PREDICTION_LENGTH]
        future = vals[-PREDICTION_LENGTH:]

    # time features normalized by HIST_NEED
    t_past = (np.arange(HIST_NEED, dtype=np.float32) / HIST_NEED).tolist()
    past_time = [[x] for x in t_past]
    t_future = (np.arange(HIST_NEED, HIST_NEED + PREDICTION_LENGTH, dtype=np.float32) / HIST_NEED).tolist()
    future_time = [[x] for x in t_future]

    past_mask = (~np.isnan(past)).astype(np.float32).tolist()
    future_mask = (~np.isnan(future)).astype(np.float32).tolist()

    return {
        "past_values": past.tolist(),
        "past_time_features": past_time,
        "past_observed_mask": past_mask,
        "future_values": future.tolist(),
        "future_time_features": future_time,
        "future_observed_mask": future_mask,
    }

# Build list of examples
train_examples = []
for _, g in train_df.groupby(cat_cols):
    vals = g["value"].values.astype(np.float32)
    ex = make_example(vals, is_train=True)
    ex["static_categorical_features"] = [cat_maps[c][g[c].iloc[0]] for c in cat_cols]
    train_examples.append(ex)

test_examples = []
for _, g in test_df.groupby(cat_cols):
    vals = g["value"].values.astype(np.float32)
    ex = make_example(vals, is_train=False)
    ex["static_categorical_features"] = [cat_maps[c][g[c].iloc[0]] for c in cat_cols]
    test_examples.append(ex)

# ─── Write examples to parquet and load via load_dataset to avoid pickling issues ─────────────────────
def examples_to_dataset(examples):
    rows = []
    for ex in examples:
        rows.append(ex)
    df = pd.DataFrame(rows)
    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as tmp:
        df.to_parquet(tmp.name, index=False)
        tmp_path = tmp.name
    loaded = load_dataset("parquet", data_files=tmp_path, split="train")
    os.unlink(tmp_path)
    return loaded

train_ds = examples_to_dataset(train_examples)
test_ds  = examples_to_dataset(test_examples)

# ─── Model config ───────────────────────────────────────────────────────
config = TimeSeriesTransformerConfig(
    prediction_length=PREDICTION_LENGTH,
    context_length=CONTEXT_LENGTH,
    num_time_features=1,
    num_static_categorical_features=len(cat_cols),
    cardinality=cardinalities,
    embedding_dimension=[min(16, c//2) for c in cardinalities],
    d_model=128,
    encoder_layers=4,
    decoder_layers=4,
    encoder_attention_heads=8,
    decoder_attention_heads=8,
    dropout=0.1,
    scaling=True,
    lags_sequence=LAGS_SEQUENCE,
)

model = TimeSeriesTransformerForPrediction(config).to(DEVICE)

# ─── Training args ──────────────────────────────────────────────────────
args = TrainingArguments(
    output_dir="./transformer_ckpt",
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    learning_rate=LR,
    num_train_epochs=5,  # quick test
    weight_decay=1e-2,
    eval_strategy="no",
    save_strategy="epoch",
    logging_steps=20,
    report_to="none",
    remove_unused_columns=False,
    dataloader_num_workers=0,
)

def compute_metrics(eval_pred):
    preds, labels = eval_pred
    mae = np.mean(np.abs(preds - labels))
    return {"mae": mae}

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=test_ds,
    compute_metrics=compute_metrics,
)

# ─── Train ──────────────────────────────────────────────────────────────
print(f"Training on {DEVICE} …")
trainer.train()

# ─── Save & push ────────────────────────────────────────────────────────
trainer.save_model("./transformer_ckpt")
create_repo(REPO_ID, repo_type="model", exist_ok=True)
upload_folder(
    repo_id=REPO_ID,
    folder_path="./transformer_ckpt",
    repo_type="model",
    commit_message="TimeSeriesTransformer trained on BIS XTD (25 currencies, 6-month horizon)",
)
print(f"✅ Model pushed to {REPO_ID}")

# ─── Generate 2027-H1 forecasts for ALL series ─────────────────────────
model.eval()
full_df = pd.concat([train_df, test_df]).sort_values("period")

forecast_rows = []
with torch.no_grad():
    for _, g in full_df.groupby(cat_cols):
        vals = g["value"].values.astype(np.float32)
        ex = make_example(vals, is_train=False)
        ex["static_categorical_features"] = [cat_maps[c][g[c].iloc[0]] for c in cat_cols]
        past = torch.tensor(ex["past_values"]).unsqueeze(0).to(DEVICE)
        past_tf = torch.tensor(ex["past_time_features"]).unsqueeze(0).to(DEVICE)
        past_mask = torch.tensor(ex["past_observed_mask"]).unsqueeze(0).to(DEVICE)
        static_cat = torch.tensor(ex["static_categorical_features"]).unsqueeze(0).to(DEVICE)
        future_tf = torch.tensor(ex["future_time_features"]).unsqueeze(0).to(DEVICE)
        future_mask = torch.tensor(ex["future_observed_mask"]).unsqueeze(0).to(DEVICE)

        out = model.generate(
            past_values=past,
            past_time_features=past_tf,
            past_observed_mask=past_mask,
            static_categorical_features=static_cat,
            future_time_features=future_tf,
        )
        fc = out.sequences.squeeze(0).cpu().numpy()
        # Ensure it's 1D
        if fc.ndim > 1:
            fc = fc.flatten()
        fc = fc[:PREDICTION_LENGTH]

        row = {c: g[c].iloc[0] for c in cat_cols}
        for i, v in enumerate(fc):
            row[f"forecast_m{i+1}"] = float(v)
        forecast_rows.append(row)

fc_df = pd.DataFrame(forecast_rows)
fc_path = "forecasts_2027_H1.parquet"
fc_df.to_parquet(fc_path, index=False)
print(f"📈 Forecasts saved → {fc_path}")
print(fc_df.head())