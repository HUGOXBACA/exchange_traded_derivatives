# BIS Exchange-Traded Derivatives (XTD) Project

This repository contains a complete pipeline to fetch, parse, store, model, and visualise the BIS **WS_XTD_DERIV** data set.

## Data source
Bank for International Settlements (2026), *Exchange-traded derivatives statistics*, BIS WS_XTD_DERIV 1.0 (data set), https://data.bis.org/topics/XTD_DER/data (accessed on 23 October 2026).

## Project layout
```
exchange_traded_derivatives/
├── data/
│   └── raw/
│       └── bis_xtd_deriv.parquet          # tidy table produced by src/fetch_and_parse.py
├── src/
│   ├── fetch_and_parse.py                 # downloads SDMX‑JSON → parquet
│   └── build_dataset.py                   # builds HF Dataset and pushes to the Hub
├── models/
│   └── train_baseline.py                  # quick Prophet baseline (optional)
├── spaces/
│   └── xtd_dashboard/
│       ├── app.py                         # Gradio dashboard
│       └── requirements.txt
├── requirements.txt                        # root dependencies
└── README.md                               # this file
```

## Quick start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Fetch & parse the BIS data
python src/fetch_and_parse.py

# 3. Create & push the Hugging Face Dataset (requires `hf auth login`)
python src/build_dataset.py

# 4. (Optional) Run a simple Prophet baseline
python models/train_baseline.py

# 5. Launch the Gradio dashboard locally
cd spaces/xtd_dashboard
pip install -r requirements.txt
python app.py
```

## Hugging Face Hub artefacts
- **Dataset** – `your-username/bis-xtd-deriv` (train / test split)
- **Model** – `your-username/bis-xtd-forecaster` (replace with your own model)
- **Space** – `your-username/bis-xtd-dashboard` (interactive Plotly chart)

Replace `your-username` with your actual Hugging Face user / org name before pushing.

## Extending the project
- Add more `MEASURE`, `INSTRUMENT`, or `MATURITY` codes to the request URL.
- Pull a longer history by moving `startPeriod` earlier.
- Join external macro series (policy rates, FX spot, CPI) for richer conditioning.
- Replace the Prophet baseline with a 🤗 Transformers `TimeSeriesTransformerForPrediction` for probabilistic multi‑series forecasting.