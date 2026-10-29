import json
import requests
import pandas as pd

# ─── COMPREHENSIVE BIS XTD QUERY ───────────────────────────────────────
# Dimensions:
#   Frequency: M (monthly), Q (quarterly), A (annual)
#   Measure: U (Turnover - notional daily avg), A (Outstanding notional)
#   Risk category: B (FX), C (IR), I (IR short-term), J (IR long-term)
#   Instrument: A (All), T (Total futures), H (Options)
#   Issue currency: 25 currencies
#   Location: 8A (All exchanges), 8B, 8C, 8E, 8F, 8G, 8K

# Build the key: freq.measure.risk_cat.instrument.currencies.location
# Use + to separate multiple values within a dimension

CURRENCIES = "AUD+BRL+CAD+CHF+CNY+DKK+EU1+GBP+HKD+HUF+INR+JPY+KRW+MXN+NOK+NZD+PLN+RUB+SEK+SGD+TO1+TRY+TWD+USD+ZAR"

# Query for: Monthly, both measures, all risk categories, all instruments, all currencies, all exchanges
URL = (
    "https://stats.bis.org/api/v2/data/dataflow/BIS/WS_XTD_DERIV/1.0/"
    f"M.U+A.B+C+I+J.A+T+H.{CURRENCIES}.8A+8B+8C+8E+8F+8G+8K?"
    "startPeriod=1995-01-01&endPeriod=2026-10-23&format=sdmx-json"
)

print(f"Query URL length: {len(URL)} chars")
print(f"URL: {URL[:200]}...")


def parse_sdmx_json(raw: dict) -> pd.DataFrame:
    """Parse BIS SDMX‑JSON (v2) into a tidy DataFrame."""
    structure = raw["data"]["structure"]
    dataset = raw["data"]["dataSets"][0]

    series_dims = structure["dimensions"]["series"]
    dim_names = [d["name"] for d in series_dims]
    dim_values = [[v["id"] for v in d["values"]] for d in series_dims]

    obs_dim = structure["dimensions"]["observation"][0]
    periods = [v["id"] for v in obs_dim["values"]]

    rows = []
    for series_key, series_obj in dataset["series"].items():
        positions = [int(p) for p in series_key.split(":")]
        labels = {
            name: dim_values[i][pos]
            for i, (name, pos) in enumerate(zip(dim_names, positions))
        }

        for obs_idx, obs_arr in series_obj["observations"].items():
            value = obs_arr[0]
            if value is None:
                continue
            rows.append(
                {
                    **labels,
                    "period": periods[int(obs_idx)],
                    "value": float(value),
                }
            )

    df = pd.DataFrame(rows)

    # Rename to friendly column names
    df = df.rename(columns={
        "Frequency": "frequency",
        "Measure": "measure",
        "Risk category": "risk_category",
        "Instrument": "instrument",
        "Issue currency": "currency",
        "Location of trade (Exchange or country)": "location",
    })

    return df


if __name__ == "__main__":
    print("Fetching comprehensive BIS XTD data...")
    raw = requests.get(URL, timeout=120).json()
    df = parse_sdmx_json(raw)

    out_path = "data/raw/bis_xtd_deriv.parquet"
    df.to_parquet(out_path, index=False)
    print(f"Saved {len(df):,} rows → {out_path}")
    print(f"Shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"\nUnique values per dimension:")
    for col in ["frequency", "measure", "risk_category", "instrument", "currency", "location"]:
        print(f"  {col}: {df[col].nunique()} ({sorted(df[col].unique())[:10]}{'...' if df[col].nunique() > 10 else ''})")
    print(f"\nPeriod range: {df['period'].min()} to {df['period'].max()}")
    print(df.head())