import pandas as pd
from prophet import Prophet
from sklearn.metrics import mean_absolute_error

df = pd.read_parquet("data/raw/bis_xtd_deriv.parquet")

# Columns already friendly
# Example: one currency/measure combo (USD, measure U)
sub = df[(df["currency"] == "USD") & (df["measure"] == "U")].copy()
sub["ds"] = pd.to_datetime(sub["period"])
sub = sub.sort_values("ds")

train = sub[sub["ds"] < "2026-01-01"]
test = sub[sub["ds"] >= "2026-01-01"]

m = Prophet(yearly_seasonality=True)
m.fit(train[["ds", "value"]].rename(columns={"value": "y"}))
future = m.make_future_dataframe(periods=len(test), freq="MS")
fc = m.predict(future)

mae = mean_absolute_error(test["value"], fc["yhat"].tail(len(test)))
print(f"Prophet MAE: {mae:.2f}")