"""
generate_dataset.py
Generates a realistic synthetic energy consumption dataset simulating
smart grid logs for a mid-size commercial building (2 years of hourly data).
"""

import numpy as np
import pandas as pd

np.random.seed(42)

# --- Date range: 2 years of hourly readings ---
dates = pd.date_range(start="2022-01-01", end="2023-12-31 23:00:00", freq="h")
n = len(dates)

hour       = dates.hour
dayofweek  = dates.dayofweek          # 0 = Monday … 6 = Sunday
month      = dates.month
is_weekend = (dayofweek >= 5).astype(int)

# Base load (kWh): higher during office hours, lower at night
base = (
    20
    + 30 * np.sin(np.pi * (hour - 6) / 12) * (hour >= 6) * (hour <= 22)
    - 10 * is_weekend                          # weekends use less
    + 5  * np.sin(2 * np.pi * month / 12)      # seasonal variation
)

# Noise + occasional spikes
noise  = np.random.normal(0, 3, n)
spikes = np.random.choice([0, 15], size=n, p=[0.97, 0.03])
energy = np.clip(base + noise + spikes, 5, 120)

df = pd.DataFrame({"Datetime": dates, "Energy_kWh": energy.round(2)})
df.to_csv("data/energy.csv", index=False)
print(f"Dataset created: {len(df):,} rows  |  {df['Energy_kWh'].mean():.2f} kWh avg")
