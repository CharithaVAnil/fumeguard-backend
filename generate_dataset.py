import pandas as pd
import numpy as np

np.random.seed(42)
n = 600
hours = np.tile(np.arange(8, 20), 50)[:n]  # 8am to 8pm pattern

# Simulate base gas levels with some daily pattern
base_mq135 = 200 + 50 * np.sin(np.linspace(0, 4 * np.pi, n)) + np.random.normal(0, 20, n)
base_mq2 = 1500 + 300 * np.sin(np.linspace(0, 4 * np.pi, n)) + np.random.normal(0, 100, n)

# Add some spikes (anomalies)
spike_indices = np.random.choice(n, 30, replace=False)
base_mq135[spike_indices] += np.random.uniform(200, 500, 30)
base_mq2[spike_indices] += np.random.uniform(500, 1500, 30)

temp = 25 + 5 * np.sin(np.linspace(0, 2 * np.pi, n)) + np.random.normal(0, 1, n)
humidity = 60 + 10 * np.sin(np.linspace(0, 2 * np.pi, n)) + np.random.normal(0, 2, n)

# Create future targets (t+30s and t+60s = next 1 and 2 rows)
df = pd.DataFrame({
    'timestamp': pd.date_range('2024-01-01 08:00', periods=n, freq='2s'),
    'mq135_ppm': np.clip(base_mq135, 50, 1000).round(1),
    'mq2_ppm': np.clip(base_mq2, 200, 5000).round(1),
    'temp': temp.round(1),
    'humidity': humidity.round(1),
    'hour': hours
})

# Targets: what ppm will be 30s (15 rows) and 60s (30 rows) in future
df['target_t30'] = df['mq135_ppm'].shift(-15).ffill()
df['target_t60'] = df['mq135_ppm'].shift(-30).ffill()

# Gas type labels for classifier
def label_gas(row):
    if row['mq2_ppm'] > 2000:
        return 'LPG'
    elif row['mq135_ppm'] > 400 and row['temp'] > 28:
        return 'NH3'
    elif row['mq135_ppm'] > 300:
        return 'CO2'
    elif row['mq2_ppm'] > 1800 and row['mq135_ppm'] > 250:
        return 'Smoke'
    else:
        return 'Clean'

df['gas_type'] = df.apply(label_gas, axis=1)
df.to_csv('sensor_data.csv', index=False)
print(f"Dataset created: {len(df)} rows")
print(df['gas_type'].value_counts())
print(df.head())