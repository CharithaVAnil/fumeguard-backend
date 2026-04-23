import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import joblib
import os

print("Loading dataset...")
df = pd.read_csv('sensor_data.csv')

# Features used for anomaly detection
# ppm_delta = change from previous reading
df['ppm_delta'] = df['mq135_ppm'].diff().fillna(0)
df['rolling_std'] = df['mq135_ppm'].rolling(5).std().fillna(0)

anomaly_features = ['ppm_delta', 'rolling_std', 'hour']
X = df[anomaly_features]

# Train Isolation Forest
# contamination=0.05 means we expect 5% of data to be anomalies
iso_forest = IsolationForest(
    n_estimators=100,
    contamination=0.05,
    random_state=42
)

print("Training Isolation Forest...")
iso_forest.fit(X)

os.makedirs('models', exist_ok=True)
joblib.dump(iso_forest, 'models/isolation_forest.pkl')
print("✅ Isolation Forest saved to models/isolation_forest.pkl")

# Test it
scores = iso_forest.decision_function(X)
predictions = iso_forest.predict(X)
anomalies = (predictions == -1).sum()
print(f"Found {anomalies} anomalies in training data ({anomalies/len(df)*100:.1f}%)")