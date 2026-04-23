import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib
import os

# TensorFlow / Keras
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

print("Loading dataset...")
df = pd.read_csv('sensor_data.csv')

# Features used (must match what ESP32 sends)
features = ['mq135_ppm', 'mq2_ppm', 'temp', 'humidity', 'hour']
targets = ['target_t30', 'target_t60']

df = df.dropna()

# Scale features between 0 and 1
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X_scaled = scaler_X.fit_transform(df[features])
y_scaled = scaler_y.fit_transform(df[targets])

# Create sliding window of 30 readings
WINDOW = 30

def create_sequences(X, y, window):
    Xs, ys = [], []
    for i in range(len(X) - window):
        Xs.append(X[i:i+window])
        ys.append(y[i+window])
    return np.array(Xs), np.array(ys)

X_seq, y_seq = create_sequences(X_scaled, y_scaled, WINDOW)
print(f"Sequences: X={X_seq.shape}, y={y_seq.shape}")

# Split: 80% train, 20% test
split = int(0.8 * len(X_seq))
X_train, X_test = X_seq[:split], X_seq[split:]
y_train, y_test = y_seq[:split], y_seq[split:]

# Build LSTM model
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(WINDOW, len(features))),
    Dropout(0.2),
    LSTM(32),
    Dropout(0.2),
    Dense(16, activation='relu'),
    Dense(2)  # predicts t+30 and t+60
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])
model.summary()

early_stop = EarlyStopping(patience=10, restore_best_weights=True)

print("\nTraining LSTM model...")
history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=16,
    validation_data=(X_test, y_test),
    callbacks=[early_stop],
    verbose=1
)

# Save model and scalers
os.makedirs('models', exist_ok=True)
model.save('models/lstm_model.keras')
joblib.dump(scaler_X, 'models/scaler_X.pkl')
joblib.dump(scaler_y, 'models/scaler_y.pkl')

print("\n✅ LSTM model saved to models/lstm_model.h5")
print("✅ Scalers saved to models/")

# Quick test prediction
test_pred = model.predict(X_test[:1])
test_pred_real = scaler_y.inverse_transform(test_pred)
print(f"\nSample prediction → t+30: {test_pred_real[0][0]:.1f} ppm, t+60: {test_pred_real[0][1]:.1f} ppm")