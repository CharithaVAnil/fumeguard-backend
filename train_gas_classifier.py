import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
import os

print("Loading dataset...")
df = pd.read_csv('sensor_data.csv')

# Features: sensor readings identify gas type
features = ['mq135_ppm', 'mq2_ppm', 'temp', 'humidity']
target = 'gas_type'

X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training Random Forest Gas Classifier (5 classes)...")
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Evaluate
y_pred = clf.predict(X_test)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Save model
os.makedirs('models', exist_ok=True)
joblib.dump(clf, 'models/gas_classifier.pkl')
print("✅ Gas Classifier saved to models/gas_classifier.pkl")

# Feature importance
fi = pd.DataFrame({
    'feature': features,
    'importance': clf.feature_importances_
}).sort_values('importance', ascending=False)
print("\nFeature Importance:")
print(fi)