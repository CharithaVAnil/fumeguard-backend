import requests
import time

# Change this to your Railway URL
URL = "https://fumeguard-backend-production.up.railway.app/data"

test_cases = [
    {
        "name": "✅ Clean Air (no alert)",
        "data": {"mq135_ppm": 150, "mq2_ppm": 800, "temp": 25.0, "humidity": 60.0, "fan_pct": 0, "trigger": "none"}
    },
    {
        "name": "⚠️ MQ135 Danger (NH3/CO2)",
        "data": {"mq135_ppm": 600, "mq2_ppm": 900, "temp": 30.0, "humidity": 55.0, "fan_pct": 100, "trigger": "threshold_exceeded"}
    },
    {
        "name": "🔥 MQ2 Danger (LPG)",
        "data": {"mq135_ppm": 200, "mq2_ppm": 3500, "temp": 28.0, "humidity": 50.0, "fan_pct": 100, "trigger": "threshold_exceeded"}
    },
    {
        "name": "🤖 LSTM Predictive Trigger",
        "data": {"mq135_ppm": 360, "mq2_ppm": 1500, "temp": 27.0, "humidity": 58.0, "fan_pct": 70, "trigger": "lstm_predictive"}
    },
    {
        "name": "💨 Smoke Detected",
        "data": {"mq135_ppm": 450, "mq2_ppm": 2500, "temp": 35.0, "humidity": 40.0, "fan_pct": 100, "trigger": "smoke_detected"}
    },
    {
        "name": "🧪 NH3 High Level",
        "data": {"mq135_ppm": 700, "mq2_ppm": 1000, "temp": 32.0, "humidity": 45.0, "fan_pct": 100, "trigger": "threshold_exceeded"}
    },
]

print("Starting FumeGuard Alert Tests...\n")

for i, test in enumerate(test_cases):
    print(f"[{i+1}/{len(test_cases)}] Sending: {test['name']}")
    try:
        response = requests.post(URL, json=test['data'], timeout=10)
        result = response.json()
        print(f"  → Gas: {result.get('gas_type')} | Anomaly: {result.get('is_anomaly')} | Fan Trigger: {result.get('fan_trigger')}")
        print(f"  → Predicted t+30: {result.get('predicted_ppm_t30')} ppm")
    except Exception as e:
        print(f"  → Error: {e}")
    
    print(f"  Waiting 5 seconds before next test...\n")
    time.sleep(5)  # wait 5 seconds between each so Telegram messages are separate

print("✅ All tests done! Check your Telegram.")