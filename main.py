from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import sqlite3
import numpy as np
import joblib
import tensorflow as tf
from datetime import datetime
import requests
import os
from collections import deque
import threading

# ─────────────────────────────────────────
# CONFIG — Load from environment or .env
# ─────────────────────────────────────────
from dotenv import load_dotenv
load_dotenv()

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
DANGER_THRESHOLD_MQ135 = float(os.getenv("DANGER_THRESHOLD_MQ135", 500))
DANGER_THRESHOLD_MQ2 = float(os.getenv("DANGER_THRESHOLD_MQ2", 3000))
PREDICTIVE_TRIGGER_PCT = 0.70  # fan triggers at 70% of danger threshold

# ─────────────────────────────────────────
# LOAD ML MODELS
print("Loading ML models...")
lstm_model = None
iso_forest = None
gas_clf = None
scaler_X = None
scaler_y = None

try:
    lstm_model = tf.keras.models.load_model('models/lstm_model.keras')
    scaler_X = joblib.load('models/scaler_X.pkl')
    scaler_y = joblib.load('models/scaler_y.pkl')
    iso_forest = joblib.load('models/isolation_forest.pkl')
    gas_clf = joblib.load('models/gas_classifier.pkl')
    print("✅ All models loaded")
except Exception as e:
    print(f"⚠️ Model loading error: {e}")
    print(f"⚠️ Model loading error: {e}")
    lstm_model = None

# ─────────────────────────────────────────
# DATABASE SETUP
# ─────────────────────────────────────────
def init_db():
    conn = sqlite3.connect('fumeguard.db')
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS readings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            mq135_ppm REAL,
            mq2_ppm REAL,
            temp REAL,
            humidity REAL,
            hour INTEGER,
            fan_pct INTEGER,
            trigger TEXT,
            anomaly_flag INTEGER DEFAULT 0,
            gas_type TEXT
        )
    ''')
    c.execute('''
        CREATE TABLE IF NOT EXISTS incidents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            trigger_reason TEXT,
            mq135_ppm REAL,
            mq2_ppm REAL,
            gas_type TEXT,
            fan_pct INTEGER,
            pdf_path TEXT
        )
    ''')
    conn.commit()
    conn.close()

init_db()

# ─────────────────────────────────────────
# SLIDING WINDOW (for LSTM — needs 30 readings)
# ─────────────────────────────────────────
reading_buffer = deque(maxlen=30)
buffer_lock = threading.Lock()

# ─────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────
def send_telegram_alert(message: str):
    """Send push notification via Telegram Bot API."""
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print("⚠️ Telegram not configured")
        return
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "HTML"}
        resp = requests.post(url, json=payload, timeout=5)
        if resp.status_code == 200:
            print(f"✅ Telegram alert sent")
        else:
            print(f"⚠️ Telegram error: {resp.text}")
    except Exception as e:
        print(f"⚠️ Telegram failed: {e}")

def log_incident(trigger_reason, mq135, mq2, gas_type, fan_pct):
    """Log an incident to DB and generate PDF."""
    ts = datetime.now().isoformat()
    pdf_path = generate_incident_pdf(ts, trigger_reason, mq135, mq2, gas_type, fan_pct)
    conn = sqlite3.connect('fumeguard.db')
    c = conn.cursor()
    c.execute('''
        INSERT INTO incidents (timestamp, trigger_reason, mq135_ppm, mq2_ppm, gas_type, fan_pct, pdf_path)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', (ts, trigger_reason, mq135, mq2, gas_type, fan_pct, pdf_path))
    conn.commit()
    conn.close()
    return pdf_path

def generate_incident_pdf(timestamp, trigger_reason, mq135, mq2, gas_type, fan_pct):
    """Generate a PDF incident report using ReportLab."""
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib import colors
        from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
        from reportlab.lib.styles import getSampleStyleSheet

        os.makedirs('reports', exist_ok=True)
        safe_ts = timestamp.replace(":", "-").replace(".", "-")
        pdf_path = f"reports/incident_{safe_ts}.pdf"

        doc = SimpleDocTemplate(pdf_path, pagesize=A4)
        styles = getSampleStyleSheet()
        elements = []

        # Title
        elements.append(Paragraph("🛡 FumeGuard AI — Incident Report", styles['Title']))
        elements.append(Spacer(1, 12))

        # Info table
        data = [
            ["Field", "Value"],
            ["Timestamp", timestamp],
            ["Trigger Reason", trigger_reason],
            ["Gas Type Detected", gas_type],
            ["MQ-135 PPM", f"{mq135:.1f}"],
            ["MQ-2 PPM", f"{mq2:.1f}"],
            ["Fan Speed", f"{fan_pct}%"],
            ["Danger Threshold MQ135", f"{DANGER_THRESHOLD_MQ135} PPM"],
        ]

        table = Table(data, colWidths=[200, 250])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.darkred),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 11),
            ('BACKGROUND', (0, 1), (-1, -1), colors.lightyellow),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightyellow]),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('PADDING', (0, 0), (-1, -1), 8),
        ]))
        elements.append(table)
        elements.append(Spacer(1, 20))
        elements.append(Paragraph("Generated by FumeGuard AI — Lab Safety System", styles['Normal']))

        doc.build(elements)
        print(f"✅ PDF generated: {pdf_path}")
        return pdf_path
    except Exception as e:
        print(f"⚠️ PDF generation failed: {e}")
        return ""

# ─────────────────────────────────────────
# FASTAPI APP
# ─────────────────────────────────────────
app = FastAPI(title="FumeGuard AI API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────
# MODELS (Pydantic — data structure validation)
# ─────────────────────────────────────────
class SensorReading(BaseModel):
    mq135_ppm: float
    mq2_ppm: float
    temp: float
    humidity: float
    fan_pct: int = 0
    trigger: str = "none"

class PredictionResponse(BaseModel):
    predicted_ppm_t30: float
    predicted_ppm_t60: float
    anomaly_score: float
    is_anomaly: bool
    gas_type: str
    fan_trigger: bool
    trigger_reason: str

# ─────────────────────────────────────────
# ENDPOINTS
# ─────────────────────────────────────────

@app.get("/")
def root():
    return {"status": "FumeGuard AI API is running 🛡", "version": "1.0.0"}

@app.get("/health")
def health():
    return {
        "status": "ok",
        "models_loaded": lstm_model is not None,
        "buffer_size": len(reading_buffer)
    }

@app.post("/data")
def post_data(reading: SensorReading):
    """
    ESP32 sends sensor data here every 5 seconds.
    We store it, run ML predictions, and return results.
    """
    ts = datetime.now().isoformat()
    hour = datetime.now().hour

    # ── 1. Classify gas type ──
    gas_type = "Unknown"
    try:
        clf_input = np.array([[reading.mq135_ppm, reading.mq2_ppm, reading.temp, reading.humidity]])
        gas_type = gas_clf.predict(clf_input)[0]
    except Exception as e:
        print(f"Gas classifier error: {e}")

    # ── 2. Anomaly detection ──
    is_anomaly = False
    anomaly_score = 0.0
    try:
        with buffer_lock:
            if len(reading_buffer) > 0:
                prev_ppm = reading_buffer[-1][0]
            else:
                prev_ppm = reading.mq135_ppm

        ppm_delta = reading.mq135_ppm - prev_ppm
        rolling_vals = [r[0] for r in reading_buffer][-5:] + [reading.mq135_ppm]
        rolling_std = float(np.std(rolling_vals)) if len(rolling_vals) > 1 else 0.0

        iso_input = np.array([[ppm_delta, rolling_std, hour]])
        iso_pred = iso_forest.predict(iso_input)[0]
        anomaly_score = float(-iso_forest.decision_function(iso_input)[0])
        is_anomaly = (iso_pred == -1)
    except Exception as e:
        print(f"Isolation Forest error: {e}")

    # ── 3. Add to sliding window ──
    with buffer_lock:
        reading_buffer.append([
            reading.mq135_ppm, reading.mq2_ppm,
            reading.temp, reading.humidity, hour
        ])

    # ── 4. LSTM Prediction ──
    predicted_t30 = reading.mq135_ppm
    predicted_t60 = reading.mq135_ppm
    fan_trigger = False
    trigger_reason = "none"

    if lstm_model and len(reading_buffer) >= 30:
        try:
            with buffer_lock:
                window_data = list(reading_buffer)

            X_input = scaler_X.transform(window_data)
            X_input = np.expand_dims(X_input, axis=0)  # shape: (1, 30, 5)
            pred_scaled = lstm_model.predict(X_input, verbose=0)
            pred_real = scaler_y.inverse_transform(pred_scaled)[0]
            predicted_t30 = float(pred_real[0])
            predicted_t60 = float(pred_real[1])

            # Fan triggers if predicted value is 70% of danger threshold
            if predicted_t30 >= PREDICTIVE_TRIGGER_PCT * DANGER_THRESHOLD_MQ135:
                fan_trigger = True
                trigger_reason = "lstm_predictive"
        except Exception as e:
            print(f"LSTM prediction error: {e}")

    # ── 5. Override: immediate danger threshold ──
    if reading.mq135_ppm >= DANGER_THRESHOLD_MQ135 or reading.mq2_ppm >= DANGER_THRESHOLD_MQ2:
        fan_trigger = True
        trigger_reason = "threshold_exceeded"

    if is_anomaly:
        trigger_reason = f"{trigger_reason}+anomaly" if trigger_reason != "none" else "anomaly"

    # ── 6. Save to database ──
    conn = sqlite3.connect('fumeguard.db')
    c = conn.cursor()
    c.execute('''
        INSERT INTO readings (timestamp, mq135_ppm, mq2_ppm, temp, humidity, hour, fan_pct, trigger, anomaly_flag, gas_type)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (ts, reading.mq135_ppm, reading.mq2_ppm, reading.temp, reading.humidity,
          hour, reading.fan_pct, reading.trigger, int(is_anomaly), gas_type))
    conn.commit()
    conn.close()

    # ── 7. Send alerts if needed ──
    if fan_trigger or is_anomaly:
        alert_msg = (
            f"🚨 <b>FumeGuard AI Alert!</b>\n"
            f"📍 Lab B-204\n"
            f"🕐 {ts[:19]}\n"
            f"💨 Gas: <b>{gas_type}</b>\n"
            f"📊 MQ-135: {reading.mq135_ppm:.0f} ppm | MQ-2: {reading.mq2_ppm:.0f} ppm\n"
            f"🌡 Temp: {reading.temp:.1f}°C\n"
            f"💨 Fan: {reading.fan_pct}%\n"
            f"⚠️ Trigger: {trigger_reason}\n"
            f"🔮 Predicted t+30: {predicted_t30:.0f} ppm"
        )
        send_telegram_alert(alert_msg)
        log_incident(trigger_reason, reading.mq135_ppm, reading.mq2_ppm, gas_type, reading.fan_pct)

    return {
        "status": "ok",
        "timestamp": ts,
        "gas_type": gas_type,
        "is_anomaly": is_anomaly,
        "anomaly_score": round(anomaly_score, 3),
        "predicted_ppm_t30": round(predicted_t30, 1),
        "predicted_ppm_t60": round(predicted_t60, 1),
        "fan_trigger": fan_trigger,
        "trigger_reason": trigger_reason
    }

@app.get("/history")
def get_history(limit: int = 100):
    """Return last N readings for dashboard charts."""
    conn = sqlite3.connect('fumeguard.db')
    c = conn.cursor()
    c.execute('''
        SELECT timestamp, mq135_ppm, mq2_ppm, temp, humidity, fan_pct, trigger, anomaly_flag, gas_type
        FROM readings ORDER BY id DESC LIMIT ?
    ''', (limit,))
    rows = c.fetchall()
    conn.close()

    keys = ['timestamp', 'mq135_ppm', 'mq2_ppm', 'temp', 'humidity', 'fan_pct', 'trigger', 'anomaly_flag', 'gas_type']
    return {"data": [dict(zip(keys, row)) for row in reversed(rows)]}

@app.get("/predict")
def get_predict():
    """Return the latest LSTM prediction (for dashboard)."""
    with buffer_lock:
        buf_size = len(reading_buffer)

    if not lstm_model or buf_size < 30:
        return {
            "status": "waiting",
            "message": f"Need 30 readings. Have {buf_size}.",
            "predicted_ppm_t30": None,
            "predicted_ppm_t60": None
        }

    try:
        with buffer_lock:
            window_data = list(reading_buffer)
        X_input = scaler_X.transform(window_data)
        X_input = np.expand_dims(X_input, axis=0)
        pred_scaled = lstm_model.predict(X_input, verbose=0)
        pred_real = scaler_y.inverse_transform(pred_scaled)[0]
        return {
            "status": "ok",
            "predicted_ppm_t30": round(float(pred_real[0]), 1),
            "predicted_ppm_t60": round(float(pred_real[1]), 1),
            "buffer_size": buf_size
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/incidents")
def get_incidents(limit: int = 50):
    """Return incident log for dashboard."""
    conn = sqlite3.connect('fumeguard.db')
    c = conn.cursor()
    c.execute('SELECT * FROM incidents ORDER BY id DESC LIMIT ?', (limit,))
    rows = c.fetchall()
    conn.close()
    keys = ['id', 'timestamp', 'trigger_reason', 'mq135_ppm', 'mq2_ppm', 'gas_type', 'fan_pct', 'pdf_path']
    return {"incidents": [dict(zip(keys, row)) for row in rows]}