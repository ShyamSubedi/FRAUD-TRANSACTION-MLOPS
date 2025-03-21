
from fastapi import FastAPI, Request
import xgboost as xgb
import pandas as pd
from datetime import datetime
import sqlite3

app = FastAPI()

model = xgb.Booster()
model.load_model("model.load_model("fraud_detection_v3_model.json")")

DEFAULT_FEATURES = {
    'step': 743,
    'amount': 0.0,
    'isFlaggedFraud': 0,
    'isMerchant': 0,
    'amount_ratio': 1.0,
    'type_encoded': 4
}

FEATURE_ORDER = [
    'step', 'amount', 'isFlaggedFraud',
    'isMerchant', 'amount_ratio', 'type_encoded'
]

conn = sqlite3.connect("logs.db", check_same_thread=False)
cursor = conn.cursor()
cursor.execute("""
CREATE TABLE IF NOT EXISTS logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT,
    amount REAL,
    prediction INTEGER,
    probability REAL
)
""")
conn.commit()

@app.get("/health")
def health():
    return {"status": "API is running with updated defaults"}

@app.post("/predict/")
async def predict(payload: dict):
    try:
        amount = payload.get("amount", 0.0)
        features = DEFAULT_FEATURES.copy()
        features["amount"] = amount

        df = pd.DataFrame([features])[FEATURE_ORDER]
        dmatrix = xgb.DMatrix(df)

        prob = float(model.predict(dmatrix)[0])
        prediction = int(prob > 0.5)

        timestamp = datetime.utcnow().isoformat()
        cursor.execute(
            "INSERT INTO logs (timestamp, amount, prediction, probability) VALUES (?, ?, ?, ?)",
            (timestamp, amount, prediction, prob)
        )
        conn.commit()

        return {"fraud_prediction": prediction, "fraud_probability": prob}
    except Exception as e:
        return {"error": str(e)}

@app.get("/logs/")
def get_logs():
    df = pd.read_sql_query("SELECT * FROM logs ORDER BY id DESC", conn)
    return df.to_dict(orient="records")
