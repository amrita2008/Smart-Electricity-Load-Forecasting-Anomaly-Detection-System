from fastapi import FastAPI
import numpy as np
import joblib
import tensorflow as tf
from tensorflow import keras
import os

app = FastAPI()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "lstm_model")
SCALER_PATH = os.path.join(BASE_DIR, "models", "scaler.pkl")

WINDOW_SIZE = 14

model = None
scaler = None

@app.on_event("startup")
def load_model():
    global model, scaler
    model = keras.layers.TFSMLayer(
        MODEL_PATH,
        call_endpoint="serving_default"
    )
    scaler = joblib.load(SCALER_PATH)

@app.get("/")
def home():
    return {"status": "Electricity Load Forecasting API is running"}

@app.post("/predict")
def predict(data: dict):
    values = np.array(data["past_values"]).reshape(-1, 1)

    if len(values) != WINDOW_SIZE:
        return {"error": "Provide exactly 14 past values"}

    scaled = scaler.transform(values)
    X = tf.constant(scaled.reshape(1, WINDOW_SIZE, 1), dtype=tf.float32)

    prediction = model(X)["output_0"]
    predicted_value = scaler.inverse_transform(prediction.numpy())[0][0]

    return {
        "predicted_next_day_load": float(predicted_value)
    }
