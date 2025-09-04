import json
import pickle
import mlflow
import mlflow.keras
import tensorflow as tf

from app.config import (
    MLFLOW_TRACKING_URI,  # file:///…/mlflow_results
    MODEL_URI,            # runs:/<run_id>/model
    MODEL_PATH,           # models/lstm_model.keras  (local fallback)
    SCALER_PATH,          # models/scaler.pkl
    FEATURES_JSON,        # models/feature_cols.json
)

def _mlflow_setup():
    """
    Point MLflow at your local tracking store so `runs:/…` can be resolved.
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

def load_lstm_model():
    """
    Load the Keras model.
    - First try MLflow with MODEL_URI (runs:/<run_id>/model).
    - If that fails (e.g., wrong URI/path), fall back to the local MODEL_PATH.
    """
    _mlflow_setup()
    try:
        return mlflow.keras.load_model(MODEL_URI)
    except Exception as e:
        print(f"[MLflow load failed] {e}\\nFalling back to local MODEL_PATH: {MODEL_PATH}")
        return tf.keras.models.load_model(MODEL_PATH)

def load_scaler_and_features():
    """
    Load the preprocessing artifacts required for inference:
    - scaler.pkl : the fitted scaler used during training
    - feature_cols.json : the exact feature order expected by the model
    """
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)
    with open(FEATURES_JSON) as f:
        feature_cols = json.load(f)
    return scaler, feature_cols

def predict_scaled(model, X_3d):
    """
    Run the model on a 3D tensor shaped (batch, seq_len, n_features).
    Returns a scalar (float) prediction in the **scaled space**.
    """
    return model.predict(X_3d, verbose=0).ravel()[0]
