import os

BASE_DIR = os.path.dirname(os.path.dirname(__file__))

MLFLOW_TRACKING_URI = 'file:///Users/eliah/git/retail_demand_analysis/resources/mlflow'

MODEL_URI = 'runs:/d1d556c81b20417f818370bacc7b1460/model'

MODEL_PATH    = os.path.join(BASE_DIR, "mlflow_results/models", "model.keras")
SCALER_PATH   = os.path.join(BASE_DIR, "mlflow_results/models", "scaler.pkl")
FEATURES_JSON = os.path.join(BASE_DIR, "mlflow_results/models", "feature_cols.json")

SEQ_LEN = 14

DATAFRAMES = os.path.join(BASE_DIR, "resources", "2_dataframes.pkl")