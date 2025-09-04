import pandas as pd
import requests, io, pickle
from app.config import DATAFRAMES

# ---------- Drive helpers ----------

def make_drive_url(file_id: str) -> str:
    """
    Build a direct-download URL for a Google Drive file ID.
    NOTE: The Drive file must be shared as “Anyone with the link can view”.
    """
    return f"https://drive.google.com/uc?export=download&id={file_id}"

def load_csv_from_url(url: str) -> pd.DataFrame:
    """
    Download a CSV from a direct URL and read it into a DataFrame.
    Uses requests for HTTP and StringIO to feed text to pandas.
    """
    r = requests.get(url)
    r.raise_for_status()                 # raise if HTTP request failed (e.g., 403/404)
    return pd.read_csv(io.StringIO(r.text))

# ---------- Data loading ----------

def read_metadata_files():
    """
    Load the 5 metadata tables used for context/joins:
      - holiday_events, items, oil, stores, transactions
    Returns 5 DataFrames in that order.
    """
    with open(DATAFRAMES, 'rb') as f:
        dataframes = pickle.load(f)

    df_holiday_events = dataframes['df_holiday_events']
    df_items = dataframes['df_items']
    df_oil = dataframes['df_oil']
    df_stores = dataframes['df_stores']
    df_transactions = dataframes['df_transactions']
    df_filtered = dataframes['df_filtered']
    return df_holiday_events, df_items, df_oil, df_stores, df_transactions, df_filtered

def load_data():
    """
    Download metadata CSVs and the filtered training series (pickled DataFrame).

    Returns (in order):
        df_stores, df_items, df_transactions, df_oil, df_holiday_events, df_filtered
    """
    # Load the five metadata tables
    df_holiday_events, df_items, df_oil, df_stores, df_transactions, df_train = read_metadata_files()

    return df_stores, df_items, df_transactions, df_oil, df_holiday_events, df_train

# ---------- Feature engineering ----------

def creating_features(df_filtered: pd.DataFrame) -> pd.DataFrame:
    """
    Add features the LSTM expects.
    Assumes:
      - 'unit_sales' column exists
    """
    df = df_filtered.copy()

    df.set_index("date", inplace=True)

    # Target lags: yesterday, last week, last month (by day count)
    df["lag_1"]  = df["unit_sales"].shift(1)
    df["lag_7"]  = df["unit_sales"].shift(7)
    df["lag_30"] = df["unit_sales"].shift(30)
    df.dropna(inplace=True)  # drop rows made NaN by the shifts above

    # Calendar features derived from the index
    df["day_of_week"] = df.index.dayofweek     # 0=Mon ... 6=Sun
    df["month"]       = df.index.month
    df["is_weekend"]  = (df["day_of_week"] >= 5).astype(int)

    # Rolling stats on the target (shift by 1 to avoid peeking into the current day)
    df["rolling_mean_7"] = df["unit_sales"].shift(1).rolling(window=7).mean()
    df["rolling_std_7"]  = df["unit_sales"].shift(1).rolling(window=7).std()

    # Drop any rows still NaN due to rolling window warm-up
    df.dropna(inplace=True)
    return df

def preprocess_input_data(df_filtered: pd.DataFrame, pickled: bool = True) -> pd.DataFrame:
    """
    Prepare a DataFrame for inference.

    If 'pickled' is True:
      - assume df_filtered is already daily, indexed by date, and aggregated.

    If 'pickled' is False:
      - parse 'date' column to datetime
      - aggregate to daily totals
      - set daily frequency and fill missing days with zeros
      - then build features
    """
    if not pickled:
        df = df_filtered.copy()
        df["date"] = pd.to_datetime(df["date"])
        # Aggregate to daily totals (numeric_only guards against non-numeric columns)
        df = df.groupby("date").sum(numeric_only=True)["unit_sales"].reset_index()
        df.set_index("date", inplace=True)
        df = df.asfreq("D").fillna(0)  # fill gaps with 0 sales
    else:
        df = df_filtered

    return df
