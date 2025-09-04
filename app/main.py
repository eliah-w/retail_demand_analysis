# --- Make project packages importable (app, data, model) ---
# Add the project root (…/corporacion_favorita) to sys.path so we can do:
#   from app.config import ...
#   from data.data_utils import ...
#   from model.model_utils import ...
import sys, os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
# ------------------------------------------------------------

# Core libs and UI
import streamlit as st
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Project config and modules
from app.config import SEQ_LEN          # SEQ_LEN = LSTM window
from data.data_utils import load_data, preprocess_input_data
from model.model_utils import load_lstm_model, load_scaler_and_features, predict_scaled

# Fixed series selection (the pickled df is already filtered to this pair)
# These are just displayed in the UI; the actual df is already filtered.
FIXED_STORE_ID = 44
FIXED_ITEM_ID  = 1047679

# Streamlit page setup
st.set_page_config(page_title="Demand Forecast App", layout="wide")

# ---------- Helpers for multi-day forecasting ----------

def _predict_next(model, scaler, feature_cols, history, seq_len):
    """
    Predict next day's unit_sales using the last `seq_len` rows from `history`.

    Steps:
      1) take last seq_len rows and select the trained feature columns
      2) scale with the saved scaler
      3) reshape to 3D (batch, seq_len, n_features) and run the model
      4) inverse-scale ONLY the target (assumed to be the FIRST column)
    """
    window = history.tail(seq_len)
    X_win = window[feature_cols].to_numpy()
    X_scaled = scaler.transform(X_win)
    X_3d = X_scaled.reshape(1, seq_len, len(feature_cols))
    y_scaled = predict_scaled(model, X_3d)

    # Inverse-scale just the target: place the scaled y in column 0, zeros elsewhere.
    pad = np.zeros((1, len(feature_cols)))
    pad[:, 0] = y_scaled
    y = float(scaler.inverse_transform(pad)[0, 0])
    return y

def forecast_horizon(model, scaler, feature_cols, feats, start_date, seq_len, horizon=7):
    """
    Autoregressive forecast for `horizon` days, starting the day *after* `start_date`.
    Feed each prediction back into the working series to produce the next step.

    `feats` is the engineered feature DataFrame with a DatetimeIndex and a 'unit_sales' column.
    """
    # Work on a copy up to the chosen cut-off date
    work = feats.loc[:start_date].copy()
    preds = []
    current = pd.to_datetime(start_date)

    for _ in range(horizon):
        # 1) Predict the next day using the current history window
        y = _predict_next(model, scaler, feature_cols, work, seq_len)
        next_date = current + pd.Timedelta(days=1)
        preds.append((next_date, y))

        # 2) Create the new row of engineered features for `next_date`
        #    Note: rolling features are computed from past ACTUALS ONLY (no leakage).
        last = work["unit_sales"].iloc[-1]
        last7  = work["unit_sales"].tail(7)
        last30 = work["unit_sales"].tail(30)
        new = {
            "unit_sales":      y,                                # we append the prediction as if it were the next actual
            "lag_1":           last,                             # yesterday's actual
            "lag_7":           (work["unit_sales"].iloc[-7]  if len(work) >= 7  else last),
            "lag_30":          (work["unit_sales"].iloc[-30] if len(work) >= 30 else last),
            "day_of_week":     next_date.dayofweek,
            "month":           next_date.month,
            "is_weekend":      1 if next_date.dayofweek >= 5 else 0,
            # Training used shift(1).rolling(7) → compute from past actuals only (no inclusion of y)
            "rolling_mean_7":  last7.mean() if len(last7) >= 1 else np.nan,
            "rolling_std_7":   last7.std(ddof=1) if len(last7) >= 2 else 0.0,
        }
        # Append the synthetic row to extend the working history and move forward one day
        work = pd.concat([work, pd.DataFrame([new], index=[next_date])])
        current = next_date

    # Collect predictions into a DataFrame indexed by date
    fcst = pd.DataFrame(preds, columns=["date", "prediction"]).set_index("date")
    return fcst, work

# ---------- App ----------

def main():
    st.title("Sales Forecasting")

    # 1) Data → engineered features
    # Load metadata + filtered series from Drive, then build the LSTM features.
    try:
        _stores, _items, _tx, _oil, _hol, df_train = load_data()
    except Exception as e:
        # If you see this: check Drive sharing/IDs; the 'train' file must allow "Anyone with the link can view".
        st.error(f"Failed to load data. Check Drive sharing and FILE_IDS. Details:\n{e}")
        st.stop()

    feats = preprocess_input_data(df_train)  # adds lags/rolling/calendar features

    # 2) Model + scaler + feature order
    # Try to load from MLflow (runs:/...) and fall back to local files if needed.
    try:
        model = load_lstm_model()
        scaler, feature_cols = load_scaler_and_features()
    except Exception as e:
        st.error(f"Failed to load model or artifacts. Details:\n{e}")
        st.stop()

    # 3) Date & horizon selection
    # Choose a cut-off date (use history up to this date). Forecast starts the next day.
    default_date = datetime.date(2014, 3, 1)
    date = st.date_input(
        "Forecast cut‑off (use history up to this date)",
        value=default_date,
        min_value=feats.index.min().date(),
        max_value=feats.index.max().date(),
    )
    ts = pd.to_datetime(date)

    # Single day vs multi-day (N) selection
    mode = st.radio("Forecast mode", ["Single day", "Next N days"], horizontal=True)
    horizon = st.slider("N days", 1, 30, 7) if mode == "Next N days" else 1

    # 4) Predict
    if st.button("Get Forecast"):
        # Validate the chosen date and ensure we have at least SEQ_LEN days of history
        if ts not in feats.index:
            st.error("Date out of range."); st.stop()
        window = feats.loc[:ts].tail(SEQ_LEN)
        if len(window) < SEQ_LEN:
            st.error(f"Need {SEQ_LEN} days of history before {date}."); st.stop()

        if horizon == 1:
            # Single-day forecast (original behavior)
            X_win = window[feature_cols].to_numpy()
            X_scaled = scaler.transform(X_win)
            X_3d = X_scaled.reshape(1, SEQ_LEN, len(feature_cols))
            y_pred_scaled = predict_scaled(model, X_3d)

            # Inverse-scale only the target dimension
            pad = np.zeros((1, len(feature_cols))); pad[:, 0] = y_pred_scaled
            y_pred = float(scaler.inverse_transform(pad)[0, 0])

            st.success(f"Predicted sales for {date}: {y_pred:.2f}")
            # For consistency with plotting/table, put the result on the NEXT day
            fcst = pd.DataFrame({"prediction": [y_pred]},
                                index=[ts + pd.Timedelta(days=1)])
            work = feats
        else:
            # Autoregressive N-day forecast
            fcst, work = forecast_horizon(model, scaler, feature_cols, feats, ts, SEQ_LEN, horizon)
            st.success(f"Predicted {horizon} days: {fcst.index[0].date()} → {fcst.index[-1].date()}.")

        # 5) Plot: last ~6 months of history + forecast overlay
        hist = feats.loc[max(feats.index.min(), ts - pd.Timedelta(days=180)): ts]["unit_sales"]
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.plot(hist.index, hist.values, label="Actual (history)")
        ax.plot(fcst.index, fcst["prediction"].values, marker="o", label="Forecast")
        ax.axvline(ts, ls="--", alpha=0.5)  # vertical line at cut-off date
        ax.set_xlabel("Date"); ax.set_ylabel("Unit sales"); ax.legend()
        st.pyplot(fig, clear_figure=True)

        # 6) Show forecast table and provide a CSV download
        st.dataframe(fcst.rename_axis("date"))
        st.download_button(
            "Download forecast CSV",
            fcst.rename_axis("date").to_csv(),
            file_name=f"forecast_{horizon}d_from_{ts.date()}.csv",
            mime="text/csv",
        )

# Standard Python entry point so the script can be run directly
if __name__ == "__main__":
    main()
