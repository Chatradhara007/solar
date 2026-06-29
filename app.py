from flask import Flask, render_template, jsonify, request
import pandas as pd
import numpy as np
from scipy.ndimage import uniform_filter1d
import os
import pickle
import torch
import torch.nn as nn

app = Flask(__name__)

class FlareLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(FlareLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return self.sigmoid(out).squeeze()

lstm_model = None
lstm_scaler = None
if os.path.exists("models/forecast_lstm.pt") and os.path.exists("models/scaler.pkl"):
    try:
        with open("models/scaler.pkl", "rb") as f:
            lstm_scaler = pickle.load(f)
        lstm_model = FlareLSTM(input_size=4, hidden_size=32, num_layers=2)
        lstm_model.load_state_dict(torch.load("models/forecast_lstm.pt", map_location=torch.device('cpu')))
        lstm_model.eval()
        print("Loaded LSTM forecasting model.")
    except Exception as e:
        print(f"Error loading LSTM model: {e}")


# ── Load & prep data once at startup ──────────────────────────────────────────
df = pd.read_csv("data/solexs_all.csv")
df["timestamp"] = df["timestamp"].astype(float)
df["counts"] = df["counts"].astype(float)
df["date"] = df["iso"].str[:10]

AVAILABLE_DATES = sorted(df["date"].unique().tolist())

# ── Flare detection algorithm ─────────────────────────────────────────────────
def detect_flares(counts_arr, timestamps, sigma=5, min_duration=6):
    """
    Nowcasting: detect flares using rolling background + sigma threshold.
    Returns list of flare dicts.
    """
    counts = np.array(counts_arr, dtype=float)
    valid = ~np.isnan(counts)

    # Rolling baseline: 30-min window = 180 points (10s cadence)
    background = np.full_like(counts, np.nan)
    for i in range(len(counts)):
        lo = max(0, i - 180)
        window = counts[lo:i]
        window = window[~np.isnan(window)]
        if len(window) >= 10:
            background[i] = np.nanpercentile(window, 25)  # low percentile = quiet sun

    # Threshold
    std_est = np.array([
        np.nanstd(counts[max(0,i-180):i]) if i > 10 else np.nanstd(counts[:10])
        for i in range(len(counts))
    ])
    threshold = background + sigma * std_est

    # Flag detections
    above = (counts > threshold) & valid
    in_flare = False
    flare_start = None
    flares = []

    for i in range(len(counts)):
        if above[i] and not in_flare:
            in_flare = True
            flare_start = i
        elif not above[i] and in_flare:
            duration = i - flare_start
            if duration >= min_duration:
                seg = counts[flare_start:i]
                peak_idx = flare_start + np.nanargmax(seg)
                peak_counts = counts[peak_idx]
                goes_class = classify_flare(peak_counts, np.nanmedian(background[flare_start:i]))
                flares.append({
                    "start_time": str(timestamps[flare_start]),
                    "peak_time": str(timestamps[peak_idx]),
                    "end_time": str(timestamps[i - 1]),
                    "peak_counts": float(peak_counts),
                    "goes_class": goes_class,
                    "duration_s": int(duration * 10),
                    "start_idx": int(flare_start),
                    "peak_idx": int(peak_idx),
                    "end_idx": int(i - 1),
                })
            in_flare = False

    return flares, background, threshold


def classify_flare(peak_counts, background):
    """Map count ratio to approximate GOES class."""
    ratio = peak_counts / max(background, 1)
    if ratio >= 200:  return "X"
    elif ratio >= 50: return "M"
    elif ratio >= 15: return "C"
    elif ratio >= 5:  return "B"
    else:             return "A"


def forecast_alerts(counts_arr, timestamps, lookahead=90):
    """
    Forecasting: detect rising slope precursors and issue early alerts.
    If LSTM model is available, uses it for 15-minute probability.
    """
    counts = np.array(counts_arr, dtype=float)
    counts_no_nan = np.nan_to_num(counts, nan=np.nanmedian(counts))
    smoothed = uniform_filter1d(counts_no_nan, size=6)
    deriv = np.gradient(smoothed)
    roll_deriv = uniform_filter1d(deriv, size=12)

    alerts = []
    
    if lstm_model is not None and lstm_scaler is not None:
        # LSTM path
        # 4. Rolling std
        df_counts = pd.Series(counts)
        roll_std = df_counts.rolling(window=180, min_periods=1).std().fillna(0).values
        
        # Precompute scaled versions
        counts_scaled = lstm_scaler.transform(counts_no_nan.reshape(-1, 1)).flatten()
        smoothed_scaled = lstm_scaler.transform(smoothed.reshape(-1, 1)).flatten()
        features = np.column_stack([
            counts_scaled,
            smoothed_scaled,
            roll_deriv / 10.0,
            roll_std / 100.0
        ])
        
        window_size = 60
        num_windows = len(counts) - lookahead - window_size
        if num_windows > 0:
            windows = np.array([features[i-window_size:i] for i in range(window_size, len(counts) - lookahead)])
            x_tensor = torch.tensor(windows, dtype=torch.float32)
            with torch.no_grad():
                probs = lstm_model(x_tensor).numpy()
                
            in_alert = False
            for i_offset, prob in enumerate(probs):
                i = i_offset + window_size
                if prob > 0.8 and not in_alert: # 80% confidence threshold
                    alerts.append({
                        "alert_time": str(timestamps[i]),
                        "alert_idx": int(i),
                        "lead_time_s": int(lookahead * 10),
                    })
                    in_alert = True
                elif prob < 0.3:
                    in_alert = False
                
        return alerts

    # Fallback legacy path
    background = np.nanpercentile(counts[:60], 50) if len(counts) > 60 else np.nanmedian(counts)
    deriv_thresh = np.nanstd(deriv[:120]) * 3 if len(deriv) > 120 else np.nanstd(deriv) * 3

    in_alert = False
    for i in range(60, len(counts) - lookahead):
        if roll_deriv[i] > deriv_thresh and counts[i] > background * 1.5 and not in_alert:
            alerts.append({
                "alert_time": timestamps[i],
                "alert_idx": i,
                "lead_time_s": lookahead * 10,
            })
            in_alert = True
        elif roll_deriv[i] < deriv_thresh * 0.3:
            in_alert = False

    return alerts


# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html", dates=AVAILABLE_DATES)


@app.route("/api/data")
def get_data():
    date = request.args.get("date", AVAILABLE_DATES[0])
    day = df[df["date"] == date].copy()

    counts = day["counts"].values.astype(float)
    timestamps = day["timestamp"].values.astype(float)
    isos = day["iso"].tolist()

    flares, background, threshold = detect_flares(counts, timestamps)
    forecast = forecast_alerts(counts, timestamps)

    # Smooth for display
    smoothed = uniform_filter1d(np.nan_to_num(counts, nan=np.nanmedian(counts)), size=6).tolist()

    return jsonify({
        "timestamps": isos,
        "counts": [None if np.isnan(c) else float(c) for c in counts],
        "smoothed": smoothed,
        "background": [None if np.isnan(b) else float(b) for b in background],
        "threshold": [None if np.isnan(t) else float(t) for t in threshold],
        "flares": flares,
        "forecasts": forecast,
        "stats": {
            "date": date,
            "max_counts": float(np.nanmax(counts)),
            "baseline": float(np.nanmedian(counts)),
            "n_flares": len(flares),
            "n_forecasts": len(forecast),
        }
    })


@app.route("/api/dates")
def get_dates():
    return jsonify(AVAILABLE_DATES)


if __name__ == "__main__":
    app.run(debug=True, port=5050)
