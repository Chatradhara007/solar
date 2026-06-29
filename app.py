from flask import Flask, render_template, jsonify, request
import pandas as pd
import numpy as np
from scipy.ndimage import uniform_filter1d
import os
import pickle
import torch
import torch.nn as nn
import json

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
# Using the new combined telemetry
df = pd.read_csv("data/combined_telemetry.csv")
df["timestamp"] = df["timestamp"].astype(float)
df["counts"] = df["counts"].astype(float)
df["hel1os_counts"] = df["hel1os_counts"].astype(float)
df["date"] = df["iso"].str[:10]

AVAILABLE_DATES = sorted(df["date"].unique().tolist())

# ── Flare detection algorithm ─────────────────────────────────────────────────
def detect_flares(counts_arr, timestamps, sigma=5, min_duration=6, window_size=180):
    counts = np.array(counts_arr, dtype=float)
    valid = ~np.isnan(counts)

    # Fast-path for entirely empty/zero datasets (e.g. missing HEL1OS data)
    if not np.any(counts[valid] > 0):
        return [], np.full_like(counts, np.nan), np.full_like(counts, np.nan)

    background = np.full_like(counts, np.nan)
    for i in range(len(counts)):
        lo = max(0, i - window_size)
        window = counts[lo:i]
        window = window[~np.isnan(window)]
        if len(window) >= 10:
            background[i] = np.nanpercentile(window, 25)

    std_est = np.array([
        np.nanstd(counts[max(0,i-window_size):i]) if i > 10 else np.nanstd(counts[:10])
        for i in range(len(counts))
    ])
    threshold = background + sigma * std_est

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
                flares.append({
                    "start_time": str(timestamps[flare_start]),
                    "peak_time": str(timestamps[peak_idx]),
                    "end_time": str(timestamps[i - 1]),
                    "peak_counts": float(peak_counts),
                    "duration_s": int(duration * 10),
                    "start_idx": int(flare_start),
                    "peak_idx": int(peak_idx),
                    "end_idx": int(i - 1),
                })
            in_flare = False

    return flares, background, threshold

def classify_flare(peak_counts, background):
    ratio = peak_counts / max(background, 1)
    if ratio >= 200:  return "X"
    elif ratio >= 50: return "M"
    elif ratio >= 15: return "C"
    elif ratio >= 5:  return "B"
    else:             return "A"

def generate_master_catalogue(solexs_flares, hel1os_flares, background_arr):
    master = []
    for s_flare in solexs_flares:
        matched_h = None
        for h_flare in hel1os_flares:
            # Check if hel1os peak occurs during the solexs flare window
            if s_flare["start_idx"] <= h_flare["peak_idx"] <= s_flare["end_idx"] + 10:
                matched_h = h_flare
                break
        
        bg = background_arr[s_flare["start_idx"]]
        goes_class = classify_flare(s_flare["peak_counts"], bg)
        
        master.append({
            "start_time": s_flare["start_time"],
            "peak_time": s_flare["peak_time"],
            "end_time": s_flare["end_time"],
            "solexs_peak": float(s_flare["peak_counts"]),
            "hel1os_peak": float(matched_h["peak_counts"]) if matched_h else None,
            "goes_class": goes_class,
            "verified_by_hel1os": matched_h is not None,
            "start_idx": s_flare["start_idx"],
            "peak_idx": s_flare["peak_idx"],
            "end_idx": s_flare["end_idx"],
            "duration_s": s_flare["duration_s"]
        })
    return master

def forecast_alerts(counts_arr, timestamps, lookahead=90):
    counts = np.array(counts_arr, dtype=float)
    counts_no_nan = np.nan_to_num(counts, nan=np.nanmedian(counts))
    smoothed = uniform_filter1d(counts_no_nan, size=6)
    deriv = np.gradient(smoothed)
    roll_deriv = uniform_filter1d(deriv, size=12)

    alerts = []
    
    if lstm_model is not None and lstm_scaler is not None:
        df_counts = pd.Series(counts)
        roll_std = df_counts.rolling(window=180, min_periods=1).std().fillna(0).values
        
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
                if prob > 0.8 and not in_alert:
                    alerts.append({
                        "alert_time": str(timestamps[i]),
                        "alert_idx": int(i),
                        "lead_time_s": int(lookahead * 10),
                    })
                    in_alert = True
                elif prob < 0.3:
                    in_alert = False
                
        return alerts

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
    hel1os_counts = day["hel1os_counts"].values.astype(float)
    timestamps = day["timestamp"].values.astype(float)
    isos = day["iso"].tolist()

    # Dual-Instrument Detection
    s_flares, s_bg, s_thresh = detect_flares(counts, timestamps, sigma=5)
    h_flares, h_bg, h_thresh = detect_flares(hel1os_counts, timestamps, sigma=8, window_size=30, min_duration=2)
    
    # Generate Master Catalogue
    master_catalogue = generate_master_catalogue(s_flares, h_flares, s_bg)
    
    forecast = forecast_alerts(counts, timestamps)

    smoothed = uniform_filter1d(np.nan_to_num(counts, nan=np.nanmedian(counts)), size=6).tolist()
    smoothed_hel1os = uniform_filter1d(np.nan_to_num(hel1os_counts, nan=np.nanmedian(hel1os_counts)), size=3).tolist()

    # Save to continuous learning buffer if verified flares exist
    verified_flares = [f for f in master_catalogue if f['verified_by_hel1os']]
    if verified_flares:
        try:
            buffer_file = "data/retrain_buffer.json"
            buffer = []
            if os.path.exists(buffer_file):
                with open(buffer_file, 'r') as f:
                    buffer = json.load(f)
            
            for f in verified_flares:
                # Add to buffer if not already present
                if not any(b['peak_time'] == f['peak_time'] for b in buffer):
                    buffer.append(f)
                    
            with open(buffer_file, 'w') as f:
                json.dump(buffer, f)
        except Exception as e:
            print("Error updating retraining buffer:", e)

    return jsonify({
        "timestamps": isos,
        "counts": [None if np.isnan(c) else float(c) for c in counts],
        "hel1os_counts": [None if np.isnan(c) else float(c) for c in hel1os_counts],
        "smoothed": smoothed,
        "smoothed_hel1os": smoothed_hel1os,
        "background": [None if np.isnan(b) else float(b) for b in s_bg],
        "threshold": [None if np.isnan(t) else float(t) for t in s_thresh],
        "flares": master_catalogue,
        "forecasts": forecast,
        "stats": {
            "date": date,
            "max_counts": float(np.nanmax(counts)),
            "baseline": float(np.nanmedian(counts)),
            "n_flares": len(master_catalogue),
            "n_verified": len(verified_flares),
            "n_forecasts": len(forecast),
        }
    })

@app.route("/api/dates")
def get_dates():
    return jsonify(AVAILABLE_DATES)

# ── Chatbot RAG Endpoint ──────────────────────────────────────────────────────
@app.route("/api/chat", methods=["POST"])
def chat():
    """
    Mock RAG Chatbot endpoint. Reads the live data and returns a natural language response.
    """
    data = request.json
    query = data.get("query", "").lower()
    date = data.get("date", AVAILABLE_DATES[0])
    
    day = df[df["date"] == date]
    if day.empty:
        return jsonify({"reply": f"I don't have telemetry data for {date}."})
        
    if "flare" in query or "summarize" in query:
        # Re-run detection for the requested date to answer
        counts = day["counts"].values
        hel1os = day["hel1os_counts"].values
        ts = day["timestamp"].values
        s_flares, _, _ = detect_flares(counts, ts)
        h_flares, _, _ = detect_flares(hel1os, ts, sigma=8, window_size=30, min_duration=2)
        master = generate_master_catalogue(s_flares, h_flares, np.full_like(counts, np.nanmedian(counts)))
        
        if not master:
            return jsonify({"reply": f"The Sun was quiet on {date}. I detected 0 solar flares."})
        
        verified = sum(1 for m in master if m['verified_by_hel1os'])
        biggest = max(master, key=lambda x: x['solexs_peak'])
        
        reply = f"On {date}, I detected **{len(master)} total flares**, out of which **{verified} were verified** by Hard X-ray spikes (HEL1OS).\n\n"
        reply += f"The most intense event was a **Class {biggest['goes_class']}** flare peaking at {biggest['solexs_peak']:.0f} counts at {biggest['peak_time'][11:19]} UTC."
        return jsonify({"reply": reply})
        
    elif "intensity" in query or "peak" in query:
        max_c = day["counts"].max()
        return jsonify({"reply": f"The absolute peak intensity on {date} was **{max_c:.0f} counts** (Soft X-rays)."})
        
    return jsonify({"reply": "I am the Solar Weather Assistant. You can ask me to summarize flares, check peak intensities, or verify Hard X-ray anomalies for the currently selected date."})

if __name__ == "__main__":
    app.run(debug=True, port=5050, host='0.0.0.0')
