from flask import Flask, render_template, jsonify, request
import pandas as pd
import numpy as np
from scipy.ndimage import uniform_filter1d

app = Flask(__name__)

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
                    "start_time": timestamps[flare_start],
                    "peak_time": timestamps[peak_idx],
                    "end_time": timestamps[i - 1],
                    "peak_counts": float(peak_counts),
                    "goes_class": goes_class,
                    "duration_s": duration * 10,
                    "start_idx": flare_start,
                    "peak_idx": peak_idx,
                    "end_idx": i - 1,
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


def forecast_alerts(counts_arr, timestamps, lookahead=30):
    """
    Forecasting: detect rising slope precursors and issue early alerts.
    Uses derivative + rolling mean to catch pre-flare ramp-up.
    Returns list of forecast alert dicts.
    """
    counts = np.array(counts_arr, dtype=float)
    smoothed = uniform_filter1d(np.nan_to_num(counts, nan=np.nanmedian(counts)), size=6)
    deriv = np.gradient(smoothed)
    roll_deriv = uniform_filter1d(deriv, size=12)

    background = np.nanpercentile(counts[:60], 50) if len(counts) > 60 else np.nanmedian(counts)
    deriv_thresh = np.nanstd(deriv[:120]) * 3 if len(deriv) > 120 else np.nanstd(deriv) * 3

    alerts = []
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
