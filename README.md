# SoLEXS FlareWatch — Aditya-L1 Solar Flare Demo
**Team Pixel Lords | ISRO BAH 2026**

Real-time solar flare nowcasting and forecasting dashboard using SoLEXS (Aditya-L1) Level-1 data.

---

## Setup

```bash
pip install -r requirements.txt
```

## First-time data prep

If you have new SoLEXS zips:
1. Unzip all `AL1_SLX_L1_YYYYMMDD_v1_0.zip` files into `data/raw/`
2. Run: `python preprocess.py --raw_dir data/raw/`

The included `data/solexs_all.csv` already contains Jun 12–25 2026 data.

## Run

```bash
python app.py
```

Open **http://localhost:5050**

---

## What the demo shows

| Feature | Description |
|---|---|
| **Date selector** | Browse any of the 14 loaded days |
| **Light curve** | SoLEXS SDD2 raw + smoothed counts at 10s cadence |
| **Background** | Rolling 30-min baseline (25th percentile) |
| **5σ threshold** | Detection threshold above background |
| **Nowcast alerts** | Red flare regions + star markers at peaks |
| **Forecast alerts** | Blue triangle markers = precursor ramp-up detected |
| **Flare catalogue** | Sidebar list with GOES-equivalent class + click-to-zoom |
| **Log scale toggle** | See weak and strong flares on same view |
| **Range slider** | Bottom slider for fine time navigation |

## Key dates to demo

| Date | Activity |
|---|---|
| **2026-06-21** | 2× X-class flares (19:28 UTC = 4855 cts peak) |
| **2026-06-20** | M-class + multiple C/B flares |
| **2026-06-23** | Active day, multiple flare clusters |
| **2026-06-17** | Quiet sun baseline |

## Algorithm

**Nowcasting:**
- Rolling 30-min background (25th percentile of window = below flares)
- Robust σ from quiet fraction of window (below 75th percentile)
- Flag: counts > background + 5σ for ≥60s → flare event
- GOES-class proxy from count-to-background ratio

**Forecasting:**
- Smooth counts → compute derivative
- Alert when rolling derivative > 3σ AND counts already above 1.5× background
- Lead time: configurable (default 5 min lookahead window)
