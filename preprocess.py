"""
preprocess.py — Run this ONCE before starting the app.
Reads all SoLEXS L1 FITS light curve files from the data/raw/ directory
and outputs a single CSV (data/solexs_all.csv) that the app loads at startup.

Usage:
    python preprocess.py --raw_dir data/raw/

Directory structure expected:
    data/raw/
        AL1_SLX_L1_YYYYMMDD_v1.0/
            SDD2/
                AL1_SOLEXS_YYYYMMDD_SDD2_L1.lc   (or .lc.gz)
"""

import argparse
import glob
import gzip
import os
import shutil
import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.time import Time


def load_lc(path):
    """Load a SoLEXS L1 light curve FITS file. Handles .lc and .lc.gz."""
    if path.endswith(".gz"):
        with gzip.open(path, "rb") as f:
            hdul = fits.open(f)
            times = hdul[1].data["TIME"].astype(np.float64)
            counts = hdul[1].data["COUNTS"].astype(np.float64)
            hdul.close()
    else:
        hdul = fits.open(path)
        times = hdul[1].data["TIME"].astype(np.float64)
        counts = hdul[1].data["COUNTS"].astype(np.float64)
        hdul.close()
    return times, counts


def main(raw_dir):
    lc_files = sorted(
        glob.glob(os.path.join(raw_dir, "**", "*SDD2*L1.lc"), recursive=True)
        + glob.glob(os.path.join(raw_dir, "**", "*SDD2*L1.lc.gz"), recursive=True)
    )

    if not lc_files:
        print(f"No SDD2 LC files found under {raw_dir}")
        return

    all_dfs = []
    for lc_path in lc_files:
        print(f"  Loading {os.path.basename(lc_path)} ...", end=" ")
        try:
            times, counts = load_lc(lc_path)
            df = pd.DataFrame({"timestamp": times, "counts": counts})
            # 10s binning
            df["bin"] = (df["timestamp"] // 10).astype(int)
            df_10s = df.groupby("bin").agg({"timestamp": "first", "counts": "mean"}).reset_index(drop=True)
            t10 = Time(df_10s["timestamp"].values, format="unix", scale="utc")
            df_10s["iso"] = t10.iso
            all_dfs.append(df_10s)
            print(f"{len(df_10s)} rows")
        except Exception as e:
            print(f"ERROR: {e}")

    if not all_dfs:
        print("No data loaded.")
        return

    full = pd.concat(all_dfs, ignore_index=True).sort_values("timestamp").reset_index(drop=True)
    out = "data/solexs_all.csv"
    os.makedirs("data", exist_ok=True)
    full.to_csv(out, index=False)
    print(f"\nSaved {len(full)} rows to {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir", default="data/raw", help="Directory containing unzipped SoLEXS L1 files")
    args = parser.parse_args()
    main(args.raw_dir)
