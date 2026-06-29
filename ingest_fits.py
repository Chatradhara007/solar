import os
import glob
import time
import pandas as pd
from astropy.io import fits

RAW_DIR = "data/raw_fits"
CSV_FILE = "data/solexs_all.csv"

def process_fits(filepath):
    print(f"[*] Processing {filepath}...")
    try:
        with fits.open(filepath) as hdul:
            # Assuming data is in the first binary table extension (index 1)
            data = hdul[1].data
            
            # Extract standard columns
            times = data['TIME']
            counts = data['COUNTS']
            
            # Create a raw dataframe (1-second cadence)
            df_raw = pd.DataFrame({'timestamp': times, 'counts': counts}).dropna()
            
            # Convert timestamp (Unix) to datetime for resampling
            df_raw['dt'] = pd.to_datetime(df_raw['timestamp'], unit='s')
            df_raw.set_index('dt', inplace=True)
            
            # Downsample to 10-second cadence using mean
            df_resampled = df_raw.resample('10s').mean().dropna()
            
            # Reconstruct the expected CSV format: timestamp, counts, iso
            df_resampled['iso'] = df_resampled.index.strftime('%Y-%m-%d %H:%M:%S.%f').str[:-3]
            df_final = df_resampled[['timestamp', 'counts', 'iso']].copy()
            
            # Append to our live database
            append_header = not os.path.exists(CSV_FILE)
            df_final.to_csv(CSV_FILE, mode='a', header=append_header, index=False)
            
            print(f"[+] Successfully appended {len(df_final)} 10-second bins to {CSV_FILE}.")
            
            # Move file to 'processed' folder so we don't double-process it
            processed_dir = os.path.join(RAW_DIR, "processed")
            os.makedirs(processed_dir, exist_ok=True)
            new_path = os.path.join(processed_dir, os.path.basename(filepath))
            os.rename(filepath, new_path)
            print(f"[>] Moved to {new_path}\n")
            
    except Exception as e:
        print(f"[!] Error processing {filepath}: {e}")

def watch_directory():
    print(f"Monitoring {RAW_DIR} for incoming Aditya-L1 SoLEXS telemetry (.fits)...")
    while True:
        fits_files = glob.glob(os.path.join(RAW_DIR, "*.fits"))
        for f in fits_files:
            process_fits(f)
        time.sleep(3) # Check every 3 seconds

if __name__ == "__main__":
    os.makedirs(RAW_DIR, exist_ok=True)
    watch_directory()
