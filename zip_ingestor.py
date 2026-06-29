import os
import glob
import time
import zipfile
import pandas as pd
from astropy.io import fits
import shutil

ZIP_DIR = "data/raw_zips"
CSV_FILE = "data/combined_telemetry.csv"

def extract_and_process(zip_path):
    print(f"[*] Extracting {zip_path}...")
    temp_dir = "data/temp_extract"
    os.makedirs(temp_dir, exist_ok=True)
    
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)
            
        fits_files = glob.glob(os.path.join(temp_dir, "**", "*.fits"), recursive=True)
        
        # Group by instrument (mocking the extraction logic)
        solexs_data = None
        hel1os_data = None
        
        for f in fits_files:
            fname = os.path.basename(f).lower()
            with fits.open(f) as hdul:
                data = hdul[1].data
                df_raw = pd.DataFrame({'timestamp': data['TIME'], 'counts': data['COUNTS']}).dropna()
                df_raw['dt'] = pd.to_datetime(df_raw['timestamp'], unit='s')
                df_raw.set_index('dt', inplace=True)
                df_resampled = df_raw.resample('10s').mean().dropna()
                
                if 'solexs' in fname:
                    solexs_data = df_resampled
                elif 'hel1os' in fname:
                    hel1os_data = df_resampled
                    
        if solexs_data is not None and hel1os_data is not None:
            # Sync the instruments using an inner join on the temporal grid
            merged = pd.merge(solexs_data, hel1os_data, left_index=True, right_index=True, suffixes=('_solexs', '_hel1os'))
            
            merged['iso'] = merged.index.strftime('%Y-%m-%d %H:%M:%S.%f').str[:-3]
            df_final = pd.DataFrame({
                'timestamp': merged['timestamp_solexs'],
                'counts': merged['counts_solexs'],
                'hel1os_counts': merged['counts_hel1os'],
                'iso': merged['iso']
            })
            
            append_header = not os.path.exists(CSV_FILE)
            df_final.to_csv(CSV_FILE, mode='a', header=append_header, index=False)
            print(f"[+] Successfully merged and appended {len(df_final)} rows to {CSV_FILE}.")
        else:
            print("[!] ZIP did not contain both instrument files.")
            
        # Move processed zip
        processed_dir = os.path.join(ZIP_DIR, "processed")
        os.makedirs(processed_dir, exist_ok=True)
        os.rename(zip_path, os.path.join(processed_dir, os.path.basename(zip_path)))
        
    except Exception as e:
        print(f"[!] Error processing {zip_path}: {e}")
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

def watch_directory():
    print(f"Monitoring {ZIP_DIR} for incoming dual-instrument .zip telemetry...")
    while True:
        zip_files = glob.glob(os.path.join(ZIP_DIR, "*.zip"))
        for z in zip_files:
            extract_and_process(z)
        time.sleep(3)

if __name__ == "__main__":
    os.makedirs(ZIP_DIR, exist_ok=True)
    watch_directory()
