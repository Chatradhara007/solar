import os
import glob
import time
import zipfile
import pandas as pd
import numpy as np
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
        
        solexs_data = None
        hel1os_data = None
        
        for f in fits_files:
            fname = os.path.basename(f).lower()
            
            # Skip Good Time Interval (GTI) and Housekeeping (HK) files
            if fname.endswith('.gti') or fname.endswith('.hk'):
                print(f"    -> Skipping auxiliary file: {fname}")
                continue
                
            try:
                with fits.open(f) as hdul:
                    data = hdul[1].data
                    
                    # Ensure it's a lightcurve by checking column names
                    if 'TIME' not in data.columns.names or 'COUNTS' not in data.columns.names:
                        print(f"    -> Skipping {fname} (Missing TIME or COUNTS columns)")
                        continue
                        
                    df_raw = pd.DataFrame({'timestamp': data['TIME'], 'counts': data['COUNTS']}).dropna()
                    df_raw['dt'] = pd.to_datetime(df_raw['timestamp'], unit='s')
                    df_raw.set_index('dt', inplace=True)
                    df_resampled = df_raw.resample('10s').mean().dropna()
                    
                    # Check naming convention for instrument
                    if 'solexs' in fname or 'slx' in fname:
                        solexs_data = df_resampled.rename(columns={'counts': 'counts'})
                        print(f"    -> Successfully extracted SoLEXS lightcurve: {len(solexs_data)} rows.")
                    elif 'hel1os' in fname or 'hlo' in fname:
                        hel1os_data = df_resampled.rename(columns={'counts': 'hel1os_counts'})
                        print(f"    -> Successfully extracted HEL1OS lightcurve: {len(hel1os_data)} rows.")
            except Exception as e:
                print(f"    -> Error reading {fname}: {e}")
                    
        if solexs_data is None and hel1os_data is None:
            print("[!] No valid SoLEXS or HEL1OS .fits files found in zip.")
        else:
            # Load existing database to merge new data
            if os.path.exists(CSV_FILE):
                df_existing = pd.read_csv(CSV_FILE)
                df_existing['dt'] = pd.to_datetime(df_existing['timestamp'], unit='s')
                df_existing.set_index('dt', inplace=True)
            else:
                df_existing = pd.DataFrame()
                
            # Combine the incoming data
            new_data = None
            if solexs_data is not None and hel1os_data is not None:
                new_data = solexs_data.join(hel1os_data[['hel1os_counts']], how='outer')
            elif solexs_data is not None:
                new_data = solexs_data
                new_data['hel1os_counts'] = np.nan
            elif hel1os_data is not None:
                new_data = hel1os_data
                new_data['counts'] = np.nan
                
            if new_data is not None:
                # Merge incoming data into existing database, prioritizing new data
                df_existing = new_data.combine_first(df_existing)
                
                # Interpolate missing values so the AI model doesn't crash on NaNs
                df_existing['counts'] = df_existing['counts'].interpolate(method='time').fillna(0)
                df_existing['hel1os_counts'] = df_existing['hel1os_counts'].interpolate(method='time').fillna(0)
                
                # Format and save
                df_existing['iso'] = df_existing.index.strftime('%Y-%m-%d %H:%M:%S.%f').str[:-3]
                df_final = df_existing[['timestamp', 'counts', 'hel1os_counts', 'iso']].copy()
                df_final.to_csv(CSV_FILE, index=False)
                print(f"[+] Data successfully merged into database. Total rows: {len(df_final)}")
            
        # Move processed zip so it doesn't get processed again
        processed_dir = os.path.join(ZIP_DIR, "processed")
        os.makedirs(processed_dir, exist_ok=True)
        dest_path = os.path.join(processed_dir, os.path.basename(zip_path))
        if os.path.exists(dest_path):
            os.remove(dest_path) # Prevent WinError 183
        os.rename(zip_path, dest_path)
        print(f"[>] Moved zip to {processed_dir}\n")
        
    except Exception as e:
        print(f"[!] Error processing {zip_path}: {e}")
        # If the file is corrupt or fails to parse, move it to an error folder to stop infinite loops
        error_dir = os.path.join(ZIP_DIR, "error")
        os.makedirs(error_dir, exist_ok=True)
        dest_path = os.path.join(error_dir, os.path.basename(zip_path))
        if os.path.exists(dest_path):
            try: os.remove(dest_path)
            except: pass
        try:
            os.rename(zip_path, dest_path)
            print(f"[>] Moved corrupted zip to {error_dir}\n")
        except:
            pass
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

def watch_directory():
    print(f"Monitoring {ZIP_DIR} for incoming telemetry (.zip)...")
    while True:
        zip_files = glob.glob(os.path.join(ZIP_DIR, "*.zip"))
        for z in zip_files:
            extract_and_process(z)
        time.sleep(3)

if __name__ == "__main__":
    os.makedirs(ZIP_DIR, exist_ok=True)
    watch_directory()
