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
            
        # ISRO file extensions vary (.fits, .lc, .evt, or even no extension). 
        # We will scan ALL files extracted.
        all_files = glob.glob(os.path.join(temp_dir, "**", "*"), recursive=True)
        data_files = [f for f in all_files if os.path.isfile(f)]
        
        solexs_data = None
        hel1os_data = None
        
        for f in data_files:
            fname = os.path.basename(f).lower()
            
            # Skip known auxiliary files immediately
            if fname.endswith('.gti') or fname.endswith('.hk') or fname.endswith('.xml') or fname.endswith('.txt'):
                print(f"    -> Skipping auxiliary/metadata file: {fname}")
                continue
                
            try:
                # Try opening it as a FITS file regardless of extension
                with fits.open(f) as hdul:
                    # Look through extensions to find the binary table with data
                    data = None
                    for ext in hdul:
                        if isinstance(ext, fits.BinTableHDU):
                            cols = ext.columns.names
                            if 'TIME' in cols and ('COUNTS' in cols or 'RATE' in cols):
                                data = ext.data
                                break
                            else:
                                print(f"    -> [Debug] Found table in {fname}, but columns are: {cols}")
                    
                    if data is None:
                        continue
                        
                    counts_col = 'COUNTS' if 'COUNTS' in data.columns.names else 'RATE'
                    df_raw = pd.DataFrame({'timestamp': data['TIME'], 'counts': data[counts_col]}).dropna()
                    df_raw['dt'] = pd.to_datetime(df_raw['timestamp'], unit='s')
                    df_raw.set_index('dt', inplace=True)
                    df_resampled = df_raw.resample('10s').mean().dropna()
                    
                    if 'solexs' in fname or 'slx' in fname:
                        solexs_data = df_resampled.rename(columns={'counts': 'counts'})
                        print(f"    -> Successfully extracted SoLEXS data from {fname}: {len(solexs_data)} rows.")
                    elif 'hel1os' in fname or 'hlo' in fname:
                        hel1os_data = df_resampled.rename(columns={'counts': 'hel1os_counts'})
                        print(f"    -> Successfully extracted HEL1OS data from {fname}: {len(hel1os_data)} rows.")
            except Exception as e:
                if fname.endswith('.lc') or fname.endswith('.evt'):
                    print(f"    -> [!] Failed to read {fname}: {e}")
                    
        if solexs_data is None and hel1os_data is None:
            print("[!] No telemetry data found in this ZIP. (Only found auxiliary files like .gti)")
        else:
            # Merge logic
            if os.path.exists(CSV_FILE):
                df_existing = pd.read_csv(CSV_FILE)
                df_existing['dt'] = pd.to_datetime(df_existing['timestamp'], unit='s')
                df_existing.set_index('dt', inplace=True)
            else:
                df_existing = pd.DataFrame()
                
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
                df_existing = new_data.combine_first(df_existing)
                df_existing['counts'] = df_existing['counts'].interpolate(method='time').fillna(0)
                df_existing['hel1os_counts'] = df_existing['hel1os_counts'].interpolate(method='time').fillna(0)
                df_existing['iso'] = df_existing.index.strftime('%Y-%m-%d %H:%M:%S.%f').str[:-3]
                df_final = df_existing[['timestamp', 'counts', 'hel1os_counts', 'iso']].copy()
                df_final.to_csv(CSV_FILE, index=False)
                print(f"[+] Data successfully merged. Total rows in database: {len(df_final)}")
                
    except Exception as e:
        print(f"[!] Error processing {zip_path}: {e}")
        
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
        
        # Move processed/failed zip with Retry for WinError 32
        target_dir = os.path.join(ZIP_DIR, "processed") if solexs_data is not None or hel1os_data is not None else os.path.join(ZIP_DIR, "error")
        os.makedirs(target_dir, exist_ok=True)
        dest_path = os.path.join(target_dir, os.path.basename(zip_path))
        
        for attempt in range(5):
            try:
                if os.path.exists(dest_path):
                    os.remove(dest_path)
                os.rename(zip_path, dest_path)
                print(f"[>] Safely moved zip to {target_dir}\n")
                return True
            except Exception as move_err:
                time.sleep(1)
        else:
            print(f"[!] Critical: Could not move {zip_path} (File locked by Windows). Please delete it manually.")
            return False

def watch_directory():
    print(f"Monitoring {ZIP_DIR} for incoming telemetry (.zip)...")
    skip_list = set()
    while True:
        zip_files = glob.glob(os.path.join(ZIP_DIR, "*.zip"))
        for z in zip_files:
            if z in skip_list:
                continue
            
            success = extract_and_process(z)
            if not success:
                skip_list.add(z)
                
        time.sleep(3)

if __name__ == "__main__":
    os.makedirs(ZIP_DIR, exist_ok=True)
    watch_directory()
