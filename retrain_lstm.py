import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import pickle
from scipy.ndimage import uniform_filter1d

BUFFER_FILE = "data/retrain_buffer.json"
DATA_FILE = "data/combined_telemetry.csv"
MODEL_PATH = "models/forecast_lstm.pt"
SCALER_PATH = "models/scaler.pkl"

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

def run_retraining():
    print("[*] Checking Continuous Learning Retraining Buffer...")
    if not os.path.exists(BUFFER_FILE) or os.path.getsize(BUFFER_FILE) == 0:
        print("[+] Buffer empty. No retraining needed.")
        return
        
    with open(BUFFER_FILE, 'r') as f:
        buffer = json.load(f)
        
    if len(buffer) < 5:
        print(f"[~] Only {len(buffer)} new verified flares. Waiting for at least 5 before batch retraining.")
        return
        
    print(f"[*] Found {len(buffer)} new verified events. Initiating background retraining...")
    
    # Load Model and Scaler
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)
        
    # NOTE: In the future we will use 8 features (both instruments). 
    # For now, to hot-swap seamlessly with the existing 4-feature model in production without breaking, 
    # we continue fine-tuning the 4-feature model on the verified edge cases.
    model = FlareLSTM(input_size=4, hidden_size=32, num_layers=2)
    model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
    model.train()
    
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCELoss()
    
    # Load Telemetry to extract the sequences
    df = pd.read_csv(DATA_FILE)
    counts = df["counts"].values
    counts_no_nan = np.nan_to_num(counts, nan=np.nanmedian(counts))
    smoothed = uniform_filter1d(counts_no_nan, size=6)
    deriv = np.gradient(smoothed)
    roll_deriv = uniform_filter1d(deriv, size=12)
    df_counts = pd.Series(counts)
    roll_std = df_counts.rolling(window=180, min_periods=1).std().fillna(0).values
    
    counts_scaled = scaler.transform(counts_no_nan.reshape(-1, 1)).flatten()
    smoothed_scaled = scaler.transform(smoothed.reshape(-1, 1)).flatten()
    
    features = np.column_stack([
        counts_scaled,
        smoothed_scaled,
        roll_deriv / 10.0,
        roll_std / 100.0
    ])
    
    # Build dataset from buffer
    X_train = []
    y_train = []
    
    lookahead = 90
    window_size = 60
    
    for event in buffer:
        peak_idx = event['peak_idx']
        start_idx = event['start_idx']
        
        # Positive sample: 15 mins (90 steps) before the flare starts
        pos_target_idx = start_idx - lookahead
        if pos_target_idx > window_size:
            X_train.append(features[pos_target_idx - window_size : pos_target_idx])
            y_train.append(1.0)
            
        # Add a negative sample from a quiet period (e.g. 300 steps before)
        neg_target_idx = start_idx - 300
        if neg_target_idx > window_size:
            X_train.append(features[neg_target_idx - window_size : neg_target_idx])
            y_train.append(0.0)
            
    if len(X_train) == 0:
        print("[!] No valid sequences extracted.")
        return
        
    X_train = torch.tensor(np.array(X_train), dtype=torch.float32)
    y_train = torch.tensor(np.array(y_train), dtype=torch.float32)
    
    print(f"[*] Training on {len(X_train)} extracted sequences...")
    
    epochs = 10
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        print(f"    Epoch {epoch+1}/{epochs} | Loss: {loss.item():.4f}")
        
    # Save the updated model
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"[+] Model fine-tuning complete. Hot-swapped new weights to {MODEL_PATH}")
    
    # Clear the buffer so we don't overfit on these same events again
    with open(BUFFER_FILE, 'w') as f:
        json.dump([], f)
    print("[+] Buffer cleared for next batch.")

if __name__ == "__main__":
    run_retraining()
