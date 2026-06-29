import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter1d
import pickle
import sys

# Import flare detection from app.py to generate labels
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from app import detect_flares

# Configuration
WINDOW_SIZE = 60      # 10 minutes (60 * 10s)
LEAD_TIME = 90        # 15 minutes (90 * 10s) - Target: flare in next 15 mins
BATCH_SIZE = 1024
EPOCHS = 10
LEARNING_RATE = 0.001
HIDDEN_SIZE = 32
NUM_LAYERS = 2

class FlareDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class FlareLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(FlareLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        out, _ = self.lstm(x)
        # Take the output of the last time step
        out = out[:, -1, :]
        out = self.fc(out)
        return self.sigmoid(out).squeeze()

def prepare_data(csv_path="data/solexs_all.csv"):
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Process per day to avoid cross-day rolling artifacts
    dates = df["iso"].str[:10].unique()
    
    all_X = []
    all_y = []
    
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    
    # Fit scaler on full counts
    counts_all = np.nan_to_num(df["counts"].values, nan=np.nanmedian(df["counts"].values))
    scaler.fit(counts_all.reshape(-1, 1))
    
    for date in dates:
        day_df = df[df["iso"].str.startswith(date)].copy()
        counts = day_df["counts"].values.astype(float)
        timestamps = day_df["timestamp"].values.astype(float)
        
        # Fill nans
        counts = np.nan_to_num(counts, nan=np.nanmedian(counts))
        
        # Generate labels using the existing logic
        flares, _, _ = detect_flares(counts, timestamps)
        
        labels = np.zeros(len(counts))
        for flare in flares:
            start_idx = flare["start_idx"]
            # Label 1 for LEAD_TIME before the flare starts
            label_start = max(0, start_idx - LEAD_TIME)
            labels[label_start:start_idx] = 1
            
        # Feature Engineering
        # 1. Normalized counts
        counts_scaled = scaler.transform(counts.reshape(-1, 1)).flatten()
        
        # 2. Smoothed counts
        smoothed = uniform_filter1d(counts, size=6)
        smoothed_scaled = scaler.transform(smoothed.reshape(-1, 1)).flatten()
        
        # 3. Flux Slope (Derivative of smoothed)
        deriv = np.gradient(smoothed)
        roll_deriv = uniform_filter1d(deriv, size=12)
        
        # 4. Rolling std
        df_counts = pd.Series(counts)
        roll_std = df_counts.rolling(window=180, min_periods=1).std().fillna(0).values
        
        # Stack features
        features = np.column_stack([
            counts_scaled,
            smoothed_scaled,
            roll_deriv / 10.0, # scale down
            roll_std / 100.0   # scale down
        ])
        
        # Create sequences
        for i in range(len(counts) - WINDOW_SIZE):
            window_features = features[i:i+WINDOW_SIZE]
            target_label = labels[i+WINDOW_SIZE-1]
            all_X.append(window_features)
            all_y.append(target_label)
            
    X = np.array(all_X)
    y = np.array(all_y)
    
    # Save scaler
    os.makedirs("models", exist_ok=True)
    with open("models/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
        
    return X, y

def train():
    X, y = prepare_data()
    print(f"Data prepared. X shape: {X.shape}, y shape: {y.shape}")
    
    # Simple train/val split (last 20% for validation)
    split_idx = int(len(X) * 0.8)
    X_train, y_train = X[:split_idx], y[:split_idx]
    X_val, y_val = X[split_idx:], y[split_idx:]
    
    # Compute class weights (highly imbalanced)
    pos_weight = (len(y_train) - sum(y_train)) / max(sum(y_train), 1)
    print(f"Positive samples: {sum(y_train)} / {len(y_train)} -> Weight: {pos_weight:.2f}")
    
    train_dataset = FlareDataset(X_train, y_train)
    val_dataset = FlareDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = FlareLSTM(input_size=X.shape[2], hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS).to(device)
    
    # BCELoss
    def weighted_binary_cross_entropy(output, target, weight):
        loss = -(weight * target * torch.log(output + 1e-8) + (1 - target) * torch.log(1 - output + 1e-8))
        return loss.mean()
        
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    print("Starting training...")
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = weighted_binary_cross_entropy(outputs, batch_y, pos_weight)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * batch_X.size(0)
            
        train_loss /= len(train_loader.dataset)
        
        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                loss = weighted_binary_cross_entropy(outputs, batch_y, pos_weight)
                val_loss += loss.item() * batch_X.size(0)
                
                preds = (outputs > 0.5).float()
                correct += (preds == batch_y).sum().item()
                
        val_loss /= len(val_loader.dataset)
        val_acc = correct / len(val_loader.dataset)
        
        print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
        
    # Save model
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/forecast_lstm.pt")
    print("Model saved to models/forecast_lstm.pt")

if __name__ == "__main__":
    train()
