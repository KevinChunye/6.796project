import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Import your synthetic data generation
from synthetic_data import simulate_heavy_t_ar1, simulate_season_trend_outliers

# Import the Informer model (ensure models/ folder is present)
from models.model import Informer

# -------------------------------------------------------------------
# 1. Custom Dataset for Informer
# -------------------------------------------------------------------
class InformerDataset(Dataset):
    def __init__(self, data, seq_len, label_len, pred_len, freq='h'):
        """
        data: np.ndarray (T, 1) - The raw synthetic time series
        seq_len: Input sequence length for Encoder
        label_len: Start token length for Decoder (overlap with Encoder)
        pred_len: Prediction length
        """
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len
        
        # 1. Standardization
        self.scaler = StandardScaler()
        self.data_x = self.scaler.fit_transform(data)
        self.data_y = self.data_x # In self-supervised/forecasting, X=Y usually

        # 2. Create Dummy Timestamps (Required by Informer)
        # We generate a date range starting from 2020-01-01
        dates = pd.date_range(start='2020-01-01', periods=len(data), freq=freq)
        
        # Extract features: [Month, Day, Weekday, Hour]
        # Normalized roughly to [-0.5, 0.5] or used as embeddings
        df_stamp = pd.DataFrame({'date': dates})
        df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
        df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
        df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.dayofweek, 1)
        df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
        
        self.data_stamp = df_stamp.drop(columns=['date']).values

    def __getitem__(self, index):
        # Calculate indices
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        # Encoder Input
        seq_x = self.data_x[s_begin:s_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]

        # Decoder Input
        # (Label_len of known data + Pred_len of zeros/placeholders)
        dec_x = self.data_x[r_begin:r_end]
        # Mask the future part with zeros (standard Informer practice)
        dec_x[self.label_len:, :] = 0 
        
        dec_x_mark = self.data_stamp[r_begin:r_end]

        # Target
        seq_y = self.data_y[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, dec_x, dec_x_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

# -------------------------------------------------------------------
# 2. The Training Loop (Adapted for Informer Inputs)
# -------------------------------------------------------------------
def train_informer_on_series(
    series,
    seq_len=96,
    label_len=48,
    pred_len=24,
    batch_size=32,
    n_epochs=5,
    lr=1e-4,
    device=None
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Ensure Series shape (T, 1)
    if len(series.shape) == 1:
        series = series.reshape(-1, 1)

    # Prepare Data
    dataset = InformerDataset(series, seq_len, label_len, pred_len)
    
    # Split Train/Val (Simple split for demo)
    train_size = int(len(dataset) * 0.8)
    val_size = len(dataset) - train_size
    train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, drop_last=True)

    # Initialize Informer
    # enc_in=1, dec_in=1, c_out=1 because our synthetic data is univariate
    model = Informer(
        enc_in=1, dec_in=1, c_out=1, 
        seq_len=seq_len, label_len=label_len, out_len=pred_len,
        factor=5, d_model=512, n_heads=8, e_layers=2, d_layers=1, d_ff=2048, 
        dropout=0.05, attn='prob', embed='fixed', freq='h', activation='gelu', 
        output_attention=False, distil=True, mix=True,
        device=device
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    print(f"Start Training on {device}...")

    for epoch in range(n_epochs):
        model.train()
        train_loss = []
        
        for i, (batch_x, batch_y, batch_x_mark, batch_dec_x, batch_dec_x_mark) in enumerate(train_loader):
            optimizer.zero_grad()
            
            # Move to device & cast
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            batch_x_mark = batch_x_mark.float().to(device)
            batch_dec_x = batch_dec_x.float().to(device)
            batch_dec_x_mark = batch_dec_x_mark.float().to(device)

            # Informer Forward Pass
            # Enc_out is not used here, we only need the decoder output for loss
            outputs = model(batch_x, batch_x_mark, batch_dec_x, batch_dec_x_mark)

            # Informer outputs [Batch, Pred_Len, Features]
            # We crop the target 'batch_y' to match the prediction length (last pred_len steps)
            f_dim = -1 if False else 0 # 0 for univariate
            outputs = outputs[:, -pred_len:, f_dim:]
            batch_y = batch_y[:, -pred_len:, f_dim:].to(device)

            loss = criterion(outputs, batch_y)
            train_loss.append(loss.item())
            
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        val_loss = []
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_dec_x, batch_dec_x_mark) in enumerate(val_loader):
                batch_x = batch_x.float().to(device)
                batch_y = batch_y.float().to(device)
                batch_x_mark = batch_x_mark.float().to(device)
                batch_dec_x = batch_dec_x.float().to(device)
                batch_dec_x_mark = batch_dec_x_mark.float().to(device)

                outputs = model(batch_x, batch_x_mark, batch_dec_x, batch_dec_x_mark)
                
                outputs = outputs[:, -pred_len:, f_dim:]
                batch_y = batch_y[:, -pred_len:, f_dim:]
                
                loss = criterion(outputs, batch_y)
                val_loss.append(loss.item())

        print(f"Epoch {epoch+1}: Train Loss {np.average(train_loss):.5f} | Val Loss {np.average(val_loss):.5f}")

    return model

# -------------------------------------------------------------------
# 3. Main Execution
# -------------------------------------------------------------------
if __name__ == "__main__":
    # A. Generate Data (Using your synthetic_data.py function)
    print("Generating Synthetic Data...")
    series, _ = simulate_season_trend_outliers(
        T=4000, 
        season_period=48, # e.g. 2-day seasonality if hourly
        n_outliers=10, 
        outlier_magnitude=5.0
    )
    
    # B. Train Informer
    model = train_informer_on_series(
        series, 
        seq_len=96,   # Look back 96 steps
        label_len=48, # Known history provided to decoder
        pred_len=24,  # Predict next 24 steps
        n_epochs=3
    )
    
    print("Training Complete.")