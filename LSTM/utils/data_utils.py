import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def load_data(path, date_col='datetime'):
    df = pd.read_csv(path)
    df[date_col] = pd.to_datetime(df[date_col])
    df.set_index(date_col, inplace=True)
    return df.dropna()

def prepare_sequences(df, features, target_col, seq_len=24, forecast_horizon=1):
    scaler = MinMaxScaler()
    df_scaled = scaler.fit_transform(df[features])
    X, y = [], []
    target_idx = features.index(target_col)

    for i in range(len(df_scaled) - seq_len - forecast_horizon + 1):
        X.append(df_scaled[i:i + seq_len])
        y.append(df_scaled[i + seq_len:i + seq_len + forecast_horizon, target_idx])

    return np.array(X), np.array(y), scaler
