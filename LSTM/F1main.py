#The main function to calculate F1 SCORE.
import argparse
import yaml
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score
from model.lstm import LSTMModel
from utils.data_utils import load_data, prepare_sequences
from utils.train import LSTMTrainer


def compute_trend_f1(actual_real, pred_real):
    actual_labels = (np.diff(actual_real) > 0).astype(int)
    pred_labels = (np.diff(pred_real) > 0).astype(int)
    min_len = min(len(actual_labels), len(pred_labels))
    return f1_score(actual_labels[:min_len], pred_labels[:min_len])


def main(config_path):
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)

    df = load_data(cfg["data_path"])
    X, y, scaler = prepare_sequences(
        df,
        cfg["features"],
        cfg["target_column"],
        seq_len=cfg["model"]["seq_length"],
        forecast_horizon=cfg["model"]["forecast_horizon"]
    )

    split = int(len(X) * 0.8)
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    train_ds = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
    val_ds = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val))

    train_loader = DataLoader(train_ds, batch_size=cfg["training"]["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg["training"]["batch_size"], shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != 'cuda':
        raise EnvironmentError("CUDA device required for training")

    model = LSTMModel(
        input_size=len(cfg["features"]),
        hidden_size=cfg["model"]["hidden_size"],
        num_layers=cfg["model"]["num_layers"],
        output_size=1,
        dropout=cfg["model"]["dropout"]
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["training"]["learning_rate"])
    trainer = LSTMTrainer(model, optimizer, torch.nn.MSELoss(), device)
    trainer.train(train_loader, val_loader, cfg["training"]["epochs"], cfg["training"]["patience"])

    model.load_state_dict(torch.load("best_model.pth", weights_only=True))
    model.eval()

    preds = []
    with torch.no_grad():
        for x_batch, _ in val_loader:
            pred = model(x_batch.to(device)).cpu().numpy()
            preds.extend(pred)

    dummy_pred = np.zeros((len(preds), len(cfg["features"])))
    target_idx = cfg["features"].index(cfg["target_column"])
    for i, p in enumerate(preds):
        dummy_pred[i][target_idx] = p[0]
    pred_real = scaler.inverse_transform(dummy_pred)[:, target_idx]

    dummy_actual = np.zeros((len(y_val), len(cfg["features"])))
    for i, a in enumerate(y_val):
        dummy_actual[i][target_idx] = a[0]
    actual_real = scaler.inverse_transform(dummy_actual)[:, target_idx]

    f1 = compute_trend_f1(actual_real, pred_real)
    print(f"F1 Score (Up/Down Classification): {f1:.4f}")

    with open("AMZN_trend_f1_results.txt", "a") as f:
        f.write("Trend-Only Evaluation\n")
        f.write(f"F1 Score (Up/Down Classification): {f1:.4f}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml", help="Path to config file")
    args = parser.parse_args()
    main(args.config)

