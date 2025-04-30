import argparse
import yaml
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from model.lstm import LSTMModel
from utils.data_utils import load_data, prepare_sequences
from utils.train import LSTMTrainer
from utils.evaluate import compute_metrics, plot_predictions

def main(config_path):
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)

    df = load_data(cfg["data_path"])
    X, y, scaler = prepare_sequences(df, cfg["features"], cfg["target_column"],
                                     cfg["model"]["seq_length"],
                                     cfg["model"]["forecast_horizon"])

    split = int(len(X) * 0.8)
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    train_ds = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
    val_ds = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val))
    train_loader = DataLoader(train_ds, batch_size=cfg["training"]["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg["training"]["batch_size"], shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != 'cuda':
        raise EnvironmentError("CUDA-capable GPU not available.")
    model = LSTMModel(len(cfg["features"]),
                      cfg["model"]["hidden_size"],
                      cfg["model"]["num_layers"],
                      1,
                      cfg["model"]["dropout"]).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["training"]["learning_rate"])
    trainer = LSTMTrainer(model, optimizer, torch.nn.MSELoss(), device)
    trainer.train(train_loader, val_loader, cfg["training"]["epochs"], cfg["training"]["patience"])

    # model.load_state_dict(torch.load("best_model.pth"))
    model.eval()

    preds = []
    with torch.no_grad():
        for x_batch, _ in val_loader:
            pred = model(x_batch.to(device)).cpu().numpy()
            preds.extend(pred)

    actual = y_val.flatten()
    pred = [p[0] for p in preds]

    # Inverse transform prediction
    dummy_pred = np.zeros((len(preds), len(cfg["features"])))
    target_idx = cfg["features"].index(cfg["target_column"])
    for i, p in enumerate(preds):
        dummy_pred[i][target_idx] = p[0]
    pred_real = scaler.inverse_transform(dummy_pred)[:, target_idx]

    # Inverse transform actual values
    dummy_actual = np.zeros((len(y_val), len(cfg["features"])))
    for i, a in enumerate(y_val):
        dummy_actual[i][target_idx] = a[0]
    actual_real = scaler.inverse_transform(dummy_actual)[:, target_idx]

    print(compute_metrics(actual, pred))
    plot_predictions(df.index[-len(pred):], actual_real, pred_real, "Result/MSFT_final_prediction.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml", help="Path to YAML config")
    args = parser.parse_args()
    main(args.config)
