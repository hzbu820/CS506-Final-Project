from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import numpy as np

def compute_metrics(true, pred):
    mse = mean_squared_error(true, pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(true, pred)
    r2 = r2_score(true, pred)
    return {"MSE": mse, "RMSE": rmse, "MAE": mae, "R2": r2}

def plot_predictions(dates, actual, predicted, filename):
    plt.figure(figsize=(12, 6))
    plt.plot(dates, actual, label="Actual", color="blue")
    plt.plot(dates, predicted, label="Predicted", color="red", linestyle="--")
    plt.title("Prediction vs Actual")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
