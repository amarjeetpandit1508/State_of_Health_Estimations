import torch
import numpy as np
import math
import time
import os
import matplotlib.pyplot as plt
from data_preprocessing import preprocess_data


def evaluate(model, test_x, test_y, label_scalers):
    model.eval()
    outputs = []
    targets = []
    start_time = time.process_time()

    # Loop through each test set (tx, ty) pair
    for i in range(len(test_x)):

        # Convert numpy array → torch tensor
        input = torch.tensor(test_x[i], dtype=torch.float32)
        label = torch.tensor(test_y[i], dtype=torch.float32)

        # Initialize hidden state
        batch_size = input.shape[0]
        hidden = model.init_hidden(batch_size)
        # Forward pass
        out, h = model(input, hidden)

        # Convert predictions & labels back to original scale
        scaler = label_scalers[i]
        pred = scaler.inverse_transform(out.detach().cpu().numpy()).reshape(-1)
        true = scaler.inverse_transform(label.numpy()).reshape(-1)

        # Store results
        outputs.append(pred)
        targets.append(true)

    print(f"Evaluation Time: {time.process_time() - start_time:.2f}s")

    # ---- Compute MAE / RMSE ----
    MAE = 0
    MSE = 0

    for pred, true in zip(outputs, targets):
        MAE += np.mean(np.abs(pred - true)) / len(outputs)
        MSE += np.mean((pred - true) ** 2) / len(outputs)

    RMSE = math.sqrt(MSE)

    print(f"MAE: {MAE:.4f}")
    print(f"RMSE: {RMSE:.4f}")

    return outputs, targets, MAE, RMSE


def save_model(model, path):
    """Save trained PyTorch model state dict to a file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f"✅ Model saved to: {path}")


def plot_predictions(outputs, targets, title="Model Results", n_points=100, save_path=None):
    """Visualize predicted vs actual SOH for the last `n_points`."""
    plt.figure(figsize=(10, 6))
    plt.plot(outputs[0][-n_points:], "-o", color="g", label="Predicted")
    plt.plot(targets[0][-n_points:], color="b", label="Actual")
    plt.xlabel("Cycle")
    plt.ylabel("SOH")
    plt.title(title)
    plt.legend()
    plt.grid(True)

    if save_path:
        if not save_path.endswith(".png"):
            save_path = save_path + ".png"
        plt.savefig(save_path, format="png")
        print(f"Saved figure: {save_path}")

    plt.show()
    plt.close()