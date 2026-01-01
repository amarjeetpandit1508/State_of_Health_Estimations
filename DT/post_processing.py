import torch
import os
import matplotlib.pyplot as plt


def save_model(model, path):
    """Save trained XGBoost model to a file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    model.save_model(path)
    print(f"✅ XGBoost model saved to: {path}")

def plot_xgb_predictions(preds, targets, title="Prediction vs Actual", save_path=None):
    plt.figure(figsize=(10, 6))
    plt.plot(targets, label="Actual SOH", color="b")
    plt.plot(preds, label="Predicted SOH", marker="o", color="g")
    plt.xlabel("Cycle")
    plt.ylabel("SOH")
    plt.title(title)
    plt.legend()

    if save_path:
        plt.savefig(save_path, format="svg")  # Save as SVG
        print(f"Saved figure: {save_path}")

    plt.show()
    plt.close()

