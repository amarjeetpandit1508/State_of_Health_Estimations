import torch
import itertools
import numpy as np
from train import train
from post_processing import evaluate
from data_preprocessing import load_battery_data, preprocess_data


def tune_hyperparameters(model_type):
    """
    Performs grid search over multiple hyperparameters to find best model config.
    """

    # ---------------- Hyperparameter ranges ----------------
    learning_rates = [0.001, 0.005, 0.01]
    hidden_dims = [128, 256]
    lookbacks = [5, 10]

    results = []

    # Load data once (preprocessed differently for each lookback)
    print("📦 Loading dataset...")
    input_data = load_battery_data()

    # Loop over all combinations
    for lr, hidden_dim, lookback in itertools.product(learning_rates, hidden_dims, lookbacks):
        print(f"\n🚀 Training {model_type} with lr={lr}, hidden_dim={hidden_dim}, lookback={lookback}")
        print(f"\n🚀 Training {model_type} with lr={lr}, hidden_dim={hidden_dim}, lookback={lookback}")

        # Preprocess with given lookback
        train_loader, test_x, test_y, label_scalers = preprocess_data(input_data, lookback=lookback)

        # Train model
        model = train(train_loader, lr, hidden_dim=hidden_dim, model_type=model_type)

        # Evaluate on one validation cell (e.g., Cell 5)
        outputs, targets, MAE, RMSE = evaluate(model, [test_x[0]], [test_y[0]], [label_scalers[4]])

        results.append({
            "model_type": model_type,
            "lr": lr,
            "hidden_dim": hidden_dim,
            "lookback": lookback,
            "MAE": MAE,
            "RMSE": RMSE
        })

    # Sort by MAE ascending
    results = sorted(results, key=lambda x: x["MAE"])
    best = results[0]

    print("\n🏆 Best Configuration:")
    for k, v in best.items():
        print(f"{k}: {v}")

    return best, results



# def tune_hyperparameters(model_type):
#     """
#     Performs grid search over multiple hyperparameters to find best model config.
#     """
#
#     # ---------------- Hyperparameter ranges ----------------
#     learning_rates = [0.001, 0.01]
#     hidden_dims = [128, 256]
#     lookbacks = [5, 10]
#     epochs = [100]  # you can expand if GPU available
#
#     results = []
#
#     # Load data once (preprocessed differently for each lookback)
#     print("📦 Loading dataset...")
#     input_data = load_battery_data()
#
#     # Loop over all combinations
#     for lr, hidden_dim, lookback, EPOCHS in itertools.product(learning_rates, hidden_dims, lookbacks, epochs):
#         print(f"\n🚀 Training {model_type} with lr={lr}, hidden_dim={hidden_dim}, lookback={lookback}, epochs={EPOCHS}")
#         print(f"\n🚀 Training {model_type} with lr={lr}, hidden_dim={hidden_dim}, lookback={lookback}, epochs={EPOCHS}")
#
#         # Preprocess with given lookback
#         train_loader, test_x, test_y, label_scalers = preprocess_data(input_data, lookback=lookback)
#
#         # Train model
#         model = train(train_loader, lr, hidden_dim=hidden_dim, EPOCHS=EPOCHS, model_type=model_type)
#
#         # Evaluate on one validation cell (e.g., Cell 5)
#         outputs, targets, MAE, RMSE = evaluate(model, [test_x[0]], [test_y[0]], [label_scalers[4]])
#
#         results.append({
#             "model_type": model_type,
#             "lr": lr,
#             "hidden_dim": hidden_dim,
#             "lookback": lookback,
#             "epochs": EPOCHS,
#             "MAE": MAE,
#             "RMSE": RMSE
#         })
#
#     # Sort by MAE ascending
#     results = sorted(results, key=lambda x: x["MAE"])
#     best = results[0]
#
#     print("\n🏆 Best Configuration:")
#     for k, v in best.items():
#         print(f"{k}: {v}")
#
#     return best, results
