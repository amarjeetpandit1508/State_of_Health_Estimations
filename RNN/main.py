import os
import torch
from data_preprocessing import load_battery_data, preprocess_data
from train import train
from post_processing import evaluate, save_model, plot_predictions
from hyperparameter_tuning import tune_hyperparameters

def main():
    # -------------------- 1. Hyperparameter Tuning --------------------
    print("\n🔍 Starting Hyperparameter Tuning for GRU...")
    best_gru, _ = tune_hyperparameters("GRU")

    print("\n🔍 Starting Hyperparameter Tuning for LSTM...")
    best_lstm, _ = tune_hyperparameters("LSTM")

    print("\n🏆 Best Hyperparameters Found:")
    print(f" GRU  → lr={best_gru['lr']}, hidden_dim={best_gru['hidden_dim']}, lookback={best_gru['lookback']}")
    print(f" LSTM → lr={best_lstm['lr']}, hidden_dim={best_lstm['hidden_dim']}, lookback={best_lstm['lookback']}")


    # -------------------- 2. Preprocess Using Best Lookback --------------------
    print("\n📦 Loading and preprocessing data using best lookback values...")
    input_data = load_battery_data()
    train_loader_gru, test_x_gru, test_y_gru, label_scalers_gru = preprocess_data(input_data, lookback=best_gru["lookback"])
    train_loader_lstm, test_x_lstm, test_y_lstm, label_scalers_lstm = preprocess_data(input_data, lookback=best_lstm["lookback"])


    # -------------------- 3. Train Best Models --------------------
    print("\n🚀 Training Final GRU Model with Best Hyperparameters...")
    gru_model = train(train_loader_gru,
                      learn_rate=best_gru["lr"],
                      hidden_dim=best_gru["hidden_dim"],
                      model_type="GRU")
    save_model(gru_model, "saved_models/gru_model.pt")

    print("\n🚀 Training Final LSTM Model with Best Hyperparameters...")
    lstm_model = train(train_loader_lstm,
                       learn_rate=best_lstm["lr"],
                       hidden_dim=best_lstm["hidden_dim"],
                       model_type="LSTM")
    save_model(lstm_model, "saved_models/lstm_model.pt")


    # -------------------- 4. Evaluate Both Models --------------------
    os.makedirs("plots", exist_ok=True)

    print("\n📊 Evaluating GRU Model...")
    for i in range(4):
        print(f"\nGRU Evaluation for Cell {i + 5}")
        gru_outputs, targets, gru_MAE, gru_RMSE = evaluate(
            gru_model,
            [test_x_gru[i]],
            [test_y_gru[i]],
            [label_scalers_gru[i + 4]],
        )

        save_path = f"plots/GRU_Cell_{i + 5}"
        plot_predictions(gru_outputs, targets, title=f"Cell {i + 5} (GRU)", save_path=save_path)

    print("\n📊 Evaluating LSTM Model...")
    for i in range(4):
        print(f"\nLSTM Evaluation for Cell {i + 5}")
        lstm_outputs, targets, lstm_MAE, lstm_RMSE = evaluate(
            lstm_model,
            [test_x_lstm[i]],
            [test_y_lstm[i]],
            [label_scalers_lstm[i + 4]],
        )

        save_path = f"plots/LSTM_Cell_{i + 5}"
        plot_predictions(lstm_outputs, targets, title=f"Cell {i + 5} (LSTM)", save_path=save_path)

    print("\n✅ All models trained, tuned, and evaluated successfully!")


if __name__ == "__main__":
    main()
