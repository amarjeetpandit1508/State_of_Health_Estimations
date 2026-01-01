import os
from data_preprocessing import DATA_FILE, load_battery_data, prepare_xgb_data
from train import train_xgb_model, evaluate_xgb_model
from hyperparameter_tuning import tune_xgb_hyperparameters
from post_processing import plot_xgb_predictions, save_model

def main():
    print("\n📦 Loading and preparing data...")
    input_data = load_battery_data(DATA_FILE)
    train_x, train_y, test_x, test_y, label_scalers = prepare_xgb_data(input_data, lookback=10)

    print("\n🔍 Tuning XGBoost hyperparameters with time-series CV...")
    best_params = tune_xgb_hyperparameters(train_x, train_y)

    print("\n🚀 Training XGBoost model with early stopping...")
    xgb_model = train_xgb_model(train_x, train_y, best_params)
    save_model(xgb_model, "saved_models/xgb_model.json")

    os.makedirs("plots", exist_ok=True)
    print("\n📊 Evaluating XGBoost model for each test cell...\n")
    for i in range(4):  # Cells 5–8
        cell_num = i + 5
        preds, targets, mae, rmse = evaluate_xgb_model(
            xgb_model,
            test_x[i],
            test_y[i],
            label_scalers[cell_num - 1]
        )

        print(f"Cell {cell_num} → MAE={mae:.4f}, RMSE={rmse:.4f}")

        # Save SVG in the current directory as "Cell_5.svg", "Cell_6.svg", etc.
        save_path = f"plots/XGBoost_Cell_{cell_num}.svg"
        plot_xgb_predictions(preds, targets, title=f"XGBoost - Cell {cell_num}", save_path=save_path)

    print("\n✅ All test cells evaluated successfully!")

if __name__ == "__main__":
    main()