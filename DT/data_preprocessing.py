import os
import scipy.io
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


# Base path setup
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_FILE = os.path.join(BASE_DIR, 'datasets', 'Oxford_Battery_Degradation_Dataset_1.mat')


def load_battery_data(data_file=DATA_FILE):
    """Load Oxford Battery dataset (.mat) file"""
    mat = scipy.io.loadmat(data_file)
    input_data = [[[], []] for _ in range(8)]
    CELL_SIZE = [83, 78, 82, 52, 49, 51, 82, 82]

    for i in range(0, 8):
        cell_num = f"Cell{i + 1}"
        for j in range(0, CELL_SIZE[i]):
            cyc_num = f"cyc{j * 100:04d}"
            try:
                curr = mat[cell_num][0][cyc_num][0][0]["C1ch"][0][0]['q'][0][-1][0]
            except Exception:
                curr = np.nan
            input_data[i][0].append(j)
            input_data[i][1].append(curr)
    return input_data


def prepare_xgb_data(input_data, lookback=10):
    """
    Prepare flattened data for XGBoost.
    Train: Cells 1–4
    Test: Cells 5–8
    """
    train_x, train_y = [], []
    test_x, test_y = [[] for _ in range(4)], [[] for _ in range(4)]
    label_scalers = []

    for i in range(8):
        df_input = pd.DataFrame(input_data[i]).transpose()
        df_input = df_input.rename(columns={0: "cycle", 1: "q_value"})
        df_input['interpolate_time'] = df_input['q_value'].interpolate(option='time')
        df_input['soh'] = df_input['interpolate_time'] / 740
        df_input = df_input.drop(["cycle", "q_value", "interpolate_time"], axis=1)

        # Normalize
        scaler = MinMaxScaler()
        data = scaler.fit_transform(df_input[['soh']].values)
        label_scalers.append(scaler)

        # Create rolling windows
        X_cell, y_cell = [], []
        for j in range(lookback, len(data)):
            X_cell.append(data[j - lookback:j, 0])  # flatten window of SOH
            y_cell.append(data[j, 0])

        X_cell = np.array(X_cell)
        y_cell = np.array(y_cell)

        # Split train/test
        if i < 4:
            train_x.append(X_cell)
            train_y.append(y_cell)
        else:
            test_x[i - 4] = X_cell
            test_y[i - 4] = y_cell

    # Flatten all training cells together for XGBoost
    flat_train_x = np.concatenate(train_x, axis=0)
    flat_train_y = np.concatenate(train_y, axis=0)

    return flat_train_x, flat_train_y, test_x, test_y, label_scalers