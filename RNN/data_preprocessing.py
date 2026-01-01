import os
import scipy.io
import pandas as pd
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
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
            except ValueError:
                curr = float("NaN")
            input_data[i][0].append(j)
            input_data[i][1].append(curr)
    return input_data


def preprocess_data(input_data, lookback, batch_size=32):
    train_x, train_y = [], []
    test_x, test_y = [[] for _ in range(4)], [[] for _ in range(4)]
    label_scalers = []

    for i in range(0, 8):
        df_input = pd.DataFrame(input_data[i]).transpose()
        df_input = df_input.rename(columns={0: "cycle", 1: "q_value"})
        df_input['interpolate_time'] = df_input['q_value'].interpolate(option='time')
        df_input['soh'] = df_input['interpolate_time'] / 740

        df_input = df_input.drop(["cycle", "q_value", "interpolate_time"], axis=1)

        # Normalization
        sc = MinMaxScaler()
        label_sc = MinMaxScaler()
        data = sc.fit_transform(df_input.values)
        label_sc.fit(df_input.iloc[:, 0].values.reshape(-1, 1))
        label_scalers.append(label_sc)

        inputs = np.zeros((len(data) - lookback, lookback, df_input.shape[1]))
        labels = np.zeros(len(data) - lookback)

        for j in range(lookback, len(data)):
            inputs[j - lookback] = data[j - lookback:j]
            labels[j - lookback] = data[j, 0]

        inputs = inputs.reshape(-1, lookback, df_input.shape[1])
        labels = labels.reshape(-1, 1)

        # --- Split Train/Test ---
        if i < 4:
            if len(train_x) == 0:
                train_x, train_y = inputs[:], labels[:]
            else:
                train_x = np.concatenate((train_x, inputs[:]))
                train_y = np.concatenate((train_y, labels[:]))
        else:
            test_x[i - 4] = inputs
            test_y[i - 4] = labels

    train_data = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y))
    train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size, drop_last=True)

    return train_loader, test_x, test_y, label_scalers



