import numpy as np
from model import create_xgb_model
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

def train_xgb_model(train_x, train_y, params=None):
    """
    Train an XGBoost model on the given dataset using XGBRegressor.
    """
    model = create_xgb_model(params)
    model.fit(train_x, train_y.ravel())
    return model

def evaluate_xgb_model(model, test_x, test_y, scaler):
    """
    Evaluate XGBoost model and return predictions, targets, MAE, RMSE.
    """
    preds = model.predict(test_x)
    preds = preds.reshape(-1, 1)
    targets = test_y.reshape(-1, 1)

    preds_rescaled = scaler.inverse_transform(preds)
    targets_rescaled = scaler.inverse_transform(targets)

    mae = mean_absolute_error(targets_rescaled, preds_rescaled)
    rmse = root_mean_squared_error(targets_rescaled, preds_rescaled)

    return preds_rescaled, targets_rescaled, mae, rmse