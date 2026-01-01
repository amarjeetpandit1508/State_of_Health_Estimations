from xgboost import XGBRegressor  # MUST be XGBRegressor, not XGBModel


def create_xgb_model(params=None):

    default_params = {
        "n_estimators": 500,
        "learning_rate": 0.05,
        "max_depth": 5,
        "subsample": 0.8,
        "colsample_bytree": 1.0,
        "reg_alpha": 0.1,   # L1 regularization
        "reg_lambda": 1.0,  # L2 regularization
        "random_state": 42,
        "objective": "reg:squarederror",
        "n_jobs": -1
    }

    if params:
        default_params.update(params)

    return XGBRegressor(**default_params)