from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from xgboost import XGBRegressor

def tune_xgb_hyperparameters(train_x, train_y, n_splits=3):
    """
    Tune XGBoost hyperparameters using TimeSeriesSplit to avoid future leakage.
    Returns best parameters.
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)

    param_grid = {
        "n_estimators": [500, 1000],
        "learning_rate": [0.001, 0.01, 0.05],
        "reg_alpha": [0, 0.3],
        "reg_lambda": [0.5, 1]
    }

    model = XGBRegressor(
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1
    )

    grid = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=tscv,
        scoring="neg_mean_absolute_error",
        verbose=1
    )

    grid.fit(train_x, train_y.ravel())

    print("\n🏆 Best Parameters Found:")
    print(grid.best_params_)

    return grid.best_params_