from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    root_mean_squared_error,
    r2_score
)


def evaluate_model(y_true, y_pred):

    mae = mean_absolute_error(y_true, y_pred)

    mse = mean_squared_error(y_true, y_pred)

    rmse = root_mean_squared_error(y_true, y_pred)

    r2 = r2_score(y_true, y_pred)

    return {
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2": r2
    }


def check_overfitting(
    model,
    X_train,
    X_test,
    y_train,
    y_test
):

    train_pred = model.predict(X_train)

    test_pred = model.predict(X_test)

    train_metrics = evaluate_model(
        y_train,
        train_pred
    )

    test_metrics = evaluate_model(
        y_test,
        test_pred
    )

    r2_gap = (
        train_metrics["R2"]
        - test_metrics["R2"]
    )

    return {
        "train": train_metrics,
        "test": test_metrics,
        "r2_gap": r2_gap
    }