from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV
)

def train_linear_regression(
    preprocessor,
    X_train,
    X_test,
    y_train,
    y_test
):

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", LinearRegression())
        ]
    )

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)

    return {
        "model": pipeline,
        "predictions": y_pred
    }


def train_decision_tree(
    preprocessor,
    X_train,
    X_test,
    y_train,
    y_test
):

    depths = [5, 10, 15, 20]

    results = []

    for depth in depths:

        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                (
                    "model",
                    DecisionTreeRegressor(
                        max_depth=depth,
                        random_state=42
                    )
                )
            ]
        )

        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_test)

        y_train_pred = pipeline.predict(X_train)

        results.append({
            "model": pipeline,
            "max_depth": depth,
            "predictions": y_pred,
            "train_predictions": y_train_pred
        })

    return results


def train_random_forest(
    preprocessor,
    X_train,
    X_test,
    y_train,
    y_test
):

    estimators = [50, 100, 200, 300]

    results = []

    for n in estimators:

        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=n,
                        random_state=42,
                        n_jobs=-1
                    )
                )
            ]
        )

        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_test)

        y_train_pred = pipeline.predict(X_train)

        results.append({
            "model": pipeline,
            "n_estimators": n,
            "predictions": y_pred,
            "train_predictions": y_train_pred
        })

    return results

def train_random_forest_grid_search(
    preprocessor,
    X_train,
    X_test,
    y_train,
    y_test
):

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "model",
                RandomForestRegressor(
                    random_state=42,
                    n_jobs=-1
                )
            )
        ]
    )

    param_grid = {
        "model__n_estimators": [100, 200, 300],
        "model__max_depth": [10, 15, 20],
        "model__min_samples_split": [2, 5],
        "model__min_samples_leaf": [1, 2]
    }

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=5,
        scoring="r2",
        n_jobs=-1
    )

    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_

    y_pred = best_model.predict(X_test)

    return {
        "model": best_model,
        "best_params": grid_search.best_params_,
        "best_cv_r2": grid_search.best_score_,
        "predictions": y_pred
    }

def train_random_forest_random_search(
    preprocessor,
    X_train,
    X_test,
    y_train,
    y_test
):

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "model",
                RandomForestRegressor(
                    random_state=42,
                    n_jobs=-1
                )
            )
        ]
    )

    param_distributions = {
        "model__n_estimators": [
            100,
            200,
            300,
            500
        ],

        "model__max_depth": [
            10,
            15,
            20,
            25,
            None
        ],

        "model__min_samples_split": [
            2,
            5,
            10
        ],

        "model__min_samples_leaf": [
            1,
            2,
            4
        ]
    }

    random_search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_distributions,
        n_iter=10,
        cv=5,
        scoring="r2",
        random_state=42,
        n_jobs=-1
    )

    random_search.fit(X_train, y_train)

    best_model = random_search.best_estimator_

    y_pred = best_model.predict(X_test)

    return {
        "model": best_model,
        "best_params": random_search.best_params_,
        "best_cv_r2": random_search.best_score_,
        "predictions": y_pred
    }
