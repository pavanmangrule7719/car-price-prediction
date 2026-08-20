import os
import pandas as pd

from src.data_loader import load_dataset
from src.data_cleaning import (
    remove_duplicates,
    save_cleaned_dataset
)

from src.preprocessing import (
    split_features_target,
    create_preprocessor
)

from src.model_training import (
    train_linear_regression,
    train_decision_tree,
    train_random_forest,
    train_random_forest_grid_search,
    train_random_forest_random_search
)

from src.evaluation import evaluate_model
from src.model_selection import select_best_model

from src.model_saving import (
    save_model,
    load_model
)

from src.prediction import predict_price

from sklearn.model_selection import train_test_split

MODEL_PATH = "models/final_model.pkl"

def get_user_input():

    print("\n========== CAR DETAILS ==========")

    car = {
        "model": input("Model: "),
        "year": int(input("Year: ")),
        "mileage": float(input("Mileage: ")),
        "transmission": input("Transmission: "),
        "fuelType": input("Fuel Type: "),
        "tax": float(input("Tax: ")),
        "mpg": float(input("MPG: ")),
        "engineSize": float(input("Engine Size: "))
    }

    return pd.DataFrame([car])


def train_model():

    print("\n🚀 Training model...")

    dataset_path = "data/audi.csv"

    df = load_dataset(dataset_path)

    print("✅ Dataset loaded successfully.")

    print("\n========== DATASET INFORMATION ==========")

    print("Shape:")
    print(df.shape)

    print("\nHead:")
    print(df.head())

    print("\nDataset Information:")
    df.info()

    print("\nMissing Values:")
    print(df.isnull().sum())

    print("\nData Types:")
    print(df.dtypes)

    print("\nDuplicate Rows:")
    print(df.duplicated().sum())

    print("\nStatistical Summary:")
    print(df.describe())

    df = remove_duplicates(df)

    save_cleaned_dataset(
        df,
        "data/cleaned_audi.csv"
    )

    X, y = split_features_target(df)

    preprocessor = create_preprocessor()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    results = {}

    # ---------- Linear Regression ----------

    linear_result = train_linear_regression(
        preprocessor,
        X_train,
        X_test,
        y_train,
        y_test
    )

    linear_metrics = evaluate_model(
        y_test,
        linear_result["predictions"]
    )

    results["Linear Regression"] = {
        "model": linear_result["model"],
        "metrics": linear_metrics
    }

    print("\n--- Linear Regression ---")
    print(linear_metrics)

    # ---------- Decision Tree ----------

    tree_results = train_decision_tree(
        preprocessor,
        X_train,
        X_test,
        y_train,
        y_test
    )

    for result in tree_results:

        metrics = evaluate_model(
            y_test,
            result["predictions"]
        )

        name = f"Decision Tree Depth {result['max_depth']}"

        results[name] = {
            "model": result["model"],
            "metrics": metrics
        }

        print(f"\n--- {name} ---")
        print(metrics)

    # ---------- Random Forest ----------

    rf_results = train_random_forest(
        preprocessor,
        X_train,
        X_test,
        y_train,
        y_test
    )

    for result in rf_results:

        metrics = evaluate_model(
            y_test,
            result["predictions"]
        )

        name = f"Random Forest {result['n_estimators']} Trees"

        results[name] = {
            "model": result["model"],
            "metrics": metrics
        }

        print(f"\n--- {name} ---")
        print(metrics)

    grid_result = train_random_forest_grid_search(
        preprocessor,
        X_train,
        X_test,
        y_train,
        y_test
    )

    grid_metrics = evaluate_model(
        y_test,
        grid_result["predictions"]
    )

    results["Random Forest GridSearchCV"] = {
        "model": grid_result["model"],
        "metrics": grid_metrics
    }

    print("\n--- GridSearchCV ---")
    print("Best Parameters:")
    print(grid_result["best_params"])

    print("Best CV R²:")
    print(grid_result["best_cv_r2"])

    print("Test Metrics:")
    print(grid_metrics)

    random_result = train_random_forest_random_search(
        preprocessor,
        X_train,
        X_test,
        y_train,
        y_test
    )

    random_metrics = evaluate_model(
        y_test,
        random_result["predictions"]
    )

    results["Random Forest RandomizedSearchCV"] = {
        "model": random_result["model"],
        "metrics": random_metrics
    }

    print("\n--- RandomizedSearchCV ---")
    print("Best Parameters:")
    print(random_result["best_params"])

    print("Best CV R²:")
    print(random_result["best_cv_r2"])

    print("Test Metrics:")
    print(random_metrics)

    print("\n================================")
    print("MODEL COMPARISON")
    print("================================")

    for name, result in results.items():

        print(
            f"{name}: "
            f"R² = {result['metrics']['R2']:.4f}"
        )

    best_model_result = select_best_model(results)

    print("\n================================")
    print("BEST MODEL")
    print("================================")

    print(
        "Model:",
        best_model_result["name"]
    )

    print(
        "R²:",
        best_model_result["r2"]
    )

    save_model(
        best_model_result["model"],
        MODEL_PATH
    )

    print("\n✅ Model training completed.")
    print(f"✅ Model saved at: {MODEL_PATH}")

    return best_model_result["model"]

def prediction_system(model):

    car = get_user_input()

    prediction = predict_price(
        model,
        car
    )

    print("\n================================")
    print("PREDICTION")
    print("================================")

    print(
        f"Predicted Price: £{prediction[0]:,.2f}"
    )

def main():

    if os.path.exists(MODEL_PATH):

        print("✅ Existing model found.")
        print("⚡ Loading model instead of training...")

        model = load_model(MODEL_PATH)

    else:

        print("⚠️ Model not found.")
        print("🚀 Training model for the first time...")

        model = train_model()

    prediction_system(model)

if __name__ == "__main__":
    main()