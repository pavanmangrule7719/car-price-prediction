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
from src.model_saving import save_model

from sklearn.model_selection import train_test_split


def main():

    # ==========================================
    # PHASE 1 - DATASET + EDA
    # ==========================================

    dataset_path = "data/audi.csv"

    df = load_dataset(dataset_path)

    print("✅ Dataset loaded successfully.")

    print("\nShape:")
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


    # ==========================================
    # PHASE 2 - DATA CLEANING
    # ==========================================

    df = remove_duplicates(df)

    save_cleaned_dataset(
        df,
        "data/cleaned_audi.csv"
    )


    # ==========================================
    # PHASE 3 - PREPROCESSING
    # ==========================================

    X, y = split_features_target(df)

    preprocessor = create_preprocessor()


    # ==========================================
    # PHASE 4 - TRAIN TEST SPLIT
    # ==========================================

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    # ==========================================
    # PHASE 5 + 6 - MULTIPLE MODELS
    # ==========================================

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

        name = (
            f"Decision Tree "
            f"Depth {result['max_depth']}"
        )

        results[name] = {
            "model": result["model"],
            "metrics": metrics
        }

        print(
            f"\n--- {name} ---"
        )

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

        name = (
            f"Random Forest "
            f"{result['n_estimators']} Trees"
        )

        results[name] = {
            "model": result["model"],
            "metrics": metrics
        }

        print(
            f"\n--- {name} ---"
        )

        print(metrics)


    # ==========================================
    # PHASE 7 - GRID SEARCH
    # ==========================================

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


    # ==========================================
    # PHASE 7 - RANDOMIZED SEARCH
    # ==========================================

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


    # ==========================================
    # PHASE 6 - MODEL COMPARISON
    # ==========================================

    print("\n================================")
    print("MODEL COMPARISON")
    print("================================")

    for name, result in results.items():

        print(
            f"{name}: "
            f"R² = {result['metrics']['R2']:.4f}"
        )


    # ==========================================
    # PHASE 8 - BEST MODEL
    # ==========================================

    best_model_result = select_best_model(
        results
    )

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


    # ==========================================
    # PHASE 8 - SAVE MODEL
    # ==========================================

    save_model(
        best_model_result["model"],
        "models/final_model.pkl"
    )


    print(
        "\n✅ Phase 8 completed."
    )

    print(
        "Final model saved at:"
        " models/final_model.pkl"
    )


if __name__ == "__main__":
    main()
    