from src.data_loader import load_dataset
from src.data_cleaning import remove_duplicates, save_cleaned_dataset
from src.preprocessing import split_features_target
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def main():
    dataset_path = "data/audi.csv"

    df = load_dataset(dataset_path)

    print("✅ Dataset loaded successfully.")
    print("\nShape:", df.shape)

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

    print("\nMissing Values After Duplicate Removal:")
    print(df.isnull().sum())

    save_cleaned_dataset(df, "data/cleaned_audi.csv")

    X, y = split_features_target(df)

    print("\nFeatures Shape:", X.shape)
    print("Target Shape:", y.shape)

    print("\nFeatures:")
    print(X.head())

    print("\nTarget:")
    print(y.head())

    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2,random_state = 42)

    print("\nTraining Features Shape:", X_train.shape)
    print("Testing Features Shape:", X_test.shape)

    print("\nTraining Target Shape:", y_train.shape)
    print("Testing Target Shape:", y_test.shape)

    categorical_columns = ["model", "transmission", "fuelType"]
    numerical_columns = ["year","mileage","tax","mpg","engineSize"]

    preprocessor = ColumnTransformer(
    transformers=[
        ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_columns),
        ("numerical", StandardScaler(), numerical_columns)
    ])

    preprocessing_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor)
    ])

    X_train_processed = preprocessing_pipeline.fit_transform(X_train)
    X_test_processed = preprocessing_pipeline.transform(X_test)

    print("\nProcessed Training Shape:", X_train_processed.shape)
    print("Processed Testing Shape:", X_test_processed.shape)
    
if __name__ == "__main__":
    main()
    