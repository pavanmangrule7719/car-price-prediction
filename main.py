from src.data_loader import load_dataset
from src.data_cleaning import remove_duplicates, save_cleaned_dataset

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

if __name__ == "__main__":
    main()

