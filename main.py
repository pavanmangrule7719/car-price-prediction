from src.data_loader import load_dataset

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

if __name__ == "__main__":
    main()