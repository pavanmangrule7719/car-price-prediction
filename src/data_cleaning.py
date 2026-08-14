def remove_duplicates(df):
    duplicate_count = df.duplicated().sum()
    print(f"Duplicate rows found: {duplicate_count}")
    df = df.drop_duplicates()
    print(f"Duplicate rows remaining: {df.duplicated().sum()}")
    return df

def save_cleaned_dataset(df, output_path):
    df.to_csv(output_path, index=False)

    print(f"\nCleaned dataset saved successfully: {output_path}")
    print(f"Cleaned dataset shape: {df.shape}")