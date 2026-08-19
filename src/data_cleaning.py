def remove_duplicates(df):
    before = len(df)

    df = df.drop_duplicates().reset_index(drop=True)

    after = len(df)

    print(f"Removed duplicates: {before - after}")

    return df


def save_cleaned_dataset(df, file_path):
    df.to_csv(file_path, index=False)

    print(f"Cleaned dataset saved: {file_path}")