from pathlib import Path
import pandas as pd


SUPPORTED_EXTENSIONS = [".csv", ".xlsx", ".json", ".parquet"]


def load_dataset(file_path):
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {file_path}")

    if not path.is_file():
        raise ValueError(f"Path is not a file: {file_path}")

    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file format: {path.suffix}")

    if path.stat().st_size == 0:
        raise ValueError("Dataset file is empty.")

    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)

    elif path.suffix.lower() == ".xlsx":
        return pd.read_excel(path)

    elif path.suffix.lower() == ".json":
        return pd.read_json(path)

    elif path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    