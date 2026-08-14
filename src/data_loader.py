import pandas as pd
import pathlib

Supported_Extensions = (".csv",".xlsx",".json",".parquet")

def validate_dataset_file(dataset_path):
    path = pathlib.Path(dataset_path)

    if not path.exists():
        raise FileNotFoundError("Dataset not found. Please check the path.")

    if not path.is_file():
        raise ValueError("The provided path is not a dataset file.")

    if path.suffix.lower() not in Supported_Extensions:
        raise ValueError(f"Unsupported dataset format: {path.suffix}")

    if path.stat().st_size == 0:
        raise ValueError("Dataset file is empty.")

    try:
        with open(dataset_path,"rb"):
            pass
    except PermissionError:
        raise PermissionError("Permission denied. Cannot read dataset.")

    return True

def load_dataset(dataset_path):

    validate_dataset_file(dataset_path)

    path = pathlib.Path(dataset_path)

    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)

    elif path.suffix.lower() == ".xlsx":
        return pd.read_excel(path)

    elif path.suffix.lower() == ".json":
        return pd.read_json(path)

    elif path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    