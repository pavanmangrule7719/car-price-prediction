import pickle
from pathlib import Path


def save_model(model, file_path):

    path = Path(file_path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(path, "wb") as file:

        pickle.dump(
            model,
            file
        )

    print(
        f"✅ Model saved successfully: {path}"
    )

def load_model(file_path):

    with open(file_path, "rb") as file:

        model = pickle.load(file)

    return model
