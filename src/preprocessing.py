def split_features_target(df):
    X = df.drop("price", axis=1)
    y = df["price"]

    return X, y

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer


def create_preprocessor():

    categorical_columns = [
        "model",
        "transmission",
        "fuelType"
    ]

    numerical_columns = [
        "year",
        "mileage",
        "tax",
        "mpg",
        "engineSize"
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                categorical_columns
            ),
            (
                "numerical",
                StandardScaler(),
                numerical_columns
            )
        ]
    )

    return preprocessor