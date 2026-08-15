def split_features_target(df):
    X = df.drop("price", axis=1)
    y = df["price"]

    return X, y
