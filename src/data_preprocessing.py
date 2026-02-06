import pandas as pd

DROP_COLS = ["Alley", "PoolQC", "Fence", "MiscFeature"]

def load_data(train_path, test_path):
    train_df = pd.read_csv(train_path)
    test_df  = pd.read_csv(test_path)
    return train_df, test_df


def drop_high_missing_columns(df):
    return df.drop(columns=DROP_COLS, errors="ignore")


def fill_missing_values(df):
    # Neighborhood-wise median for LotFrontage
    if "LotFrontage" in df.columns:
        df["LotFrontage"] = (
            df.groupby("Neighborhood")["LotFrontage"]
              .transform(lambda x: x.fillna(x.median()))
        )

    # Categorical → mode
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].fillna(df[col].mode()[0])

    # Numerical → median
    for col in df.select_dtypes(include=["int64", "float64"]).columns:
        df[col] = df[col].fillna(df[col].median())

    return df
