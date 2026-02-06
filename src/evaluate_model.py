import pandas as pd
from sklearn.model_selection import cross_val_score


def cross_validate_model(pipeline, X, y, cv=10):
    scores = -cross_val_score(
        pipeline,
        X,
        y,
        scoring="neg_root_mean_squared_error",
        cv=cv
    )

    return pd.Series(scores).describe()
