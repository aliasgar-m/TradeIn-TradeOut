import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def split_data(df: pd.DataFrame):
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    train_y: np.ndarray = train_df['target']
    train_x: pd.DataFrame = train_df.drop('target', axis=1)

    test_y: np.ndarray = test_df['target']
    test_x: pd.DataFrame = test_df.drop('target', axis=1)

    return train_x, train_y, test_x, test_y


def to_pandas(iterator):
    yield pd.DataFrame(iterator)
