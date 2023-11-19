import pandas as pd


class Preprocess:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def get_null_values_per_column(self):
        print(self.df.isnull().sum())
        return self

    def remove_null_values(self, column: str):
        self.df = self.df.dropna(subset=[column])
        return self

    def fill_null_values(self, column: list[str] = None, value: int | float = None, type_fill: str = None):
        if type_fill is None:
            self.df.loc[:, column] = self.df[column].fillna(value)
            return self

        elif type_fill == 'mean':
            input_cols = [c for c in self.df.columns if c != "row_id"]
            output_cols = [c for c in self.df.columns if c != "row_id"]

            self.impute_columns(input_cols, output_cols)
            return self

    def impute_columns(self, input_cols, output_cols):
        for input_col, output_col in zip(input_cols, output_cols):
            mean_value = self.df[input_col].mean()

            self.df.loc[:, output_col] = self.df[input_col].fillna(mean_value)
        return self
