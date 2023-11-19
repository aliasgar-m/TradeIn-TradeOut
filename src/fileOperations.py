import pandas as pd


class FileOperations:
    @staticmethod
    def load_raw_dataset(filepath: str) -> pd.DataFrame:
        schema = {
            "stock_id": "Int64",
            "date_id": "Int64",
            "seconds_in_bucket": "Int64",
            "imbalance_size": "float32",
            "imbalance_buy_sell_flag": "float32",
            "reference_price": "float32",
            "matched_size": "float32",
            "far_price": "float32",
            "near_price": "float32",
            "bid_price": "float32",
            "bid_size": "float32",
            "ask_price": "float32",
            "ask_size": "float32",
            "wap": "float32",
            "target": "float32",
            "time_id": "Int64",
            "row_id": "string"
        }

        return pd.read_csv(filepath, dtype=schema)


    @staticmethod
    def save_prep_dataset(df: pd.DataFrame, filepath: str) -> None:
        df.to_csv(filepath)
        return
