import pandas as pd


class FeatureEngineer:
    def __init__(self, df: pd.DataFrame) -> None:
        self.df = df.copy()

    def generate_spread(self):
        self.df.loc[:, 'spread'] = self.df['bid_price'] - self.df['ask_price']
        return self

    def generate_imbalance_ratio(self):
        self.df.loc[:, 'imbalance_ratio'] = self.df['imbalance_size'] / self.df['matched_size']
        return self

    def generate_volume(self):
        self.df.loc[:, 'volume'] = self.df['bid_size'] + self.df['ask_size']
        return self

    def generate_mid_price(self):
        self.df.loc[:, 'mid_price'] = (self.df['ask_price'] + self.df['bid_price']) / 2
        return self

    def liquidity_imbalance(self):
        self.df.loc[:, 'liquidity_imbalance'] = ((self.df['bid_size'] - self.df['ask_size']) /
                                          (self.df['bid_size'] + self.df['ask_size']))
        return self

    def match_ratio(self):
        self.df.loc[:, 'matched_ratio'] = self.df['matched_size'] / (self.df['bid_size'] + self.df['ask_size'])
        return self

    def generate_minutes(self):
        self.df.loc[:, 'minutes'] = self.df['seconds_in_bucket'] // 60
        return self
