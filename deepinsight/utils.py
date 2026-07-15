import numpy as np
from sklearn.preprocessing import MinMaxScaler

class Normalizer:
    """Wrapper around MinMaxScaler to normalize features in the [0, 1] range."""
    def __init__(self):
        self.scaler = MinMaxScaler()

    def fit(self, X):
        self.scaler.fit(X)
        return self

    def transform(self, X):
        return self.scaler.transform(X)

    def fit_transform(self, X):
        return self.scaler.fit_transform(X)
