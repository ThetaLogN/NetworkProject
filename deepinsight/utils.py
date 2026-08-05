import numpy as np
from sklearn.preprocessing import MinMaxScaler

class Normalizer:
    """Normalizza le feature in [0, 1].

    Le feature di traffico di rete sono a coda lunga (pochi flussi con valori
    enormi): un MinMax puro schiaccia i valori tipici a ~0 e produce immagini quasi
    nere. Con log_transform=True si applica log1p prima del MinMax (la "norm-2" di
    DeepInsight), che distribuisce i valori in modo utilizzabile.
    """
    def __init__(self, log_transform=True):
        self.scaler = MinMaxScaler()
        self.log_transform = log_transform

    def _prep(self, X):
        # clip dei valori negativi (es. sentinella -1 di CICIDS) prima del log.
        return np.log1p(np.clip(X, 0, None)) if self.log_transform else X

    def fit(self, X):
        self.scaler.fit(self._prep(X))
        return self

    def transform(self, X):
        return self.scaler.transform(self._prep(X))

    def fit_transform(self, X):
        return self.scaler.fit_transform(self._prep(X))
