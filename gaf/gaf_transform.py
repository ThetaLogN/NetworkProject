"""Conversione di dataset tabulari in immagini tramite Gramian Angular Field (GAF).

Ogni riga del dataset (un campione con N feature) viene trattata come una serie
e convertita in un'immagine GAF con pyts. Sono supportati:
  - GASF (Gramian Angular Summation Field)
  - GADF (Gramian Angular Difference Field)
  - entrambi impilati come 2 canali -> tensore (N, 2, H, W)
 
Uso da riga di comando:
    python3 gaf_transform.py ../dataset/CICIDS2017_sample.csv \
        --label-column Label --binary-benign BENIGN \
        --output-dir output/cicids --image-size 40 --method both

Uso come libreria:
    from gaf_transform import GAFImageEncoder
    enc = GAFImageEncoder(method='both', image_size=40)
    images = enc.fit_transform(X)      # X: (n_samples, n_features) -> (N, 2, H, W)
"""

import argparse
import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pyts.image import GramianAngularField


class GAFImageEncoder:
    """Trasforma vettori di feature tabulari in immagini GAF.

    Args:
        method: 'summation' (GASF), 'difference' (GADF) oppure 'both' (2 canali).
        image_size: dimensione dell'immagine di output. int = numero di pixel
            assoluto; float in (0, 1] = frazione della lunghezza della serie.
        sample_range: intervallo di riscalatura applicato da pyts prima della
            trasformazione (GAF richiede valori in [-1, 1]).
    """

    _METHODS = {
        'summation': ['summation'],
        'difference': ['difference'],
        'both': ['summation', 'difference'],
    }

    def __init__(self, method='both', image_size=1.0, sample_range=(-1, 1)):
        if method not in self._METHODS:
            raise ValueError(f"method deve essere uno di {list(self._METHODS)}, "
                             f"ricevuto '{method}'")
        self.method = method
        self.image_size = image_size
        self.sample_range = sample_range
        self._encoders = [
            GramianAngularField(image_size=image_size, method=m,
                                sample_range=sample_range)
            for m in self._METHODS[method]
        ]

    def fit(self, X):
        # GramianAngularField è stateless rispetto ai dati (nessun parametro
        # appreso), ma manteniamo l'API fit/transform per coerenza.
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=np.float64)
        channels = [enc.fit_transform(X).astype(np.float32) for enc in self._encoders]
        if len(channels) == 1:
            return channels[0]                      # (N, H, W)
        return np.stack(channels, axis=1)           # (N, C, H, W)

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    @property
    def channel_names(self):
        return self._METHODS[self.method]

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path):
        with open(path, 'rb') as f:
            return pickle.load(f)


def load_dataset(csv_path, label_column, binary_benign=None):
    """Carica un CSV e ritorna (X, y, feature_names, class_names).

    Pulisce righe con inf/NaN, tiene solo colonne numeriche come feature e
    codifica le label come interi (binaria se binary_benign è indicato,
    altrimenti multiclasse).
    """
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()

    if label_column not in df.columns:
        raise ValueError(f"Colonna label '{label_column}' non trovata. "
                         f"Colonne disponibili: {list(df.columns)}")

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    labels = df[label_column]
    features = df.drop(columns=[label_column]).select_dtypes(include=[np.number])
    if features.shape[1] == 0:
        raise ValueError("Nessuna colonna numerica trovata nel dataset.")

    if binary_benign is not None:
        y = np.where(labels.astype(str) == binary_benign, 0, 1)
        class_names = [binary_benign, "OTHER"]
    else:
        class_names, y = np.unique(labels.astype(str), return_inverse=True)
        class_names = class_names.tolist()

    return features.values, y, features.columns.tolist(), class_names


def plot_class_samples(images, y, class_names, channel_names, save_path):
    """Salva una figura con un campione GAF per classe (una riga per canale)."""
    classes = np.unique(y)
    # images: (N, H, W) oppure (N, C, H, W)
    if images.ndim == 3:
        images = images[:, None, :, :]
        channel_names = [channel_names[0] if channel_names else 'gaf']
    n_ch = images.shape[1]

    fig, axes = plt.subplots(n_ch, len(classes),
                             figsize=(4 * len(classes), 4 * n_ch),
                             squeeze=False)
    for ci in range(n_ch):
        for cj, c in enumerate(classes):
            idx = np.where(y == c)[0][0]
            ax = axes[ci][cj]
            ax.imshow(images[idx, ci], cmap='rainbow', origin='lower')
            ax.set_title(f"{class_names[c]} — {channel_names[ci]}")
            ax.axis('off')
    plt.suptitle("Immagini GAF per classe")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def run(csv_path, label_column, output_dir, binary_benign=None,
        method='both', image_size=40):
    os.makedirs(output_dir, exist_ok=True)

    print(f"Caricamento {csv_path}...")
    X, y, feature_names, class_names = load_dataset(csv_path, label_column, binary_benign)
    print(f"Dataset: {X.shape[0]} campioni, {X.shape[1]} feature numeriche, "
          f"{len(class_names)} classi")

    enc = GAFImageEncoder(method=method, image_size=image_size)
    print(f"Trasformazione GAF (method={method}, image_size={image_size})...")
    images = enc.fit_transform(X)
    print(f"Tensore immagini generato: {images.shape}")

    np.save(os.path.join(output_dir, "gaf_images.npy"), images)
    np.save(os.path.join(output_dir, "gaf_labels.npy"), y)
    with open(os.path.join(output_dir, "feature_names.txt"), "w") as f:
        f.write("\n".join(feature_names))
    with open(os.path.join(output_dir, "class_names.txt"), "w") as f:
        f.write("\n".join(f"{i}\t{name}" for i, name in enumerate(class_names)))
    enc.save(os.path.join(output_dir, "gaf_encoder.pkl"))
    plot_class_samples(images, y, class_names, enc.channel_names,
                       os.path.join(output_dir, "gaf_samples.png"))

    print(f"Output salvati in: {output_dir}")
    return images, y


def main():
    parser = argparse.ArgumentParser(
        description="Converte un dataset tabulare CSV in immagini GAF.")
    parser.add_argument("csv_path", help="Percorso del CSV di input")
    parser.add_argument("--label-column", required=True, help="Nome della colonna label")
    parser.add_argument("--output-dir", required=True, help="Cartella per gli output generati")
    parser.add_argument("--binary-benign", default=None,
                        help="Se indicato, le label uguali a questo valore diventano 0 "
                             "e tutte le altre 1")
    parser.add_argument("--method", default="both",
                        choices=["summation", "difference", "both"],
                        help="GASF (summation), GADF (difference) o entrambi (default: both)")
    parser.add_argument("--image-size", type=float, default=40,
                        help="Dimensione immagine: intero = pixel assoluti, "
                             "frazione in (0,1] = quota della lunghezza serie (default: 40)")
    args = parser.parse_args()

    # image_size intero se l'utente passa un valore >= 1
    image_size = int(args.image_size) if args.image_size >= 1 else args.image_size
    run(args.csv_path, args.label_column, args.output_dir,
        binary_benign=args.binary_benign, method=args.method, image_size=image_size)


if __name__ == "__main__":
    main()
