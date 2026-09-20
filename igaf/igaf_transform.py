"""Conversione di dataset tabulari in immagini RGB tramite Improved Gramian
Angular Field (iGAF).

Riferimento: Improved GAF, MDPI Electronics 2023
https://www.mdpi.com/2079-9292/12/11/2540

Ogni riga del dataset (un campione con N feature) viene trattata come una serie e
convertita in un'immagine a 3 canali RGB in cui ogni canale porta un'informazione
distinta:
  - FFT del segnale -> spettro di ampiezza e di fase
  - 3 matrici di Gram (GADF): G_D (dati grezzi), G_M (ampiezza), G_P (fase)
  - mapping RGB: R <- G_P (fase), G <- G_M (ampiezza), B <- G_D (dati grezzi)

Output per il dataset: tensore (N, H, W, 3) uint8.

Uso da riga di comando:
    python3 igaf_transform.py ../dataset/CICIDS2017_sample.csv \
        --label-column Label --binary-benign BENIGN \
        --output-dir output/cicids_igaf

Uso come libreria:
    from igaf_transform import IGAFImageEncoder
    enc = IGAFImageEncoder(image_size=None)
    images = enc.fit_transform(X)      # X: (n_samples, n_features) -> (N, H, W, 3)
"""

import argparse
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pyts.image import GramianAngularField


class IGAFImageEncoder:
    """Trasforma vettori di feature tabulari in immagini RGB iGAF.

    Per ogni campione:
      1. FFT completa del vettore -> ampiezza (|FFT|) e fase (angle(FFT)).
      2. Tre matrici GADF (metodo 'difference') su dati grezzi, ampiezza, fase,
         ognuna riscalata indipendentemente in [0, 1] (richiesto dall'arccos).
      3. Ogni matrice mappata in [0, 255] uint8 e assegnata a un canale:
         R = fase, G = ampiezza, B = dati grezzi.

    Args:
        image_size: dimensione dell'immagine di output. None = usa l'intera
            lunghezza N della serie (matrici N x N); int <= N per ridurre via
            PAA interna a pyts; float in (0, 1] = frazione della lunghezza.
    """

    def __init__(self, image_size=None):
        self.image_size = image_size
        # sample_range=(0,1): pyts riscala ogni serie indipendentemente prima
        # dell'arccos; essenziale perche ampiezza e fase hanno scale diverse.
        size = 1.0 if image_size is None else image_size
        self._gadf = GramianAngularField(image_size=size, method='difference',
                                         sample_range=(0, 1))

    @staticmethod
    def _to_uint8(mat):
        """Mappa linearmente una matrice in [0, 255] uint8."""
        lo, hi = mat.min(), mat.max()
        norm = (mat - lo) / (hi - lo + 1e-8)
        return (norm * 255).astype(np.uint8)

    def _gram(self, series_2d):
        """Applica GADF a un batch di serie (n_samples, n_points)."""
        return self._gadf.fit_transform(series_2d)

    def fit(self, X):
        # iGAF non ha parametri appresi dai dati; API fit/transform per coerenza.
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=np.float64)

        # 1. Analisi in frequenza: FFT completa (lunghezza N per allineare le
        #    tre matrici che diventeranno i canali RGB).
        fft = np.fft.fft(X, axis=1)
        amplitude = np.abs(fft)
        phase = np.angle(fft)
        raw = X

        # 2. Tre matrici di Gram (GADF), una per rappresentazione.
        g_d = self._gram(raw)         # (N, H, W) dati grezzi
        g_m = self._gram(amplitude)   # (N, H, W) ampiezza
        g_p = self._gram(phase)       # (N, H, W) fase

        # 3. Mapping RGB per campione: R=fase, G=ampiezza, B=dati grezzi.
        n = X.shape[0]
        h, w = g_d.shape[1], g_d.shape[2]
        images = np.zeros((n, h, w, 3), dtype=np.uint8)
        for i in range(n):
            images[i, :, :, 0] = self._to_uint8(g_p[i])   # R
            images[i, :, :, 1] = self._to_uint8(g_m[i])   # G
            images[i, :, :, 2] = self._to_uint8(g_d[i])   # B
        return images

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    @property
    def channel_names(self):
        return ['R=fase (G_P)', 'G=ampiezza (G_M)', 'B=dati grezzi (G_D)']

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


def plot_class_samples(images, y, class_names, save_path):
    """Salva una figura con un'immagine RGB iGAF per classe."""
    classes = np.unique(y)
    fig, axes = plt.subplots(1, len(classes), figsize=(4 * len(classes), 4.5),
                             squeeze=False)
    for cj, c in enumerate(classes):
        idx = np.where(y == c)[0][0]
        ax = axes[0][cj]
        ax.imshow(images[idx], origin='lower')   # immagine RGB già in [0,255]
        ax.set_title(class_names[c])
        ax.axis('off')
    plt.suptitle("Immagini iGAF (RGB: R=fase, G=ampiezza, B=grezzo) per classe")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def run(csv_path, label_column, output_dir, binary_benign=None, image_size=None):
    os.makedirs(output_dir, exist_ok=True)

    print(f"Caricamento {csv_path}...")
    X, y, feature_names, class_names = load_dataset(csv_path, label_column, binary_benign)
    print(f"Dataset: {X.shape[0]} campioni, {X.shape[1]} feature numeriche, "
          f"{len(class_names)} classi")

    enc = IGAFImageEncoder(image_size=image_size)
    print(f"Trasformazione iGAF (image_size={image_size})...")
    images = enc.fit_transform(X)
    print(f"Tensore immagini generato: {images.shape} (dtype={images.dtype})")

    np.save(os.path.join(output_dir, "igaf_images.npy"), images)
    np.save(os.path.join(output_dir, "igaf_labels.npy"), y)
    with open(os.path.join(output_dir, "feature_names.txt"), "w") as f:
        f.write("\n".join(feature_names))
    with open(os.path.join(output_dir, "class_names.txt"), "w") as f:
        f.write("\n".join(f"{i}\t{name}" for i, name in enumerate(class_names)))
    enc.save(os.path.join(output_dir, "igaf_encoder.pkl"))
    plot_class_samples(images, y, class_names,
                       os.path.join(output_dir, "igaf_samples.png"))

    print(f"Output salvati in: {output_dir}")
    return images, y


def main():
    parser = argparse.ArgumentParser(
        description="Converte un dataset tabulare CSV in immagini RGB iGAF.")
    parser.add_argument("csv_path", help="Percorso del CSV di input")
    parser.add_argument("--label-column", required=True, help="Nome della colonna label")
    parser.add_argument("--output-dir", required=True, help="Cartella per gli output generati")
    parser.add_argument("--binary-benign", default=None,
                        help="Se indicato, le label uguali a questo valore diventano 0 "
                             "e tutte le altre 1")
    parser.add_argument("--image-size", type=float, default=None,
                        help="Dimensione immagine: vuoto = intera lunghezza serie; "
                             "intero <= N = pixel assoluti; frazione in (0,1] = quota")
    args = parser.parse_args()

    image_size = args.image_size
    if image_size is not None and image_size >= 1:
        image_size = int(image_size)
    run(args.csv_path, args.label_column, args.output_dir,
        binary_benign=args.binary_benign, image_size=image_size)


if __name__ == "__main__":
    main()
