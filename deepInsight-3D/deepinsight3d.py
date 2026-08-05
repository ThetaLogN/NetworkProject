"""DeepInsight-3D: core generico per convertire dati multi-layer in immagini.

Riferimento: Sharma et al., "DeepInsight-3D for precision oncology"
(bioRxiv 2022.07.14.500140).

Il metodo lavora su dati multi-layer dove ogni layer ha gli STESSI elementi nello
STESSO ordine, ma valori diversi (es. multi-omics: stesso gene, misure diverse).
Meccanica (eq. 1-3 del paper):
  1. Il layout dei pixel si calcola SOLO sul layer dominante:  P_1 = H(M_dominant)
  2. Gli altri layer riusano P_1 (non ricalcolano il layout) e applicano solo la
     mappatura -> tutti i canali sono spazialmente allineati.
  3. L layer -> immagine a L canali (tensore (N, L, H, W); se L==3 anche RGB).

Questo modulo e' DATASET-AGNOSTICO: riceve i layer gia' costruiti. La scelta di
QUALI sono i canali e' una fase separata a monte (vedi prepare_cicids_channels.py).

Uso come libreria:
    from deepinsight3d import DeepInsight3D
    di3d = DeepInsight3D(pixels=(40, 40))
    volume = di3d.fit_transform([M_fwd, M_bwd, M_ratio])   # -> (N, 3, 40, 40)
    rgb = di3d.to_rgb(volume)                               # -> (N, 40, 40, 3)

Uso da riga di comando (legge i layer da un .npz, chiavi = nomi canali):
    python3 deepinsight3d.py output/cicids_3d/layers.npz --output-dir output/cicids_3d
"""

import argparse
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
from pyDeepInsight import ImageTransformer
from sklearn.preprocessing import MinMaxScaler


class DeepInsight3D:
    """Converte una lista di layer allineati in un volume immagine multi-canale.

    Args:
        pixels: dimensione (H, W) della griglia immagine.
        feature_extractor: riduttore per il layout ('tsne', 'pca', 'kpca').
        discretization: metodo di discretizzazione coordinate ('bin', 'qtb').
        dominant: indice del layer dominante su cui calcolare il layout condiviso.
    """

    def __init__(self, pixels=(40, 40), feature_extractor='kpca',
                 discretization='qtb', dominant=0, log_transform=True):
        self.pixels = pixels
        self.feature_extractor = feature_extractor
        self.discretization = discretization
        self.dominant = dominant
        # log_transform: applica log1p prima del MinMax. I dati di rete sono a coda
        # lunga (norm-2 del paper); senza, il MinMax schiaccia i valori tipici a ~0.
        self.log_transform = log_transform
        self._it = ImageTransformer(feature_extractor=feature_extractor,
                                    discretization=discretization, pixels=pixels)
        self._scalers = None
        self._n_layers = None
        self._n_elements = None

    def _prep(self, layer):
        """Trasformazione a coda lunga (log1p su valori non negativi) opzionale."""
        if self.log_transform:
            return np.log1p(np.clip(layer, 0, None))
        return layer

    @staticmethod
    def _check_layers(layers):
        if len(layers) == 0:
            raise ValueError("Serve almeno un layer.")
        shapes = [np.asarray(l).shape for l in layers]
        if len({s for s in shapes}) != 1:
            raise ValueError(f"Tutti i layer devono avere la stessa shape (n, d); "
                             f"ricevute: {shapes}")
        if len(shapes[0]) != 2:
            raise ValueError(f"Ogni layer deve essere 2D (n_campioni, n_elementi); "
                             f"ricevuto {shapes[0]}")
        return [np.asarray(l, dtype=np.float64) for l in layers]

    def fit(self, layers):
        layers = self._check_layers(layers)
        self._n_layers = len(layers)
        self._n_elements = layers[0].shape[1]
        if not 0 <= self.dominant < self._n_layers:
            raise ValueError(f"dominant={self.dominant} fuori range "
                             f"[0, {self._n_layers}).")

        # Normalizza ogni layer in [0, 1] con uno scaler indipendente (i layer sono
        # misure diverse con scale diverse), previa trasformazione log a coda lunga.
        self._scalers = [MinMaxScaler().fit(self._prep(l)) for l in layers]

        # Layout condiviso: fittato SOLO sul layer dominante (eq. 1 del paper).
        dom = layers[self.dominant]
        dominant_norm = self._scalers[self.dominant].transform(self._prep(dom))
        self._it.fit(dominant_norm)
        return self

    def transform(self, layers):
        layers = self._check_layers(layers)
        if self._scalers is None:
            raise RuntimeError("Chiamare fit prima di transform.")
        if len(layers) != self._n_layers:
            raise ValueError(f"Attesi {self._n_layers} layer, ricevuti {len(layers)}.")
        if layers[0].shape[1] != self._n_elements:
            raise ValueError(f"Attesi {self._n_elements} elementi per layer.")

        # Stesso layout (self._it) applicato a ogni layer: l'elemento i e' nella
        # stessa colonna in tutti i layer, quindi finisce nello stesso pixel (eq. 2-3).
        channels = []
        for scaler, layer in zip(self._scalers, layers):
            layer_norm = np.clip(scaler.transform(self._prep(layer)), 0, 1)
            channels.append(self._it.transform(layer_norm, img_format='scalar'))
        return np.stack(channels, axis=1).astype(np.float32)   # (N, L, H, W)

    def fit_transform(self, layers):
        return self.fit(layers).transform(layers)

    @staticmethod
    def to_rgb(volume):
        """Converte un volume (N, 3, H, W) in immagini RGB (N, H, W, 3) uint8."""
        if volume.shape[1] != 3:
            raise ValueError("to_rgb richiede esattamente 3 canali.")
        rgb = np.moveaxis(volume, 1, -1)                # (N, H, W, 3)
        return (np.clip(rgb, 0, 1) * 255).astype(np.uint8)

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path):
        with open(path, 'rb') as f:
            return pickle.load(f)


def plot_channel_samples(volume, y, channel_names, save_path, class_names=None):
    """Salva una griglia (canali x classe) di un campione per classe.

    Per ogni classe sceglie il campione che massimizza l'attivazione MINIMA tra i
    canali, così l'esempio mostrato è pieno su tutti i canali (evita i flussi
    unidirezionali che avrebbero il canale backward vuoto).
    """
    classes = np.unique(y)
    n_ch = volume.shape[1]
    # attivazione per (campione, canale), poi il minimo tra i canali
    per_ch = volume.reshape(volume.shape[0], n_ch, -1).sum(axis=2)
    min_activation = per_ch.min(axis=1)
    rep_idx = {c: np.where(y == c)[0][np.argmax(min_activation[y == c])] for c in classes}
    fig, axes = plt.subplots(n_ch, len(classes),
                             figsize=(4 * len(classes), 4 * n_ch), squeeze=False)
    for ci in range(n_ch):
        for cj, c in enumerate(classes):
            idx = rep_idx[c]
            ax = axes[ci][cj]
            ax.imshow(volume[idx, ci], cmap='inferno', interpolation='nearest')
            cls = class_names[c] if class_names is not None else f"classe {c}"
            ax.set_title(f"{cls} — {channel_names[ci]}")
            ax.axis('off')
    plt.suptitle("DeepInsight-3D: canali per classe (stesse posizioni-pixel)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="DeepInsight-3D: da layer allineati (.npz) a volume immagine.")
    parser.add_argument("layers_npz", help="File .npz con i layer (chiavi = nomi canali)")
    parser.add_argument("--labels", default=None,
                        help="File .npy con le label (default: <dir_npz>/labels.npy se esiste)")
    parser.add_argument("--output-dir", required=True, help="Cartella di output")
    parser.add_argument("--pixels", type=int, default=40, help="Lato immagine (default: 40)")
    parser.add_argument("--feature-extractor", default="kpca",
                        choices=["tsne", "pca", "kpca"],
                        help="Riduttore per il layout (default: kpca)")
    parser.add_argument("--discretization", default="qtb", choices=["bin", "qtb"])
    parser.add_argument("--dominant", type=int, default=0,
                        help="Indice del layer dominante per il layout (default: 0)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    npz = np.load(args.layers_npz)
    channel_names = list(npz.keys())
    layers = [npz[k] for k in channel_names]
    print(f"Caricati {len(layers)} layer da {args.layers_npz}: {channel_names}")
    print(f"Shape per layer: {layers[0].shape}")

    di3d = DeepInsight3D(pixels=(args.pixels, args.pixels),
                         feature_extractor=args.feature_extractor,
                         discretization=args.discretization, dominant=args.dominant)
    print(f"Fit del layout sul layer dominante '{channel_names[args.dominant]}' "
          f"e trasformazione...")
    volume = di3d.fit_transform(layers)
    print(f"Volume generato: {volume.shape} (dtype={volume.dtype})")

    np.save(os.path.join(args.output_dir, "volume.npy"), volume)
    di3d.save(os.path.join(args.output_dir, "transformer3d.pkl"))

    if volume.shape[1] == 3:
        rgb = DeepInsight3D.to_rgb(volume)
        np.save(os.path.join(args.output_dir, "rgb.npy"), rgb)
        print(f"Immagine RGB salvata: {rgb.shape}")

    # Label opzionali per la visualizzazione
    labels_path = args.labels
    if labels_path is None:
        guess = os.path.join(os.path.dirname(args.layers_npz), "labels.npy")
        labels_path = guess if os.path.exists(guess) else None
    if labels_path is not None:
        y = np.load(labels_path)
        plot_channel_samples(volume, y, channel_names,
                             os.path.join(args.output_dir, "channel_samples.png"))
        print("Salvato channel_samples.png")

    print(f"Output salvati in: {args.output_dir}")


if __name__ == "__main__":
    main()
