"""Fase 1 (selezione canali) per DeepInsight-3D su CICIDS2017.

Costruisce 3 layer ALLINEATI a partire dalle direzioni del flusso di rete. Ogni
layer contiene gli stessi 22 elementi (metriche di base) nello stesso ordine:
  - fwd:   la metrica misurata in direzione forward
  - bwd:   la stessa metrica misurata in direzione backward
  - terzo: canale derivato (ratio fwd/bwd oppure somma), mantiene l'allineamento

I layer prodotti vengono salvati in un .npz che si passa al core generico
(deepinsight3d.py). Questo modulo e' il pezzo modificabile/sostituibile: qui si
decide COSA sono i canali. Il core resta agnostico.

Uso:
    python3 prepare_cicids_channels.py ../dataset/CICIDS2017_sample.csv \
        --label-column Label --binary-benign BENIGN --output-dir output/cicids_3d
"""

import argparse
import os

import numpy as np
import pandas as pd

# 22 elementi appaiati: (nome_elemento, colonna_fwd, colonna_bwd) con i nomi esatti
# delle colonne CICIDS2017.
DIRECTIONAL_PAIRS = [
    ("Total Packets",        "Total Fwd Packets",          "Total Backward Packets"),
    ("Total Length",         "Total Length of Fwd Packets", "Total Length of Bwd Packets"),
    ("Packet Length Max",    "Fwd Packet Length Max",      "Bwd Packet Length Max"),
    ("Packet Length Min",    "Fwd Packet Length Min",      "Bwd Packet Length Min"),
    ("Packet Length Mean",   "Fwd Packet Length Mean",     "Bwd Packet Length Mean"),
    ("Packet Length Std",    "Fwd Packet Length Std",      "Bwd Packet Length Std"),
    ("IAT Total",            "Fwd IAT Total",              "Bwd IAT Total"),
    ("IAT Mean",             "Fwd IAT Mean",               "Bwd IAT Mean"),
    ("IAT Std",              "Fwd IAT Std",                "Bwd IAT Std"),
    ("IAT Max",              "Fwd IAT Max",                "Bwd IAT Max"),
    ("IAT Min",              "Fwd IAT Min",                "Bwd IAT Min"),
    ("PSH Flags",            "Fwd PSH Flags",              "Bwd PSH Flags"),
    ("URG Flags",            "Fwd URG Flags",              "Bwd URG Flags"),
    ("Header Length",        "Fwd Header Length",          "Bwd Header Length"),
    ("Packets/s",            "Fwd Packets/s",              "Bwd Packets/s"),
    ("Avg Segment Size",     "Avg Fwd Segment Size",       "Avg Bwd Segment Size"),
    ("Avg Bytes/Bulk",       "Fwd Avg Bytes/Bulk",         "Bwd Avg Bytes/Bulk"),
    ("Avg Packets/Bulk",     "Fwd Avg Packets/Bulk",       "Bwd Avg Packets/Bulk"),
    ("Avg Bulk Rate",        "Fwd Avg Bulk Rate",          "Bwd Avg Bulk Rate"),
    ("Subflow Packets",      "Subflow Fwd Packets",        "Subflow Bwd Packets"),
    ("Subflow Bytes",        "Subflow Fwd Bytes",          "Subflow Bwd Bytes"),
    ("Init_Win_bytes",       "Init_Win_bytes_forward",     "Init_Win_bytes_backward"),
]

EPS = 1e-8


def load_and_clean(csv_path, label_column, binary_benign=None):
    """Carica il CSV, pulisce inf/NaN, ritorna (df_pulito, y, class_names)."""
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    if label_column not in df.columns:
        raise ValueError(f"Colonna label '{label_column}' non trovata. "
                         f"Colonne: {list(df.columns)}")
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    labels = df[label_column]
    if binary_benign is not None:
        y = np.where(labels.astype(str) == binary_benign, 0, 1)
        class_names = [binary_benign, "OTHER"]
    else:
        class_names, y = np.unique(labels.astype(str), return_inverse=True)
        class_names = class_names.tolist()
    return df, y, class_names


def build_directional_layers(df, third="ratio"):
    """Costruisce i layer fwd, bwd e (opzionale) un terzo canale derivato.

    Ritorna (layers_dict, elements). layers_dict mappa nome_canale -> (n, 22).
    """
    missing = [c for _, f, b in DIRECTIONAL_PAIRS for c in (f, b) if c not in df.columns]
    if missing:
        raise ValueError(f"Colonne mancanti nel dataset: {missing}")

    elements = [name for name, _, _ in DIRECTIONAL_PAIRS]
    fwd = df[[f for _, f, _ in DIRECTIONAL_PAIRS]].to_numpy(dtype=np.float64)
    bwd = df[[b for _, _, b in DIRECTIONAL_PAIRS]].to_numpy(dtype=np.float64)

    layers = {"fwd": fwd, "bwd": bwd}
    if third == "ratio":
        layers["ratio"] = fwd / (bwd + EPS)
    elif third == "sum":
        layers["sum"] = fwd + bwd
    elif third == "none":
        pass
    else:
        raise ValueError(f"--third non valido: {third}")
    return layers, elements


def main():
    parser = argparse.ArgumentParser(
        description="Fase 1: costruisce i layer direzionali CICIDS per DeepInsight-3D.")
    parser.add_argument("csv_path", help="Percorso del CSV CICIDS2017")
    parser.add_argument("--label-column", required=True, help="Nome della colonna label")
    parser.add_argument("--binary-benign", default=None,
                        help="Se indicato, label uguali a questo valore -> 0, altre -> 1")
    parser.add_argument("--output-dir", required=True, help="Cartella di output")
    parser.add_argument("--third", default="ratio", choices=["ratio", "sum", "none"],
                        help="Terzo canale derivato (default: ratio)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Caricamento {args.csv_path}...")
    df, y, class_names = load_and_clean(args.csv_path, args.label_column, args.binary_benign)
    print(f"Righe pulite: {len(df)} | classi: {len(class_names)}")

    layers, elements = build_directional_layers(df, third=args.third)
    print(f"Layer costruiti: {list(layers.keys())} | elementi per layer: {len(elements)}")

    np.savez(os.path.join(args.output_dir, "layers.npz"), **layers)
    np.save(os.path.join(args.output_dir, "labels.npy"), y)
    with open(os.path.join(args.output_dir, "elements.txt"), "w") as f:
        f.write("\n".join(f"{name}\t{fwd}\t{bwd}"
                          for name, fwd, bwd in DIRECTIONAL_PAIRS))
    with open(os.path.join(args.output_dir, "class_names.txt"), "w") as f:
        f.write("\n".join(f"{i}\t{name}" for i, name in enumerate(class_names)))

    print(f"Salvati layers.npz, labels.npy, elements.txt in: {args.output_dir}")


if __name__ == "__main__":
    main()
