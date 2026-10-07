"""Pipeline completa di DeepInsight:
Esegue in sequenza:
  1. Preprocessing e Normalizzazione (1_preprocess.py)
  2. Mappatura Feature e Bounding Box Layout (2_fit_layout.py)
  3. Trasformazione in Immagini 2D (3_transform.py)

Utilizzo:
    python3 deepinsight/deepinsight.py

Oppure con Scalene per profilare l'intera pipeline:
    scalene run -o profile_deepinsight.json deepinsight/deepinsight.py
"""

import os
import sys
import time
import importlib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

preprocess_module = importlib.import_module("1_preprocess")
fit_layout_module = importlib.import_module("2_fit_layout")
transform_module = importlib.import_module("3_transform")


def run_pipeline():
    print("=" * 65)
    print("      AVVIO PIPELINE COMPLETA DEEPINSIGHT (Fasi 1, 2, 3)")
    print("=" * 65)
    total_start_time = time.time()

    # --- FASE 1: Preprocessing & Normalization ---
    print("\n>>> ESECUZIONE FASE 1: Preprocessing & Normalization...")
    t0 = time.time()
    preprocess_module.main()
    print(f">>> Fase 1 completata in {time.time() - t0:.2f}s.\n")

    # --- FASE 2: Fit Layout (PCA / Discretization) ---
    print(">>> ESECUZIONE FASE 2: Feature Mapping & Fitting Layout...")
    t1 = time.time()
    fit_layout_module.main()
    print(f">>> Fase 2 completata in {time.time() - t1:.2f}s.\n")

    # --- FASE 3: Transform to Images ---
    print(">>> ESECUZIONE FASE 3: Generazione Immagini...")
    t2 = time.time()
    transform_module.main()
    print(f">>> Fase 3 completata in {time.time() - t2:.2f}s.\n")

    total_duration = time.time() - total_start_time
    print("=" * 65)
    print(f" PIPELINE DEEPINSIGHT COMPLETATA CON SUCCESSO IN {total_duration:.2f}s")
    print(" Output generati in: deepinsight/data/, deepinsight/models/, deepinsight/image/")
    print("=" * 65)


def main():
    run_pipeline()


if __name__ == "__main__":
    main()
