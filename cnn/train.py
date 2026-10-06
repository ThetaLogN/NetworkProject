"""Script di training per ResNet-18 Standard applicata ai dataset di anomaly detection:
  - deepinsight (40x40, 1 canale espanso a 3 canali RGB)
  - deepinsight3d (40x40, 3 canali RGB)
  - gaf (40x40, 2 canali espansi a 3 canali)
  - igaf (77x77, 3 canali RGB)

Esempio di utilizzo:
    python3 cnn/train.py --dataset deepinsight --epochs 10 --batch-size 128
    python3 cnn/train.py --dataset deepinsight3d --epochs 10
    python3 cnn/train.py --dataset gaf --epochs 10
    python3 cnn/train.py --dataset igaf --epochs 10
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (
    classification_report, confusion_matrix, f1_score, accuracy_score,
    precision_score, recall_score, roc_auc_score
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

# Permette l'importazione sia se lanciato dalla root che dalla cartella cnn
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from cnn.resnet18 import get_standard_resnet18


def load_dataset(dataset_name: str, root_dir: str):
    """Carica le immagini e le etichette, uniformandole a un tensore float32 (N, 3, H, W)."""
    dataset_name = dataset_name.lower().replace("-", "")
    
    if dataset_name == "deepinsight":
        img_path = os.path.join(root_dir, "deepinsight", "image", "cicids_images.npy")
        lbl_path = os.path.join(root_dir, "deepinsight", "image", "cicids_labels.npy")
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Immagini non trovate in: {img_path}")
        
        imgs = np.load(img_path, mmap_mode="r")
        labels = np.load(lbl_path)
        # (N, 40, 40) -> replica su 3 canali: (N, 3, 40, 40)
        X = np.stack([imgs, imgs, imgs], axis=1).astype(np.float32)

    elif dataset_name in ["deepinsight3d", "di3d"]:
        img_path = os.path.join(root_dir, "deepInsight-3D", "output", "cicids_3d", "rgb.npy")
        lbl_path = os.path.join(root_dir, "deepInsight-3D", "output", "cicids_3d", "labels.npy")
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Immagini non trovate in: {img_path}")
        
        imgs = np.load(img_path, mmap_mode="r")
        labels = np.load(lbl_path)
        # Da (N, 40, 40, 3) uint8 a (N, 3, 40, 40) float32 in [0, 1]
        X = np.transpose(imgs, (0, 3, 1, 2)).astype(np.float32) / 255.0

    elif dataset_name == "gaf":
        img_path = os.path.join(root_dir, "gaf", "output", "cicids", "gaf_images.npy")
        lbl_path = os.path.join(root_dir, "gaf", "output", "cicids", "gaf_labels.npy")
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Immagini non trovate in: {img_path}")
        
        imgs = np.load(img_path, mmap_mode="r")
        labels = np.load(lbl_path)
        # (N, 2, 40, 40) -> genera 3° canale come media dei due: (N, 3, 40, 40)
        ch3 = (imgs[:, 0:1, :, :] + imgs[:, 1:2, :, :]) / 2.0
        X = np.concatenate([imgs, ch3], axis=1).astype(np.float32)

    elif dataset_name == "igaf":
        img_path = os.path.join(root_dir, "igaf", "output", "cicids_igaf", "igaf_images.npy")
        lbl_path = os.path.join(root_dir, "igaf", "output", "cicids_igaf", "igaf_labels.npy")
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Immagini non trovate in: {img_path}")
        
        imgs = np.load(img_path, mmap_mode="r")
        labels = np.load(lbl_path)
        # Da (N, 77, 77, 3) uint8 a (N, 3, 77, 77) float32 in [0, 1]
        X = np.transpose(imgs, (0, 3, 1, 2)).astype(np.float32) / 255.0

    else:
        raise ValueError(f"Tecnica non riconosciuta: {dataset_name}. Scegli tra: deepinsight, deepinsight3d, gaf, igaf")

    y = labels.astype(np.int64)
    print(f"Caricato dataset '{dataset_name}': {X.shape[0]} campioni, shape tensore: {X.shape[1:]}")
    unique, counts = np.unique(y, return_counts=True)
    for u, c in zip(unique, counts):
        name = "BENIGN" if u == 0 else "ATTACK"
        print(f"  Classe {u} ({name}): {c} campioni ({c / len(y) * 100:.1f}%)")

    return X, y


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(targets).sum().item()
        total += targets.size(0)

    return running_loss / total, 100.0 * correct / total


def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_targets = []
    all_probs = []

    start_time = time.time()
    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * inputs.size(0)
            probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(targets.cpu().numpy())
            all_probs.extend(probs)

    inference_time = (time.time() - start_time) / len(all_targets) * 1000.0  # ms per sample
    val_loss = running_loss / len(all_targets)
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    all_probs = np.array(all_probs)

    val_acc = accuracy_score(all_targets, all_preds) * 100.0
    val_f1 = f1_score(all_targets, all_preds, average="macro")

    return val_loss, val_acc, val_f1, all_preds, all_targets, all_probs, inference_time


def main():
    parser = argparse.ArgumentParser(description="Addestramento ResNet-18 Standard per Network IDS")
    parser.add_argument("--dataset", type=str, default="deepinsight",
                        choices=["deepinsight", "deepinsight3d", "gaf", "igaf"],
                        help="Tecnica di codifica da testare")
    parser.add_argument("--epochs", type=int, default=10, help="Numero di epoche")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size per training")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--pretrained", action="store_true", help="Usa pesi preaddestrati ImageNet (default: False)")
    parser.add_argument("--test-size", type=float, default=0.2, help="Frazione di test/validation")
    parser.add_argument("--random-state", type=int, default=42, help="Seed per riproducibilità")
    args = parser.parse_args()

    # 1. Scelta dispositivo (Apple Silicon MPS se disponibile, altrimenti CUDA o CPU)
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"\n=======================================================")
    print(f" Modello:     ResNet-18 Standard (Pretrained={args.pretrained})")
    print(f" Dataset:     {args.dataset}")
    print(f" Dispositivo: {device}")
    print(f"=======================================================\n")

    # 2. Caricamento dati
    X, y = load_dataset(args.dataset, ROOT_DIR)

    # 3. Train / Validation Split stratificato
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state, stratify=y
    )

    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_train), torch.tensor(y_train)),
        batch_size=args.batch_size, shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(torch.tensor(X_val), torch.tensor(y_val)),
        batch_size=args.batch_size * 2, shuffle=False
    )

    # 4. Inizializzazione Modello
    model = get_standard_resnet18(pretrained=args.pretrained, num_classes=2).to(device)

    # Bilanciamento classi nella CrossEntropyLoss se necessario
    class_counts = np.bincount(y_train)
    weights = torch.tensor([len(y_train) / (2.0 * c) for c in class_counts], dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Directory di output per pesi e metriche
    save_dir = os.path.join(ROOT_DIR, "cnn", "checkpoints")
    results_dir = os.path.join(ROOT_DIR, "cnn", "results")
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    history = []
    best_val_f1 = 0.0
    best_epoch = 1
    time_to_best_epoch = 0.0
    best_checkpoint_path = os.path.join(save_dir, f"resnet18_best_{args.dataset}.pt")

    print("\nInizio addestramento...")
    total_start = time.time()

    for epoch in range(1, args.epochs + 1):
        ep_start = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        scheduler.step()

        val_loss, val_acc, val_f1, _, _, _, _ = evaluate(model, val_loader, criterion, device)
        ep_duration = time.time() - ep_start

        # Salvataggio del modello migliore
        saved_mark = ""
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch
            time_to_best_epoch = time.time() - total_start
            torch.save(model.state_dict(), best_checkpoint_path)
            saved_mark = " [*Best]"

        history.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "train_acc": round(train_acc, 2),
            "val_loss": round(val_loss, 4),
            "val_acc": round(val_acc, 2),
            "val_f1": round(val_f1, 4),
            "duration_s": round(ep_duration, 2)
        })

        print(f"Epoca [{epoch:02d}/{args.epochs:02d}] ({ep_duration:.1f}s) | "
              f"Train Loss: {train_loss:.4f} - Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f} - Acc: {val_acc:.2f}% - Macro F1: {val_f1:.4f}{saved_mark}")

    total_training_time = time.time() - total_start
    print(f"\nAddestramento completato in {total_training_time:.1f}s.")
    print(f"Caricamento dei pesi migliori da: {best_checkpoint_path}")
    model.load_state_dict(torch.load(best_checkpoint_path, map_location=device))

    # Valutazione finale dettagliata
    _, final_acc, final_f1, final_preds, final_targets, final_probs, inf_time = evaluate(model, val_loader, criterion, device)

    # Calcolo metriche avanzate di qualità
    f1_weighted = float(f1_score(final_targets, final_preds, average="weighted"))
    prec_macro = float(precision_score(final_targets, final_preds, average="macro"))
    prec_weighted = float(precision_score(final_targets, final_preds, average="weighted"))
    rec_macro = float(recall_score(final_targets, final_preds, average="macro"))
    rec_weighted = float(recall_score(final_targets, final_preds, average="weighted"))
    roc_auc = float(roc_auc_score(final_targets, final_probs))

    # Metriche per singola classe
    prec_per_class = precision_score(final_targets, final_preds, average=None)
    rec_per_class = recall_score(final_targets, final_preds, average=None)
    f1_per_class = f1_score(final_targets, final_preds, average=None)
    class_support = np.bincount(final_targets)

    # Stabilità e convergenza della loss
    val_loss_series = [h["val_loss"] for h in history]
    last_k = min(5, len(val_loss_series))
    loss_stability_std = float(np.std(val_loss_series[-last_k:]))
    final_train_loss = history[-1]["train_loss"]
    final_val_loss = history[-1]["val_loss"]
    loss_gap = float(abs(final_val_loss - final_train_loss))

    print("\n" + "=" * 55)
    print(f" REPORT DI VALUTAZIONE FINALE - {args.dataset.upper()}")
    print("=" * 55)
    print(f"Accuratezza Test:       {final_acc:.2f}%")
    print(f"Macro F1-Score:         {final_f1:.4f}")
    print(f"Weighted F1-Score:      {f1_weighted:.4f}")
    print(f"Macro Precision:        {prec_macro:.4f}")
    print(f"Macro Recall:           {rec_macro:.4f}")
    print(f"AUC-ROC:                {roc_auc:.4f}")
    print(f"Latenza di inferenza:   {inf_time:.3f} ms / campione")
    print(f"Epoca di convergenza:   Epoca {best_epoch} ({time_to_best_epoch:.1f}s)")
    print(f"Stabilità Loss (std u. 5 ep): {loss_stability_std:.5f}")
    print("\nMatrice di Confusione:")
    cm = confusion_matrix(final_targets, final_preds)
    print(f"[[TN={cm[0,0]} FP={cm[0,1]}], [FN={cm[1,0]} TP={cm[1,1]}]]")

    print("\nClassification Report:")
    report = classification_report(final_targets, final_preds, target_names=["BENIGN", "ATTACK"], digits=4)
    print(report)

    # Esportazione risultati su file JSON
    result_data = {
        "dataset": args.dataset,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "pretrained": args.pretrained,
        "training_quality": {
            "test_accuracy": round(final_acc, 4),
            "macro_f1": round(final_f1, 4),
            "weighted_f1": round(f1_weighted, 4),
            "macro_precision": round(prec_macro, 4),
            "weighted_precision": round(prec_weighted, 4),
            "macro_recall": round(rec_macro, 4),
            "weighted_recall": round(rec_weighted, 4),
            "roc_auc": round(roc_auc, 6),
            "per_class": {
                "BENIGN": {
                    "precision": round(float(prec_per_class[0]), 4),
                    "recall": round(float(rec_per_class[0]), 4),
                    "f1_score": round(float(f1_per_class[0]), 4),
                    "support": int(class_support[0])
                },
                "ATTACK": {
                    "precision": round(float(prec_per_class[1]), 4),
                    "recall": round(float(rec_per_class[1]), 4),
                    "f1_score": round(float(f1_per_class[1]), 4),
                    "support": int(class_support[1])
                }
            }
        },
        "convergence_and_loss_stability": {
            "total_training_time_s": round(total_training_time, 2),
            "best_epoch": int(best_epoch),
            "time_to_best_epoch_s": round(time_to_best_epoch, 2),
            "epochs_to_converge": int(best_epoch),
            "final_train_loss": round(final_train_loss, 4),
            "final_val_loss": round(final_val_loss, 4),
            "train_val_loss_gap": round(loss_gap, 4),
            "loss_stability_std_last_5_epochs": round(loss_stability_std, 6),
            "is_stable": bool(loss_stability_std < 0.01)
        },
        "inference_latency_ms": round(inf_time, 4),
        "confusion_matrix": cm.tolist(),
        "training_history": history
    }
    result_file = os.path.join(results_dir, f"results_{args.dataset}.json")
    with open(result_file, "w") as f:
        json.dump(result_data, f, indent=4)
    print(f"Metriche salvate in: {result_file}\n")


if __name__ == "__main__":
    main()
