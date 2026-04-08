"""
evaluate.py  —  Load a saved fold model and evaluate on the test set
=====================================================================
Run:
    python evaluate.py --fold 0 --results_dir results
    python evaluate.py --fold -1 --results_dir results   # all folds

Computes and saves:
  - Top-1, Top-5 accuracy
  - Macro / Weighted Precision, Recall, F1
  - Confusion matrix
  - Top-20 confused pairs
  - Per-class accuracy CSV + bar chart
  - Full classification report
  - metrics.txt summary
"""

import os
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    confusion_matrix, classification_report,
    precision_score, recall_score, f1_score,
)
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict

from dataset_class import find_h5_files, label_from_filename, user_from_path, H5SignDataset
from model import TransformerClassifier, collate_fn_packed
from train import (
    collect_preds, top_k_accuracy,
    save_confusion_matrix, save_top_confused_pairs,
    save_per_class_accuracy, print_worst_best_classes,
    save_classification_report,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# CONFIG
# ============================================================

DATA_ROOT = "/mnt/9a528fe4-4fe8-4dff-9a0c-8b1a3cf3d7ba/popsign/ISL_Goa_Data_h5_30fps"

model_config = {
    "max_frames"     : 150,
    "model_dim"      : 256,
    "nhead"          : 8,
    "num_layers"     : 4,
    "dim_feedforward": 512,
    "dropout"        : 0.3,
    "drop_path_rate" : 0.1,
    "batch_size"     : 32,
    "num_workers"    : 4,
}


def evaluate_fold(fold, results_dir, data_root=DATA_ROOT):
    fold_dir = os.path.join(results_dir, f"fold_{fold}")
    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{'='*55}")
    print(f"  Evaluating Fold {fold}  |  checkpoint=final")
    print(f"  fold_dir : {fold_dir}")
    print(f"{'='*55}")

    # ── Class names & label encoder ───────────────────────────
    class_names   = list(np.load(os.path.join(results_dir, "class_names.npy"), allow_pickle=True))
    num_classes   = len(class_names)
    label_encoder = LabelEncoder()
    label_encoder.fit(class_names)

    # ── Norm stats ────────────────────────────────────────────
    norm_stats = {
        "mean": np.load(os.path.join(fold_dir, "norm_mean.npy")),
        "std" : np.load(os.path.join(fold_dir, "norm_std.npy")),
    }

    # ── Find test user for this fold ──────────────────────────
    all_paths     = find_h5_files(data_root)
    user_to_paths = defaultdict(list)
    for p in all_paths:
        user_to_paths[user_from_path(p)].append(p)
    users      = sorted(user_to_paths.keys())
    test_user  = users[fold]
    test_paths = user_to_paths[test_user]

    print(f"  Test user : {test_user}  ({len(test_paths)} files)")

    # ── Dataset / loader ──────────────────────────────────────
    test_set = H5SignDataset(
        test_paths, label_encoder,
        n=model_config["max_frames"],
        normalize_stats=norm_stats,
        augment=False,
    )
    test_loader = DataLoader(
        test_set, batch_size=model_config["batch_size"], shuffle=False,
        collate_fn=collate_fn_packed, num_workers=model_config["num_workers"],
        pin_memory=True,
    )

    # ── Model ─────────────────────────────────────────────────
    model = TransformerClassifier(
        input_size      = H5SignDataset.FEATURE_DIM,
        num_classes     = num_classes,
        model_dim       = model_config["model_dim"],
        nhead           = model_config["nhead"],
        num_layers      = model_config["num_layers"],
        dim_feedforward = model_config["dim_feedforward"],
        dropout         = model_config["dropout"],
        drop_path_rate  = model_config["drop_path_rate"],
    ).to(device)

    model.load_state_dict(
        torch.load(os.path.join(fold_dir, "final_model.pt"),
                   map_location=device, weights_only=True)
    )
    print(f"  Loaded: final_model.pt")

    # ── Predictions ───────────────────────────────────────────
    y_true, y_pred, y_topk, test_loss = collect_preds(
        model, test_loader, device, top_k=5
    )

    # ── Metrics ───────────────────────────────────────────────
    top1  = (y_true == y_pred).mean() * 100.0
    top5  = top_k_accuracy(y_true, y_topk)
    mac_p = precision_score(y_true, y_pred, average="macro",    zero_division=0) * 100
    mac_r = recall_score(   y_true, y_pred, average="macro",    zero_division=0) * 100
    mac_f = f1_score(       y_true, y_pred, average="macro",    zero_division=0) * 100
    wtd_f = f1_score(       y_true, y_pred, average="weighted", zero_division=0) * 100

    print(f"\n  Top-1 Accuracy   : {top1:.2f}%")
    print(f"  Top-5 Accuracy   : {top5:.2f}%")
    print(f"  Macro Precision  : {mac_p:.2f}%")
    print(f"  Macro Recall     : {mac_r:.2f}%")
    print(f"  Macro F1         : {mac_f:.2f}%")
    print(f"  Weighted F1      : {wtd_f:.2f}%")
    print(f"  Test Loss        : {test_loss:.4f}")

    # ── Save artifacts ────────────────────────────────────────
    eval_dir = os.path.join(fold_dir, "eval_final")
    os.makedirs(eval_dir, exist_ok=True)

    cm = confusion_matrix(y_true, y_pred)

    save_confusion_matrix(
        cm, class_names,
        os.path.join(eval_dir, "confusion_matrix.png"),
        title=f"Confusion Matrix — Fold {fold} (final epoch)",
    )
    save_top_confused_pairs(
        cm, class_names,
        os.path.join(eval_dir, "top_confused_pairs.png"),
        n=20,
    )
    rows = save_per_class_accuracy(y_true, y_pred, class_names, eval_dir)
    print_worst_best_classes(rows, n=10)

    save_classification_report(
        y_true, y_pred, class_names,
        os.path.join(eval_dir, "classification_report.txt"),
    )

    with open(os.path.join(eval_dir, "metrics.txt"), "w") as f:
        f.write(f"Fold        : {fold}\n")
        f.write(f"Checkpoint  : final\n")
        f.write(f"Test user   : {test_user}\n")
        f.write(f"Top-1       : {top1:.2f}%\n")
        f.write(f"Top-5       : {top5:.2f}%\n")
        f.write(f"Macro P     : {mac_p:.2f}%\n")
        f.write(f"Macro R     : {mac_r:.2f}%\n")
        f.write(f"Macro F1    : {mac_f:.2f}%\n")
        f.write(f"Weighted F1 : {wtd_f:.2f}%\n")
        f.write(f"Test Loss   : {test_loss:.4f}\n")

    print(f"\n  ✅ Eval artifacts saved to: {eval_dir}")

    return {
        "fold"       : fold,
        "test_user"  : test_user,
        "top1_acc"   : top1,
        "top5_acc"   : top5,
        "macro_f1"   : mac_f,
        "weighted_f1": wtd_f,
        "y_true"     : y_true,
        "y_pred"     : y_pred,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold",        type=int, default=-1, help="-1 = all folds")
    parser.add_argument("--results_dir", type=str, default="results")
    args = parser.parse_args()

    results_dir = args.results_dir

    if args.fold == -1:
        all_results = []
        num_folds   = len([d for d in os.listdir(results_dir) if d.startswith("fold_")])
        for fold in range(num_folds):
            r = evaluate_fold(fold, results_dir)
            all_results.append(r)

        print("\n\n" + "="*55)
        print("  ALL FOLDS SUMMARY")
        print("="*55)
        for key, label in [
            ("top1_acc",    "Top-1 Acc"),
            ("top5_acc",    "Top-5 Acc"),
            ("macro_f1",    "Macro F1"),
            ("weighted_f1", "Weighted F1"),
        ]:
            vals = [r[key] for r in all_results]
            print(f"  {label:<14}: {np.mean(vals):.2f}% ± {np.std(vals):.2f}%")
    else:
        evaluate_fold(args.fold, results_dir)