"""
visualize.py  —  All visualizations from saved results
========================================================
Run:
    python visualize.py --results_dir results --fold 0
    python visualize.py --results_dir results --all_folds
    python visualize.py --results_dir results --tsne --fold 0

Produces:
    - Per-class accuracy bar chart
    - Top-20 confused pairs chart
    - Confusion matrix
    - t-SNE / UMAP of embeddings
    - Mean loss + accuracy curves across all folds
    - Cross-fold comparison bar chart (best vs final)
"""

import os
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.manifold import TSNE
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from dataset_class import find_h5_files, user_from_path, H5SignDataset
from model import TransformerClassifier, collate_fn_packed
from train import save_confusion_matrix, save_top_confused_pairs, save_per_class_accuracy


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


# ============================================================
# HELPERS
# ============================================================

def load_model(fold, results_dir, checkpoint="final", device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    fold_dir    = os.path.join(results_dir, f"fold_{fold}")
    class_names = list(np.load(os.path.join(results_dir, "class_names.npy"), allow_pickle=True))

    model = TransformerClassifier(
        input_size      = H5SignDataset.FEATURE_DIM,
        num_classes     = len(class_names),
        model_dim       = model_config["model_dim"],
        nhead           = model_config["nhead"],
        num_layers      = model_config["num_layers"],
        dim_feedforward = model_config["dim_feedforward"],
        dropout         = model_config["dropout"],
        drop_path_rate  = model_config["drop_path_rate"],
    ).to(device)

    ckpt = "final_model.pt"
    model.load_state_dict(
        torch.load(os.path.join(fold_dir, ckpt), map_location=device, weights_only=True)
    )
    model.eval()
    return model, class_names, device


def get_test_loader(fold, results_dir, data_root=DATA_ROOT):
    class_names   = list(np.load(os.path.join(results_dir, "class_names.npy"), allow_pickle=True))
    label_encoder = LabelEncoder()
    label_encoder.fit(class_names)

    fold_dir   = os.path.join(results_dir, f"fold_{fold}")
    norm_stats = {
        "mean": np.load(os.path.join(fold_dir, "norm_mean.npy")),
        "std" : np.load(os.path.join(fold_dir, "norm_std.npy")),
    }

    all_paths     = find_h5_files(data_root)
    user_to_paths = defaultdict(list)
    for p in all_paths:
        user_to_paths[user_from_path(p)].append(p)
    users      = sorted(user_to_paths.keys())
    test_paths = user_to_paths[users[fold]]

    test_set = H5SignDataset(
        test_paths, label_encoder,
        n=model_config["max_frames"],
        normalize_stats=norm_stats,
        augment=False,
    )
    loader = DataLoader(
        test_set, batch_size=model_config["batch_size"], shuffle=False,
        collate_fn=collate_fn_packed,
        num_workers=model_config["num_workers"], pin_memory=True,
    )
    return loader, class_names


# ============================================================
# t-SNE Visualization
# ============================================================

@torch.no_grad()
def extract_embeddings(model, loader, device, max_samples=2000):
    embeddings, labels = [], []
    n = 0
    for inputs, lbls, lengths, padding_mask in loader:
        inputs       = inputs.to(device)
        padding_mask = padding_mask.to(device)
        embs = model.get_embeddings(inputs, padding_mask=padding_mask)
        embeddings.append(embs.cpu().numpy())
        labels.append(lbls.numpy())
        n += inputs.size(0)
        if n >= max_samples:
            break
    return np.concatenate(embeddings)[:max_samples], np.concatenate(labels)[:max_samples]


def plot_tsne(fold, results_dir, checkpoint="final", max_samples=2000, perplexity=30):
    print(f"\n  Computing t-SNE for fold {fold}...")
    model, class_names, device = load_model(fold, results_dir, checkpoint)
    loader, _                  = get_test_loader(fold, results_dir)

    embs, labels = extract_embeddings(model, loader, device, max_samples)

    print(f"  Running t-SNE on {len(embs)} samples...")
    tsne  = TSNE(n_components=2, perplexity=perplexity, random_state=42, max_iter=1000)
    embs2 = tsne.fit_transform(embs)

    n_cls  = len(class_names)
    colors = cm.get_cmap("tab20", min(n_cls, 20))

    plt.figure(figsize=(14, 10))
    unique_labels = np.unique(labels)

    for i, lbl in enumerate(unique_labels):
        mask = labels == lbl
        plt.scatter(
            embs2[mask, 0], embs2[mask, 1],
            s=12, alpha=0.6,
            color=colors(i % 20),
            label=class_names[lbl] if n_cls <= 20 else None,
        )

    plt.title(f"t-SNE Embeddings — Fold {fold} (final checkpoint)\n"
              f"{len(embs)} samples, {n_cls} classes")
    plt.xlabel("t-SNE dim 1")
    plt.ylabel("t-SNE dim 2")

    if n_cls <= 20:
        plt.legend(fontsize=7, ncol=2, loc="best")

    plt.tight_layout()
    save_path = os.path.join(results_dir, f"fold_{fold}", f"tsne_final.png")
    plt.savefig(save_path, dpi=120)
    plt.close()
    print(f"   t-SNE saved: {save_path}")


# ============================================================
# Reload & Replot  (from saved y_true / y_pred in CSV)
# ============================================================

def replot_fold(fold, results_dir, checkpoint="final"):
    """
    Regenerates all plots for a fold by re-running inference.
    Useful if you want to tweak plot styles without retraining.
    """
    print(f"\n  Replotting fold {fold}...")
    model, class_names, device = load_model(fold, results_dir, checkpoint)
    loader, _                  = get_test_loader(fold, results_dir)

    from train import collect_preds, top_k_accuracy, print_worst_best_classes

    y_true, y_pred, y_topk, loss = collect_preds(model, loader, device, top_k=5)

    top1 = (y_true == y_pred).mean() * 100.0
    top5 = top_k_accuracy(y_true, y_topk)
    print(f"  Top-1={top1:.2f}%  Top-5={top5:.2f}%  Loss={loss:.4f}")

    out_dir = os.path.join(results_dir, f"fold_{fold}", f"replot_final")
    os.makedirs(out_dir, exist_ok=True)

    cm_mat = __import__("sklearn.metrics", fromlist=["confusion_matrix"]).confusion_matrix(y_true, y_pred)

    save_confusion_matrix(
        cm_mat, class_names,
        os.path.join(out_dir, "confusion_matrix.png"),
        title=f"Confusion Matrix — Fold {fold} (final)",
    )
    save_top_confused_pairs(cm_mat, class_names,
                            os.path.join(out_dir, "top_confused_pairs.png"), n=20)
    rows = save_per_class_accuracy(y_true, y_pred, class_names, out_dir)
    print_worst_best_classes(rows, n=10)

    print(f"   Replot saved: {out_dir}")


# ============================================================
# Cross-Fold Summary Plots  (from kfold_summary.csv)
# ============================================================

def plot_summary(results_dir):
    import csv

    csv_path = os.path.join(results_dir, "kfold_summary.csv")
    if not os.path.exists(csv_path):
        print(f"  [!] {csv_path} not found. Run runner.py first.")
        return

    rows = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["fold"].isdigit():
                rows.append(row)

    folds     = [r["fold"]              for r in rows]
    test_top1 = [float(r["test_top1(%)"]) for r in rows]
    test_top5 = [float(r["test_top5(%)"]) for r in rows]
    macro_f1  = [float(r["macro_f1(%)"])  for r in rows]

    x     = np.arange(len(folds))
    width = 0.5

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for ax, vals, title in [
        (axes[0], test_top1, "Test Top-1 Accuracy (%)"),
        (axes[1], test_top5, "Test Top-5 Accuracy (%)"),
        (axes[2], macro_f1,  "Macro F1 (%)"),
    ]:
        bars = ax.bar(x, vals, width, color="steelblue", edgecolor="black")
        ax.axhline(np.mean(vals), color="crimson", linestyle="--", linewidth=1.5,
                   label=f"Mean={np.mean(vals):.1f}%")

        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                    f"{val:.1f}", ha="center", va="bottom", fontsize=8)

        ax.set_xticks(x)
        ax.set_xticklabels([f"Fold {f}" for f in folds], fontsize=8, rotation=45)
        ax.set_ylim(0, 115)
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)

    plt.suptitle("LOSO K-Fold Summary", fontsize=13)
    plt.tight_layout()
    save_path = os.path.join(results_dir, "summary_bar_chart.png")
    plt.savefig(save_path, dpi=100)
    plt.close()
    print(f"   Summary bar chart: {save_path}")


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str,  default="results")
    parser.add_argument("--fold",        type=int,  default=0)
    parser.add_argument("--checkpoint",  type=str,  default="final")
    parser.add_argument("--tsne",        action="store_true", help="Run t-SNE on test embeddings")
    parser.add_argument("--replot",      action="store_true", help="Replot all fold visualizations")
    parser.add_argument("--summary",     action="store_true", help="Plot cross-fold summary bar chart")
    parser.add_argument("--all_folds",   action="store_true", help="Run selected action on all folds")
    args = parser.parse_args()

    results_dir = args.results_dir
    num_folds   = len([d for d in os.listdir(results_dir) if d.startswith("fold_")])
    folds       = list(range(num_folds)) if args.all_folds else [args.fold]

    if args.tsne:
        for fold in folds:
            plot_tsne(fold, results_dir, args.checkpoint)

    if args.replot:
        for fold in folds:
            replot_fold(fold, results_dir, args.checkpoint)

    if args.summary:
        plot_summary(results_dir)

    if not any([args.tsne, args.replot, args.summary]):
        print("Specify at least one action: --tsne, --replot, --summary")
        print("Example: python visualize.py --tsne --fold 0")
        print("         python visualize.py --replot --all_folds")
        print("         python visualize.py --summary")