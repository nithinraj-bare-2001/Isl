"""
analysis.py  —  Post-Training Analysis
========================================
Run AFTER runner.py has completed.

Usage:
    python analysis.py

TensorBoard (in a separate terminal):
    tensorboard --logdir=results/runs
"""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix, f1_score

from visualize import load_model, get_test_loader, extract_embeddings, plot_summary
from train import (
    collect_preds, top_k_accuracy,
    save_confusion_matrix, save_top_confused_pairs,
    save_per_class_accuracy, print_worst_best_classes,
)
from inference import SignInference


# ============================================================
# CONFIG  — change these
# ============================================================

RESULTS_DIR  = "results"
DATA_ROOT    = "/mnt/9a528fe4-4fe8-4dff-9a0c-8b1a3cf3d7ba/popsign/ISL_Goa_Data_h5_30fps"
FOLD         = 0        # which fold to analyse
CHECKPOINT   = "final"  # only final_model.pt is saved
MAX_TSNE_SAMPLES = 2000
TSNE_PERPLEXITY  = 30

# For single clip inference test (Section 10)
# Change to any real .h5 path to test
CLIP_PATH = "path/to/your/clip.h5"


# ============================================================
# SECTION 1 — Load Model & Class Names
# ============================================================

print("\n" + "="*55)
print(f"  POST-TRAINING ANALYSIS")
print(f"  Results dir : {RESULTS_DIR}")
print(f"  Fold        : {FOLD}")
print(f"  Checkpoint  : {CHECKPOINT}")
print("="*55)

model, class_names, device = load_model(FOLD, RESULTS_DIR, CHECKPOINT)
loader, _                  = get_test_loader(FOLD, RESULTS_DIR, DATA_ROOT)

print(f"\n  Classes     : {len(class_names)}")
print(f"  Device      : {device}")
print(f"  Test batches: {len(loader)}")


# ============================================================
# SECTION 2 — Run Inference
# ============================================================

print("\n--- Running Inference ---")
y_true, y_pred, y_topk, loss = collect_preds(model, loader, device, top_k=5)

top1     = (y_true == y_pred).mean() * 100.0
top5     = top_k_accuracy(y_true, y_topk)
macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0) * 100

print(f"  Top-1 Accuracy : {top1:.2f}%")
print(f"  Top-5 Accuracy : {top5:.2f}%")
print(f"  Macro F1       : {macro_f1:.2f}%")
print(f"  Test Loss      : {loss:.4f}")


# ============================================================
# SECTION 3 — Confusion Matrix
# ============================================================

print("\n--- Confusion Matrix ---")
out_dir = os.path.join(RESULTS_DIR, f"fold_{FOLD}", "analysis")
os.makedirs(out_dir, exist_ok=True)

cm_mat = confusion_matrix(y_true, y_pred)
save_confusion_matrix(
    cm_mat, class_names,
    os.path.join(out_dir, "confusion_matrix.png"),
    title=f"Confusion Matrix — Fold {FOLD} ({CHECKPOINT})",
)
print(f"   Saved: {out_dir}/confusion_matrix.png")


# ============================================================
# SECTION 4 — Top-20 Confused Pairs
# ============================================================

print("\n--- Top-20 Confused Pairs ---")
save_top_confused_pairs(
    cm_mat, class_names,
    os.path.join(out_dir, "top_confused_pairs.png"),
    n=20,
)
print(f"   Saved: {out_dir}/top_confused_pairs.png")
print(f"   Saved: {out_dir}/top_confused_pairs.csv")


# ============================================================
# SECTION 5 — Per-Class Accuracy Bar Chart
# ============================================================

print("\n--- Per-Class Accuracy ---")
rows = save_per_class_accuracy(y_true, y_pred, class_names, out_dir)
print_worst_best_classes(rows, n=10)
print(f"   Saved: {out_dir}/per_class_accuracy_bar.png")
print(f"   Saved: {out_dir}/per_class_accuracy.csv")


# ============================================================
# SECTION 6 — t-SNE of Embeddings
# ============================================================

print(f"\n--- t-SNE ({MAX_TSNE_SAMPLES} samples, perplexity={TSNE_PERPLEXITY}) ---")
embs, labels = extract_embeddings(model, loader, device, max_samples=MAX_TSNE_SAMPLES)
print(f"  Embeddings shape: {embs.shape}")

print("  Running t-SNE (this may take a minute)...")
tsne  = TSNE(n_components=2, perplexity=TSNE_PERPLEXITY, random_state=42, max_iter=1000)
embs2 = tsne.fit_transform(embs)

n_cls  = len(class_names)
colors = cm.get_cmap("tab20", min(n_cls, 20))

plt.figure(figsize=(14, 10))
for i, lbl in enumerate(np.unique(labels)):
    mask = labels == lbl
    plt.scatter(
        embs2[mask, 0], embs2[mask, 1],
        s=12, alpha=0.6,
        color=colors(i % 20),
        label=class_names[lbl] if n_cls <= 20 else None,
    )

plt.title(f"t-SNE — Fold {FOLD} ({CHECKPOINT}) | {MAX_TSNE_SAMPLES} samples, {n_cls} classes")
plt.xlabel("t-SNE dim 1")
plt.ylabel("t-SNE dim 2")
if n_cls <= 20:
    plt.legend(fontsize=8, ncol=2)
plt.tight_layout()
tsne_path = os.path.join(out_dir, "tsne.png")
plt.savefig(tsne_path, dpi=120)
plt.close()
print(f"   Saved: {tsne_path}")


# ============================================================
# SECTION 7 — Loss & Accuracy Curves
# ============================================================

print("\n--- Loss & Accuracy Curves ---")
loss_curve_path = os.path.join(RESULTS_DIR, f"fold_{FOLD}", "loss_curve.png")
top1_curve_path = os.path.join(RESULTS_DIR, f"fold_{FOLD}", "top1_curve.png")
top5_curve_path = os.path.join(RESULTS_DIR, f"fold_{FOLD}", "top5_curve.png")

for label, path in [
    ("Loss curve    ", loss_curve_path),
    ("Top-1 curve   ", top1_curve_path),
    ("Top-5 curve   ", top5_curve_path),
]:
    if os.path.exists(path):
        print(f"  {label}: {path}")
    else:
        print(f"  [!] Not found : {path}")


# ============================================================
# SECTION 8 — Cross-Fold Summary Bar Chart
# ============================================================

print("\n--- Cross-Fold Summary Bar Chart ---")
plot_summary(RESULTS_DIR)
print(f"  Saved: {RESULTS_DIR}/summary_bar_chart.png")


# ============================================================
# SECTION 9 — Mean Curves Across All Folds
# ============================================================

print("\n--- Mean Curves ---")
for fname in ["mean_loss_curve.png", "mean_top1_curve.png", "mean_top5_curve.png"]:
    path = os.path.join(RESULTS_DIR, fname)
    if os.path.exists(path):
        print(f"  {fname} : {path}")
    else:
        print(f"  [!] Not found : {path} — run curve.py first")


# ============================================================
# SECTION 10 — Single Clip Inference
# ============================================================

print("\n--- Single Clip Inference ---")
if os.path.exists(CLIP_PATH):
    inf = SignInference(fold=FOLD, results_dir=RESULTS_DIR)
    top1_label, confidence, top5_results = inf.predict(CLIP_PATH)

    print(f"  Prediction : {top1_label}  ({confidence*100:.1f}%)")
    print("  Top-5:")
    for label, prob in top5_results:
        print(f"    {label:<30s} {prob*100:.2f}%")
else:
    print(f"  [!] Clip not found: {CLIP_PATH}")
    print(f"      Set CLIP_PATH at the top of this script to test inference.")


# ============================================================
# DONE
# ============================================================

print("\n" + "="*55)
print(f"  ANALYSIS COMPLETE")
print(f"  All outputs saved to: {out_dir}")
print(f"  TensorBoard: tensorboard --logdir={RESULTS_DIR}/runs")
print("="*55)