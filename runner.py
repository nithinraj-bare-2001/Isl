"""
runner.py  —  LOSO K-Fold  (N-1 train / 1 test, no validation, no early stopping)
===================================================================================
Run:
    python runner.py

TensorBoard (in separate terminal):
    tensorboard --logdir=results/runs

Bar charts saved as separate images:
    kfold_bar_train_top1.png
    kfold_bar_train_top5.png
    kfold_bar_test_top1.png
    kfold_bar_test_top5.png
    kfold_bar_macro_f1.png
"""

import os
import sys
import random
import numpy as np
import torch
import csv
from datetime import datetime
from collections import defaultdict

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix

from dataset_class import find_h5_files, label_from_filename, user_from_path
from train import train_one_fold, Tee, save_confusion_matrix

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# CONFIG
# ============================================================

DATA_ROOT   = "/mnt/9a528fe4-4fe8-4dff-9a0c-8b1a3cf3d7ba/popsign/ISL_Goa_Data_h5_30fps"
RESULTS_DIR = "results"

config = {
    "max_frames"      : 150,
    "batch_size"      : 32,
    "epochs"          : 100,
    "warmup_epochs"   : 5,
    "min_lr_ratio"    : 0.05,
    "lr"              : 1e-4,
    "weight_decay"    : 1e-3,
    "label_smoothing" : 0.1,
    "num_workers"     : 4,
    "use_mixup"       : True,
    "mixup_alpha"     : 0.2,
    "drop_path_rate"  : 0.1,
    "model_dim"       : 256,
    "nhead"           : 8,
    "num_layers"      : 4,
    "dim_feedforward" : 512,
    "dropout"         : 0.3,
    "seed"            : 42,
}


# ============================================================
# SETUP
# ============================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(config["seed"])
os.makedirs(RESULTS_DIR, exist_ok=True)

ts    = datetime.now().strftime("%Y%m%d_%H%M%S")
log_f = open(os.path.join(RESULTS_DIR, f"kfold_summary_{ts}.log"), "w")
sys.stdout = Tee(sys.stdout, log_f)
sys.stderr = Tee(sys.stderr, log_f)

print("=" * 60)
print(f"  LOSO K-Fold Training  |  {ts}")
print(f"  DATA_ROOT   : {DATA_ROOT}")
print(f"  RESULTS_DIR : {RESULTS_DIR}")
print(f"  TensorBoard : tensorboard --logdir={RESULTS_DIR}/runs")
print("=" * 60)

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print(f"\n  Device: {device}")
if device.type == "cuda":
    print(f"  GPU   : {torch.cuda.get_device_name(1)}")


# ============================================================
# DISCOVER FILES
# ============================================================

all_paths = find_h5_files(DATA_ROOT)
if not all_paths:
    raise RuntimeError(f"No .h5 files found under: {DATA_ROOT}")

print(f"\n  Total .h5 files : {len(all_paths)}")

user_to_paths = defaultdict(list)
for p in all_paths:
    user_to_paths[user_from_path(p)].append(p)

users     = sorted(user_to_paths.keys())
num_folds = len(users)

print(f"  Users found ({num_folds}): {users}")
for u in users:
    print(f"    {u}: {len(user_to_paths[u])} files")


# ============================================================
# GLOBAL LABEL ENCODER
# ============================================================

all_labels  = [label_from_filename(p) for p in all_paths]
all_classes = sorted(set(all_labels))
num_classes = len(all_classes)

label_encoder = LabelEncoder()
label_encoder.fit(all_classes)

np.save(os.path.join(RESULTS_DIR, "class_names.npy"), np.array(label_encoder.classes_))

print(f"\n  Total word classes : {num_classes}")
print(f"  Classes (first 10) : {all_classes[:10]}{'...' if num_classes > 10 else ''}")


# ============================================================
# FOLD ASSIGNMENTS
# ============================================================

print(f"\n  {'Fold':<6} {'Test':<12} Train")
print(f"  {'-'*50}")

fold_assignments = []
for i, test_user in enumerate(users):
    train_users = [u for u in users if u != test_user]
    fold_assignments.append({"test_user": test_user, "train_users": train_users})
    print(f"  {i:<6} {test_user:<12} {train_users}")

print()


# ============================================================
# LOSO LOOP
# ============================================================

fold_results = []

for fold_idx, assignment in enumerate(fold_assignments):
    test_user   = assignment["test_user"]
    train_users = assignment["train_users"]

    train_paths  = [p for u in train_users for p in user_to_paths[u]]
    test_paths   = user_to_paths[test_user]
    fold_log_dir = os.path.join(RESULTS_DIR, f"fold_{fold_idx}")

    result = train_one_fold(
        fold_idx      = fold_idx,
        train_paths   = train_paths,
        test_paths    = test_paths,
        label_encoder = label_encoder,
        class_names   = list(label_encoder.classes_),
        fold_log_dir  = fold_log_dir,
        device        = device,
        config        = config,
        tb_root       = os.path.join(RESULTS_DIR, "runs"),
    )

    result["test_user"]   = test_user
    result["train_users"] = train_users
    fold_results.append(result)


# ============================================================
# AGGREGATE RESULTS
# ============================================================

print("\n\n" + "=" * 70)
print("  FINAL K-FOLD SUMMARY")
print("=" * 70)

print(f"\n  {'Fold':<6} {'Test':<12} {'TrTop1':>7} {'TrTop5':>7} "
      f"{'TeTop1':>7} {'TeTop5':>7} {'MacroF1':>8} {'WtdF1':>7}")
print(f"  {'-'*68}")

for r in fold_results:
    print(
        f"  {r['fold']:<6} {r['test_user']:<12} "
        f"{r['train_top1_acc']:6.2f}%  "
        f"{r['train_top5_acc']:6.2f}%  "
        f"{r['top1_acc']:6.2f}%  "
        f"{r['top5_acc']:6.2f}%  "
        f"{r['macro_f1']:7.2f}%  "
        f"{r['weighted_f1']:6.2f}%"
    )

print(f"  {'-'*68}")

metric_keys = [
    ("train_top1_acc", "Train Top-1 Acc"),
    ("train_top5_acc", "Train Top-5 Acc"),
    ("top1_acc",       "Test  Top-1 Acc"),
    ("top5_acc",       "Test  Top-5 Acc"),
    ("macro_f1",       "Macro F1"),
    ("weighted_f1",    "Weighted F1"),
]

print()
for key, label in metric_keys:
    vals = [r[key] for r in fold_results]
    print(f"  {label:<22}: {np.mean(vals):.2f}% ± {np.std(vals):.2f}%")


# ── Summary CSV ───────────────────────────────────────────────
csv_path = os.path.join(RESULTS_DIR, "kfold_summary.csv")
with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "fold", "test_user", "train_users",
        "train_top1(%)", "train_top5(%)",
        "test_top1(%)", "test_top5(%)",
        "macro_f1(%)", "weighted_f1(%)", "test_loss",
    ])
    for r in fold_results:
        writer.writerow([
            r["fold"], r["test_user"], "+".join(r["train_users"]),
            f"{r['train_top1_acc']:.2f}",
            f"{r['train_top5_acc']:.2f}",
            f"{r['top1_acc']:.2f}",
            f"{r['top5_acc']:.2f}",
            f"{r['macro_f1']:.2f}",
            f"{r['weighted_f1']:.2f}",
            f"{r['test_loss']:.4f}",
        ])
    for stat, fn in [("MEAN", np.mean), ("STD", np.std)]:
        writer.writerow(
            [stat, "—", "—"] +
            [f"{fn([r[k] for r in fold_results]):.2f}" for k, _ in metric_keys] +
            ["—"]
        )

print(f"\n  Summary CSV : {csv_path}")


# ── Aggregated Confusion Matrix ────────────────────────────────
all_y_true = np.concatenate([r["y_true"] for r in fold_results])
all_y_pred = np.concatenate([r["y_pred"] for r in fold_results])

agg_cm = confusion_matrix(all_y_true, all_y_pred)
save_confusion_matrix(
    agg_cm,
    list(label_encoder.classes_),
    os.path.join(RESULTS_DIR, "confusion_matrix_aggregated.png"),
    title="Aggregated Confusion Matrix (All Folds)",
)
print("  Aggregated confusion matrix saved.")


# ── Mean Curves across folds ───────────────────────────────────
max_ep = max(len(r["train_losses"]) for r in fold_results)

def pad_to(lst, length):
    return lst + [lst[-1]] * (length - len(lst))

mean_train_loss  = np.mean([pad_to(r["train_losses"],  max_ep) for r in fold_results], axis=0)
mean_test_loss   = np.mean([pad_to(r["test_losses"],   max_ep) for r in fold_results], axis=0)
mean_train_top1  = np.mean([pad_to(r["train_top1s"],   max_ep) for r in fold_results], axis=0)
mean_test_top1   = np.mean([pad_to(r["test_top1s"],    max_ep) for r in fold_results], axis=0)
mean_train_top5  = np.mean([pad_to(r["train_top5s"],   max_ep) for r in fold_results], axis=0)
mean_test_top5   = np.mean([pad_to(r["test_top5s"],    max_ep) for r in fold_results], axis=0)

# Loss curves
fig, ax = plt.subplots(figsize=(10, 5))
for r in fold_results:
    ep = list(range(len(r["train_losses"])))
    ax.plot(ep, r["train_losses"], alpha=0.2, color="steelblue")
    ax.plot(ep, r["test_losses"],  alpha=0.2, color="coral")
ax.plot(mean_train_loss, color="steelblue", linewidth=2, label="Mean Train Loss")
ax.plot(mean_test_loss,  color="coral",     linewidth=2, label="Mean Test Loss")
ax.set_title("Mean Loss — All Folds")
ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "mean_loss_curve.png"), dpi=100)
plt.close()
print("  Mean loss curve saved.")

# Top-1 accuracy curves
fig, ax = plt.subplots(figsize=(10, 5))
for r in fold_results:
    ep = list(range(len(r["train_top1s"])))
    ax.plot(ep, r["train_top1s"], alpha=0.2, color="steelblue")
    ax.plot(ep, r["test_top1s"],  alpha=0.2, color="coral")
ax.plot(mean_train_top1, color="steelblue", linewidth=2, label="Mean Train Top-1")
ax.plot(mean_test_top1,  color="coral",     linewidth=2, label="Mean Test Top-1")
ax.set_title("Mean Top-1 Accuracy — All Folds")
ax.set_xlabel("Epoch"); ax.set_ylabel("Top-1 Accuracy (%)")
ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "mean_top1_curve.png"), dpi=100)
plt.close()
print("  Mean Top-1 curve saved.")

# Top-5 accuracy curves
fig, ax = plt.subplots(figsize=(10, 5))
for r in fold_results:
    ep = list(range(len(r["train_top5s"])))
    ax.plot(ep, r["train_top5s"], alpha=0.2, color="steelblue")
    ax.plot(ep, r["test_top5s"],  alpha=0.2, color="coral")
ax.plot(mean_train_top5, color="steelblue", linewidth=2, label="Mean Train Top-5")
ax.plot(mean_test_top5,  color="coral",     linewidth=2, label="Mean Test Top-5")
ax.set_title("Mean Top-5 Accuracy — All Folds")
ax.set_xlabel("Epoch"); ax.set_ylabel("Top-5 Accuracy (%)")
ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "mean_top5_curve.png"), dpi=100)
plt.close()
print("  Mean Top-5 curve saved.")


# ── 5 Separate Bar Charts ──────────────────────────────────────

x     = np.arange(num_folds)
width = 0.5
xlbls = [f"Fold {r['fold']}\n({r['test_user']})" for r in fold_results]

def save_bar_chart(key, ylabel, title, filename, color="steelblue"):
    vals = [r[key] for r in fold_results]
    mean_val = np.mean(vals)

    fig, ax = plt.subplots(figsize=(max(8, num_folds * 1.2), 6))
    bars = ax.bar(x, vals, width, color=color, edgecolor="black")

    ax.axhline(mean_val, color="crimson", linestyle="--", linewidth=1.5,
               label=f"Mean = {mean_val:.2f}%")

    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.4,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(xlbls, fontsize=9)
    ax.set_ylim(0, 115)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, filename), dpi=100)
    plt.close()
    print(f"  Bar chart saved: {filename}")

save_bar_chart("train_top1_acc", "Accuracy (%)", "Train Top-1 Accuracy per Fold",
               "kfold_bar_train_top1.png", color="steelblue")

save_bar_chart("train_top5_acc", "Accuracy (%)", "Train Top-5 Accuracy per Fold",
               "kfold_bar_train_top5.png", color="steelblue")

save_bar_chart("top1_acc",       "Accuracy (%)", "Test Top-1 Accuracy per Fold",
               "kfold_bar_test_top1.png",  color="coral")

save_bar_chart("top5_acc",       "Accuracy (%)", "Test Top-5 Accuracy per Fold",
               "kfold_bar_test_top5.png",  color="coral")

save_bar_chart("macro_f1",       "Macro F1 (%)", "Test Macro F1 per Fold",
               "kfold_bar_macro_f1.png",   color="mediumseagreen")


print(f"\n  ALL DONE. Results in: {RESULTS_DIR}")
print(f"  TensorBoard: tensorboard --logdir={RESULTS_DIR}/runs")
