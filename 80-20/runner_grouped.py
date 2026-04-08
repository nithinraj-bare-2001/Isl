import os
import sys
import random
import numpy as np
import torch
import csv
from datetime import datetime
from collections import defaultdict
import json

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

DATA_ROOT   = "/mnt/9a528fe4-4fe8-4dff-9a0c-8b1a3cf3d7ba/popsign/ISL_Goa_Data_h5_5fps"
RESULTS_DIR = "results_grouped"

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
    "num_folds"       : 5,
}


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(config["seed"])


# ============================================================
# LOGGING
# ============================================================

os.makedirs(RESULTS_DIR, exist_ok=True)

ts    = datetime.now().strftime("%Y%m%d_%H%M%S")
log_f = open(os.path.join(RESULTS_DIR, f"grouped_kfold_{ts}.log"), "w")
sys.stdout = Tee(sys.stdout, log_f)
sys.stderr = Tee(sys.stderr, log_f)


print("=" * 70)
print(f"  GROUPED 5-FOLD (80/20 per user-word)  |  {ts}")
print(f"  DATA_ROOT   : {DATA_ROOT}")
print(f"  RESULTS_DIR : {RESULTS_DIR}")
print(f"  Seed        : {config['seed']}")
print("=" * 70)


device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print(f"\n  Device: {device}")
if device.type == "cuda":
    print(f"  GPU   : {torch.cuda.get_device_name(1)}")


# ============================================================
# LOAD FILES
# ============================================================

all_paths = find_h5_files(DATA_ROOT)
if not all_paths:
    raise RuntimeError(f"No .h5 files found under: {DATA_ROOT}")

print(f"\n  Total files: {len(all_paths)}")


# ============================================================
# GROUP BY (user, word)
# ============================================================

groups = defaultdict(list)

for p in all_paths:
    user  = user_from_path(p)
    label = label_from_filename(p)
    groups[(user, label)].append(p)

print(f"  Total groups (user, word): {len(groups)}")


# ============================================================
# BUILD GLOBAL LABEL ENCODER
# ============================================================

all_labels  = [label_from_filename(p) for p in all_paths]
all_classes = sorted(set(all_labels))

label_encoder = LabelEncoder()
label_encoder.fit(all_classes)

np.save(os.path.join(RESULTS_DIR, "class_names.npy"), np.array(label_encoder.classes_))

print(f"  Total classes: {len(all_classes)}")


# ============================================================
# CREATE GROUPED FOLDS
# ============================================================

num_folds = config["num_folds"]

fold_train_paths = [[] for _ in range(num_folds)]
fold_test_paths  = [[] for _ in range(num_folds)]

rng = np.random.RandomState(config["seed"])

for (user, label), paths in groups.items():

    paths = sorted(paths)
    rng.shuffle(paths)

    N = len(paths)

    if N == 1:
        for f in range(num_folds):
            fold_train_paths[f].append(paths[0])
        continue

    chunks = np.array_split(paths, num_folds)

    for f in range(num_folds):
        test_chunk = chunks[f]

        if len(test_chunk) == 0:
            fold_train_paths[f].extend(paths)
            continue

        train_chunks = [chunks[i] for i in range(num_folds) if i != f]
        train_chunk = np.concatenate(train_chunks) if train_chunks else []

        fold_test_paths[f].extend(test_chunk)
        fold_train_paths[f].extend(train_chunk)


# ============================================================
# SANITY CHECK
# ============================================================

edge_cases   = sum(1 for v in groups.values() if len(v) == 1)
sparse_cases = sum(1 for v in groups.values() if 1 < len(v) < num_folds)
print(f"\n  Edge cases  (N=1, train-only)          : {edge_cases}")
print(f"  Sparse cases (1 < N < {num_folds}, partial test) : {sparse_cases}")

for f in range(num_folds):
    overlap = set(fold_train_paths[f]) & set(fold_test_paths[f])
    assert len(overlap) == 0, f"Fold {f}: overlap detected"
print("  Overlap check passed for all folds.")

print("\n  Fold Statistics:")
for f in range(num_folds):
    print(f"    Fold {f}: Train={len(fold_train_paths[f])}, Test={len(fold_test_paths[f])}")


# ============================================================
# SAVE FOLD SPLITS
# ============================================================

splits_path = os.path.join(RESULTS_DIR, "fold_splits.json")

splits = {
    f"fold_{f}": {
        "train": fold_train_paths[f],
        "test" : fold_test_paths[f]
    }
    for f in range(num_folds)
}

with open(splits_path, "w") as f:
    json.dump(splits, f, indent=2)

print(f"  Fold splits saved : {splits_path}")


# ============================================================
# TRAINING LOOP
# ============================================================

fold_results = []

for fold_idx in range(num_folds):

    print("\n" + "="*60)
    print(f"  RUNNING FOLD {fold_idx}")
    print("="*60)

    train_paths = fold_train_paths[fold_idx]
    test_paths  = fold_test_paths[fold_idx]

    fold_dir = os.path.join(RESULTS_DIR, f"fold_{fold_idx}")

    result = train_one_fold(
        fold_idx      = fold_idx,
        train_paths   = train_paths,
        test_paths    = test_paths,
        label_encoder = label_encoder,
        class_names   = list(label_encoder.classes_),
        fold_log_dir  = fold_dir,
        device        = device,
        config        = config,
        tb_root       = os.path.join(RESULTS_DIR, "runs"),
    )

    fold_results.append(result)


    # ============================================================
    # PER-USER ACCURACY 
    # ============================================================

    user_y_true = defaultdict(list)
    user_y_pred = defaultdict(list)

    for path, yt, yp in zip(test_paths, result["y_true"], result["y_pred"]):
        user = user_from_path(path)
        user_y_true[user].append(yt)
        user_y_pred[user].append(yp)

    user_acc_path = os.path.join(fold_dir, "per_user_accuracy.csv")

    with open(user_acc_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["user", "n_samples", "top1_accuracy(%)"])

        for user in sorted(user_y_true.keys()):
            yt = np.array(user_y_true[user])
            yp = np.array(user_y_pred[user])
            acc = (yt == yp).mean() * 100.0
            writer.writerow([user, len(yt), f"{acc:.2f}"])

    print(f"  Per-user accuracy saved: {user_acc_path}")


# ============================================================
# SUMMARY
# ============================================================

print("\n\n" + "=" * 70)
print("  FINAL GROUPED K-FOLD SUMMARY")
print("=" * 70)

for r in fold_results:
    print(
        f"Fold {r['fold']} | "
        f"Top1={r['top1_acc']:.2f}% | "
        f"Top5={r['top5_acc']:.2f}% | "
        f"MacroF1={r['macro_f1']:.2f}%"
    )


# ============================================================
# SAVE CSV
# ============================================================

csv_path = os.path.join(RESULTS_DIR, "kfold_summary.csv")

with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)

    writer.writerow([
        "fold", "train_top1", "train_top5",
        "test_top1", "test_top5",
        "macro_f1", "weighted_f1", "test_loss"
    ])

    for r in fold_results:
        writer.writerow([
            r["fold"],
            f"{r['train_top1_acc']:.2f}",
            f"{r['train_top5_acc']:.2f}",
            f"{r['top1_acc']:.2f}",
            f"{r['top5_acc']:.2f}",
            f"{r['macro_f1']:.2f}",
            f"{r['weighted_f1']:.2f}",
            f"{r['test_loss']:.4f}",
        ])

print(f"\n  Summary saved: {csv_path}")


# ============================================================
# AGGREGATED CONFUSION MATRIX
# ============================================================

all_y_true = np.concatenate([r["y_true"] for r in fold_results])
all_y_pred = np.concatenate([r["y_pred"] for r in fold_results])

cm = confusion_matrix(all_y_true, all_y_pred)

save_confusion_matrix(
    cm,
    list(label_encoder.classes_),
    os.path.join(RESULTS_DIR, "confusion_matrix_aggregated.png"),
    title="Grouped K-Fold Confusion Matrix",
)

print("  Aggregated confusion matrix saved.")
