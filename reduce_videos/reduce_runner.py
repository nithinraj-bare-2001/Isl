"""
rq2_runner.py  —  RQ2: How many training videos are needed?
=============================================================
Progressively removes 1 video per user per round and evaluates
accuracy at each step under LOPO cross-validation.

Round structure:
    Round 1 : 140 train videos/word  (14 train users × 10 reps)
    Round 2 : 126 train videos/word  (14 train users × 9  reps)
    Round 3 : 112 train videos/word  (14 train users × 8  reps)
    ...
    Round 10:  14 train videos/word  (14 train users × 1  rep)

Note: "train videos per word" = 14 training users × reps.
      The 15th user is the test user (held out, full 10 reps always).

Output structure:
    results_rq2/
        140_reps/
            fold_0/   fold_1/ ... fold_14/
                final_model.pt
                norm_mean.npy, norm_std.npy
                eval/
                    confusion_matrix.png
                    top_confused_pairs.png / .csv
                    per_class_accuracy.csv / _bar.png
                    classification_report.txt
                    metrics.txt
            round_summary.csv
        126_reps/
            ...
        112_reps/ ... 14_reps/
        rq2_summary.csv
        rq2_accuracy_curve.png
        test_coverage.csv
        class_names.npy
        rq2_log_TIMESTAMP.txt

Run:
    python rq2_runner.py
    python rq2_runner.py --reps_start 10 --reps_end 1 --results_dir results_rq2
"""

import os
import sys
import csv
import random
import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from datetime import datetime
from collections import defaultdict
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.metrics import precision_score, recall_score
from torch.utils.data import DataLoader

from dataset_class import find_h5_files, label_from_filename, user_from_path
from train import (
    train_one_fold, Tee,
    collect_preds, top_k_accuracy,
    save_confusion_matrix, save_top_confused_pairs,
    save_per_class_accuracy, print_worst_best_classes,
    save_classification_report,
)


DATA_ROOT   = "/mnt/9a528fe4-4fe8-4dff-9a0c-8b1a3cf3d7ba/popsign/ISL_Goa_Data_h5_30fps"
RESULTS_DIR = "results_rq2"

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

REPS_START = 10   # start from full 10 reps per user
REPS_END   = 1    # go down to 1 rep per user


# ============================================================
# FILE GROUPING
# ============================================================

def group_paths_by_user_and_word(paths):
    """
    Returns dict[user][word] -> list of h5 paths.
    Uses label_from_filename and user_from_path from dataset_class.
    """
    user_word_paths = defaultdict(lambda: defaultdict(list))
    for p in paths:
        user = user_from_path(p)
        word = label_from_filename(p)
        user_word_paths[user][word].append(p)
    return user_word_paths


# ============================================================
# SUBSAMPLING  — deterministic per (round, fold)
# ============================================================

def sample_train_paths(train_users, user_word_paths, reps_per_user,
                       round_idx, fold_idx, base_seed):
    """
    For each training user, randomly select exactly `reps_per_user`
    videos per word. Selection is deterministic given the seed.

    Seed formula:  base_seed + round_idx * 1000 + fold_idx

    This ensures:
      - Same (round, fold) always picks the same videos  → reproducible
      - Different (round, fold) pairs get independent picks → fair
      - Round 2 picks are NOT influenced by what round 1 picked

    Parameters
    ----------
    train_users     : list[str]  — user IDs to include in training
    user_word_paths : dict       — user → word → list[path]
    reps_per_user   : int        — how many reps per word per user to keep
    round_idx       : int        — current round index (0-based)
    fold_idx        : int        — current fold index (0-based)
    base_seed       : int        — base random seed from config

    Returns
    -------
    selected : list[str]  — flat list of selected h5 file paths
    """
    seed = base_seed + round_idx * 1000 + fold_idx
    rng  = np.random.default_rng(seed=seed)

    selected = []
    for user in sorted(train_users):            # sorted for determinism
        for word in sorted(user_word_paths[user].keys()):
            word_paths = user_word_paths[user][word]
            if len(word_paths) <= reps_per_user:
                # Fewer videos than requested — keep all
                selected.extend(word_paths)
            else:
                # Randomly select reps_per_user without replacement
                chosen = rng.choice(
                    word_paths, size=reps_per_user, replace=False
                )
                selected.extend(chosen.tolist())

    return selected


# ============================================================
# AUTO-EVALUATE  — runs after each fold trains
# ============================================================

def evaluate_fold(fold_idx, fold_log_dir, test_paths,
                  label_encoder, class_names, device, config):
    """
    Load the saved final_model.pt and run full evaluation.
    Saves all artifacts into fold_log_dir/eval/

    Called automatically after each fold trains — no need to
    run evaluate.py separately.
    """
    from dataset_class import H5SignDataset
    from model import TransformerClassifier, collate_fn_packed

    eval_dir = os.path.join(fold_log_dir, "eval")
    os.makedirs(eval_dir, exist_ok=True)

    # Load normalization stats saved during training
    norm_stats = {
        "mean": np.load(os.path.join(fold_log_dir, "norm_mean.npy")),
        "std" : np.load(os.path.join(fold_log_dir, "norm_std.npy")),
    }

    # Build test loader
    test_set = H5SignDataset(
        test_paths, label_encoder,
        n=config["max_frames"],
        normalize_stats=norm_stats,
        augment=False,
    )
    test_loader = DataLoader(
        test_set,
        batch_size   = config["batch_size"],
        shuffle      = False,
        collate_fn   = collate_fn_packed,
        num_workers  = config["num_workers"],
        pin_memory   = True,
    )

    # Load model
    num_classes = len(class_names)
    model = TransformerClassifier(
        input_size      = H5SignDataset.FEATURE_DIM,
        num_classes     = num_classes,
        model_dim       = config["model_dim"],
        nhead           = config["nhead"],
        num_layers      = config["num_layers"],
        dim_feedforward = config["dim_feedforward"],
        dropout         = config["dropout"],
        drop_path_rate  = config.get("drop_path_rate", 0.1),
    ).to(device)

    model.load_state_dict(
        torch.load(
            os.path.join(fold_log_dir, "final_model.pt"),
            map_location=device, weights_only=True
        )
    )
    model.eval()

    # Run predictions
    y_true, y_pred, y_topk, test_loss = collect_preds(
        model, test_loader, device, top_k=5
    )

    # Compute all metrics
    top1  = (y_true == y_pred).mean() * 100.0
    top5  = top_k_accuracy(y_true, y_topk)
    mac_p = precision_score(y_true, y_pred, average="macro",    zero_division=0) * 100
    mac_r = recall_score(   y_true, y_pred, average="macro",    zero_division=0) * 100
    mac_f = f1_score(       y_true, y_pred, average="macro",    zero_division=0) * 100
    wtd_f = f1_score(       y_true, y_pred, average="weighted", zero_division=0) * 100

    # Save confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    save_confusion_matrix(
        cm, class_names,
        os.path.join(eval_dir, "confusion_matrix.png"),
        title=f"Confusion Matrix — Fold {fold_idx} (final epoch)",
    )

    # Save top-20 confused pairs
    save_top_confused_pairs(
        cm, class_names,
        os.path.join(eval_dir, "top_confused_pairs.png"),
        n=20,
    )

    # Save per-class accuracy bar chart + CSV
    rows = save_per_class_accuracy(y_true, y_pred, class_names, eval_dir)
    print_worst_best_classes(rows, n=5)

    # Save full classification report
    save_classification_report(
        y_true, y_pred, class_names,
        os.path.join(eval_dir, "classification_report.txt"),
    )

    # Save metrics summary text
    with open(os.path.join(eval_dir, "metrics.txt"), "w") as f:
        f.write(f"Fold        : {fold_idx}\n")
        f.write(f"Top-1       : {top1:.2f}%\n")
        f.write(f"Top-5       : {top5:.2f}%\n")
        f.write(f"Macro P     : {mac_p:.2f}%\n")
        f.write(f"Macro R     : {mac_r:.2f}%\n")
        f.write(f"Macro F1    : {mac_f:.2f}%\n")
        f.write(f"Weighted F1 : {wtd_f:.2f}%\n")
        f.write(f"Test Loss   : {test_loss:.4f}\n")

    print(f"    Eval saved → {eval_dir}")

    return {
        "top1_acc"   : top1,
        "top5_acc"   : top5,
        "macro_f1"   : mac_f,
        "weighted_f1": wtd_f,
        "test_loss"  : test_loss,
        "y_true"     : y_true,
        "y_pred"     : y_pred,
    }


# ============================================================
# PLOTTING
# ============================================================

def plot_rq2_curve(round_summaries, save_path):
    """
    Accuracy vs training videos per word.
    X-axis : videos per word (most data on left → least on right)
    Y-axis : accuracy / F1 (%)
    Shaded band around Top-1 shows ± 1 standard deviation across folds.
    """
    vids  = [s["actual_mean_vids_per_word"] for s in round_summaries]
    top1s = [s["mean_top1"]             for s in round_summaries]
    top5s = [s["mean_top5"]             for s in round_summaries]
    f1s   = [s["mean_macro_f1"]         for s in round_summaries]
    stds  = [s["std_top1"]              for s in round_summaries]

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(vids, top1s, marker="o", color="steelblue",
            linewidth=2.5, markersize=8, label="Top-1 Accuracy")
    ax.fill_between(
        vids,
        [t - s for t, s in zip(top1s, stds)],
        [t + s for t, s in zip(top1s, stds)],
        color="steelblue", alpha=0.15, label="Top-1 ± 1 std"
    )
    ax.plot(vids, top5s, marker="s", color="coral",
            linewidth=2.5, markersize=8, label="Top-5 Accuracy")
    ax.plot(vids, f1s,   marker="^", color="mediumseagreen",
            linewidth=2.5, markersize=8, label="Macro F1")

    for x, y in zip(vids, top1s):
        ax.annotate(
            f"{y:.1f}%", (x, y),
            textcoords="offset points", xytext=(0, 11),
            ha="center", fontsize=9, color="steelblue", fontweight="bold"
        )

    ax.set_xlabel("Mean Actual Training Videos per Word  (averaged across folds)", fontsize=12)
    ax.set_ylabel("Accuracy / F1 (%)", fontsize=12)
    ax.set_title(
        "RQ2 — Effect of Training Data Size on LOPO Accuracy\n"
        "(Test set fixed: held-out user, full 10 reps per word)",
        fontsize=13
    )
    ax.set_xticks(vids)
    ax.set_xticklabels([str(v) for v in vids], fontsize=10)
    ax.set_ylim(0, 108)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()
    print(f"  RQ2 curve saved: {save_path}")


# ============================================================
# SEED
# ============================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# MAIN
# ============================================================

def main(args):
    set_seed(config["seed"])
    os.makedirs(args.results_dir, exist_ok=True)

    # ── Logging ───────────────────────────────────────────────
    ts    = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_f = open(os.path.join(args.results_dir, f"rq2_log_{ts}.txt"), "w")
    sys.stdout = Tee(sys.stdout, log_f)
    sys.stderr = Tee(sys.stderr, log_f)

    print("=" * 65)
    print(f"  RQ2 — Training Data Efficiency  |  {ts}")
    print(f"  DATA_ROOT   : {args.data_root}")
    print(f"  RESULTS_DIR : {args.results_dir}")
    print(f"  Reps sweep  : {args.reps_start} → {args.reps_end}")
    print("=" * 65)

    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    print(f"\n  Device : {device}")
    if device.type == "cuda":
        print(f"  GPU    : {torch.cuda.get_device_name(1)}")

    # ── Discover & group files ────────────────────────────────
    all_paths = find_h5_files(args.data_root)
    if not all_paths:
        raise RuntimeError(f"No .h5 files found under: {args.data_root}")
    print(f"\n  Total .h5 files : {len(all_paths)}")

    user_word_paths = group_paths_by_user_and_word(all_paths)
    users           = sorted(user_word_paths.keys())
    num_folds       = len(users)
    print(f"  Users ({num_folds})      : {users}")

    # Sanity check: print reps per user
    print("\n  Reps per user per word (sanity check):")
    for u in users:
        words      = sorted(user_word_paths[u].keys())
        rep_counts = [len(user_word_paths[u][w]) for w in words]
        print(f"    {u} : {len(words)} words | "
              f"min_reps={min(rep_counts)} max_reps={max(rep_counts)}")

    # ── Label encoder ─────────────────────────────────────────
    all_labels  = [label_from_filename(p) for p in all_paths]
    all_classes = sorted(set(all_labels))
    num_classes = len(all_classes)

    label_encoder = LabelEncoder()
    label_encoder.fit(all_classes)
    np.save(
        os.path.join(args.results_dir, "class_names.npy"),
        np.array(label_encoder.classes_)
    )
    print(f"\n  Word classes : {num_classes}")

    # ── LOPO fold assignments ─────────────────────────────────
    fold_assignments = []
    for i, test_user in enumerate(users):
        train_users = [u for u in users if u != test_user]
        fold_assignments.append({
            "fold_idx"   : i,
            "test_user"  : test_user,
            "train_users": train_users,
        })

    print(f"\n  {'Fold':<6} {'Test User':<14} Train Users")
    print(f"  {'-'*55}")
    for fa in fold_assignments:
        print(f"  {fa['fold_idx']:<6} {fa['test_user']:<14} "
              f"{fa['train_users']}")

    # ── Fix test paths — identical every round, never subsampled ──
    # Patch 2: missing words computed once here, stored per fold,
    # logged to console and saved to test_coverage.csv
    test_paths_per_fold    = {}
    missing_words_per_fold = {}

    for fa in fold_assignments:
        test_paths_per_fold[fa["fold_idx"]] = [
            p
            for word_paths in user_word_paths[fa["test_user"]].values()
            for p in word_paths
        ]

        test_labels   = set(
            label_from_filename(p)
            for p in test_paths_per_fold[fa["fold_idx"]]
        )
        missing_words = sorted(set(all_classes) - test_labels)
        missing_words_per_fold[fa["fold_idx"]] = missing_words

    print(f"\n  Test set sizes and word coverage (fixed across all rounds):")
    for fa in fold_assignments:
        n_missing = len(missing_words_per_fold[fa["fold_idx"]])
        missing   = missing_words_per_fold[fa["fold_idx"]]
        print(
            f"    Fold {fa['fold_idx']} — {fa['test_user']} : "
            f"{len(test_paths_per_fold[fa['fold_idx']])} files | "
            f"missing_words={n_missing}"
            + (
                f"  {missing[:5]}{'...' if n_missing > 5 else ''}"
                if n_missing else ""
            )
        )

    # Save test-set coverage CSV once — not per round
    coverage_csv = os.path.join(args.results_dir, "test_coverage.csv")
    with open(coverage_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "fold", "test_user", "test_files",
            "n_missing_words", "missing_words",
        ])
        for fa in fold_assignments:
            fold_idx  = fa["fold_idx"]
            n_missing = len(missing_words_per_fold[fold_idx])
            writer.writerow([
                fold_idx,
                fa["test_user"],
                len(test_paths_per_fold[fold_idx]),
                n_missing,
                "|".join(missing_words_per_fold[fold_idx]),
            ])
    print(f"\n  Test coverage CSV saved : {coverage_csv}")

    # ── RQ2 Sweep ─────────────────────────────────────────────
    reps_values     = list(range(args.reps_start, args.reps_end - 1, -1))
    round_summaries = []

    for round_idx, reps in enumerate(reps_values):

        # 14 training users × reps = train videos per word (nominal)
        train_vids_per_word = (num_folds - 1) * reps

        # Folder name: e.g. "140_reps", "126_reps", ..., "14_reps"
        round_dir = os.path.join(
            args.results_dir,
            f"{train_vids_per_word}_reps"
        )
        os.makedirs(round_dir, exist_ok=True)

        print(f"\n\n{'#'*65}")
        print(f"  ROUND {round_idx + 1}/{len(reps_values)}")
        print(f"  Reps per user        : {reps}")
        print(f"  Train videos/word    : {train_vids_per_word}  "
              f"(14 users × {reps})")
        print(f"  Approx total train   : "
              f"~{train_vids_per_word * num_classes:,}")
        print(f"  Output folder        : {round_dir}")
        print(f"{'#'*65}")

        fold_results = []

        for fa in fold_assignments:
            fold_idx    = fa["fold_idx"]
            test_user   = fa["test_user"]
            train_users = fa["train_users"]

            fold_log_dir = os.path.join(round_dir, f"fold_{fold_idx}")

            # Sample reduced training set — deterministic per (round, fold)
            train_paths = sample_train_paths(
                train_users     = train_users,
                user_word_paths = user_word_paths,
                reps_per_user   = reps,
                round_idx       = round_idx,
                fold_idx        = fold_idx,
                base_seed       = config["seed"],
            )
            test_paths = test_paths_per_fold[fold_idx]

            # Patch 1: actual per-word train counts — logged and stored in CSV
            actual_per_word = defaultdict(int)
            for p in train_paths:
                actual_per_word[label_from_filename(p)] += 1
            counts                     = list(actual_per_word.values())
            actual_min                 = int(np.min(counts))
            actual_max                 = int(np.max(counts))
            actual_mean                = float(np.mean(counts))
            nominal                    = len(train_users) * reps
            actual_words_below_nominal = int(
                sum(1 for c in counts if c < nominal)
            )

            print(f"\n  ── Round {round_idx+1} | Fold {fold_idx} "
                  f"| test={test_user} "
                  f"| train={len(train_paths)} files "
                  f"| test={len(test_paths)} files ──")
            print(f"     Train vids/word — nominal={nominal} "
                  f"actual: min={actual_min} max={actual_max} "
                  f"mean={actual_mean:.1f} "
                  f"words_below_nominal={actual_words_below_nominal}")

            # ── Train from scratch ─────────────────────────────
            result = train_one_fold(
                fold_idx      = fold_idx,
                train_paths   = train_paths,
                test_paths    = test_paths,
                label_encoder = label_encoder,
                class_names   = list(label_encoder.classes_),
                fold_log_dir  = fold_log_dir,
                device        = device,
                config        = config,
                tb_root       = os.path.join(
                    args.results_dir, "runs",
                    f"{train_vids_per_word}_reps"
                ),
            )

            # ── Auto-evaluate (saves into fold_log_dir/eval/) ──
            print(f"\n  Evaluating fold {fold_idx}...")
            eval_metrics = evaluate_fold(
                fold_idx      = fold_idx,
                fold_log_dir  = fold_log_dir,
                test_paths    = test_paths,
                label_encoder = label_encoder,
                class_names   = list(label_encoder.classes_),
                device        = device,
                config        = config,
            )

            combined = {
                **result,
                "test_user"                  : test_user,
                "reps"                       : reps,
                "train_videos_per_word"      : train_vids_per_word,
                "train_files_total"          : len(train_paths),
                "actual_vids_per_word_min"   : actual_min,
                "actual_vids_per_word_max"   : actual_max,
                "actual_vids_per_word_mean"  : actual_mean,
                "words_below_nominal"        : actual_words_below_nominal,
                "missing_words_in_test"      : len(
                    missing_words_per_fold[fold_idx]
                ),
                # Authoritative metrics come from eval (clean re-run)
                "top1_acc"                   : eval_metrics["top1_acc"],
                "top5_acc"                   : eval_metrics["top5_acc"],
                "macro_f1"                   : eval_metrics["macro_f1"],
                "weighted_f1"                : eval_metrics["weighted_f1"],
                "y_true"                     : eval_metrics["y_true"],
                "y_pred"                     : eval_metrics["y_pred"],
            }
            fold_results.append(combined)

        # ── Round-level aggregation ────────────────────────────
        top1s   = [r["top1_acc"]    for r in fold_results]
        top5s   = [r["top5_acc"]    for r in fold_results]
        mac_f1s = [r["macro_f1"]    for r in fold_results]
        wtd_f1s = [r["weighted_f1"] for r in fold_results]

        mean_top1     = np.mean(top1s)
        std_top1      = np.std(top1s)
        mean_top5     = np.mean(top5s)
        mean_macro_f1 = np.mean(mac_f1s)
        mean_wtd_f1   = np.mean(wtd_f1s)

        print(f"\n  ── Round {round_idx+1} Summary "
              f"({train_vids_per_word} videos/word) ──")
        print(f"  Mean Top-1    : {mean_top1:.2f}% ± {std_top1:.2f}%")
        print(f"  Mean Top-5    : {mean_top5:.2f}%")
        print(f"  Mean Macro F1 : {mean_macro_f1:.2f}%")

        print(f"\n  {'Fold':<6} {'Test User':<14} "
              f"{'Top-1':>8} {'Top-5':>8} {'MacroF1':>9}")
        print(f"  {'-'*50}")
        for r in fold_results:
            print(f"  {r['fold']:<6} {r['test_user']:<14} "
                  f"{r['top1_acc']:7.2f}%  "
                  f"{r['top5_acc']:7.2f}%  "
                  f"{r['macro_f1']:8.2f}%")

        # Save round CSV inside the round folder
        round_csv = os.path.join(round_dir, "round_summary.csv")
        with open(round_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "fold", "test_user",
                "top1(%)", "top5(%)",
                "macro_f1(%)", "weighted_f1(%)",
                "train_files",
                "vids_per_word_nominal",
                "vids_per_word_min",
                "vids_per_word_max",
                "vids_per_word_mean",
                "words_below_nominal",
                "missing_words_in_test",
            ])
            for r in fold_results:
                writer.writerow([
                    r["fold"], r["test_user"],
                    f"{r['top1_acc']:.2f}",
                    f"{r['top5_acc']:.2f}",
                    f"{r['macro_f1']:.2f}",
                    f"{r['weighted_f1']:.2f}",
                    r["train_files_total"],
                    len(r["train_users"]) * reps,
                    r["actual_vids_per_word_min"],
                    r["actual_vids_per_word_max"],
                    f"{r['actual_vids_per_word_mean']:.1f}",
                    r["words_below_nominal"],
                    r["missing_words_in_test"],
                ])
            writer.writerow([])
            writer.writerow([
                "MEAN", "—",
                f"{mean_top1:.2f}", f"{mean_top5:.2f}",
                f"{mean_macro_f1:.2f}", f"{mean_wtd_f1:.2f}",
                "—", "—", "—", "—", "—", "—", "—",
            ])
            writer.writerow([
                "STD", "—",
                f"{std_top1:.2f}", "—",
                "—", "—",
                "—", "—", "—", "—", "—", "—", "—",
            ])

        print(f"  Round CSV saved : {round_csv}")

        round_summaries.append({
            "round"                     : round_idx + 1,
            "reps_per_user"             : reps,
            "train_videos_per_word"     : train_vids_per_word,
            "actual_mean_vids_per_word" : float(np.mean([
                r["actual_vids_per_word_mean"] for r in fold_results
            ])),
            "mean_top1"                 : mean_top1,
            "std_top1"                  : std_top1,
            "mean_top5"                 : mean_top5,
            "mean_macro_f1"             : mean_macro_f1,
            "mean_weighted_f1"          : mean_wtd_f1,
            "max_missing_words_any_fold": max(
                r["missing_words_in_test"] for r in fold_results
            ),
        })

    # ── Global summary CSV ────────────────────────────────────
    csv_path = os.path.join(args.results_dir, "rq2_summary.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "round", "reps_per_user", "train_videos_per_word",
            "actual_mean_vids_per_word",
            "mean_top1(%)", "std_top1(%)",
            "mean_top5(%)", "mean_macro_f1(%)", "mean_weighted_f1(%)",
            "max_missing_words_any_fold",
        ])
        for s in round_summaries:
            writer.writerow([
                s["round"],
                s["reps_per_user"],
                s["train_videos_per_word"],
                f"{s['actual_mean_vids_per_word']:.1f}",
                f"{s['mean_top1']:.2f}",
                f"{s['std_top1']:.2f}",
                f"{s['mean_top5']:.2f}",
                f"{s['mean_macro_f1']:.2f}",
                f"{s['mean_weighted_f1']:.2f}",
                s["max_missing_words_any_fold"],
            ])
    print(f"\n  Global summary CSV : {csv_path}")

    # ── Accuracy curve plot ───────────────────────────────────
    plot_rq2_curve(
        round_summaries,
        save_path=os.path.join(args.results_dir, "rq2_accuracy_curve.png")
    )

    # ── Final console summary ─────────────────────────────────
    print("\n\n" + "=" * 65)
    print("  RQ2 COMPLETE — FINAL SUMMARY")
    print("=" * 65)
    print(f"  {'Folder':<14} {'Reps':<6} {'Vids/Word':<11} "
          f"{'Top-1':>9} {'±std':>7} {'Top-5':>9} {'MacroF1':>9}")
    print(f"  {'-'*65}")
    for s in round_summaries:
        folder = f"{s['train_videos_per_word']}_reps"
        print(
            f"  {folder:<14} {s['reps_per_user']:<6} "
            f"{s['train_videos_per_word']:<11} "
            f"{s['mean_top1']:7.2f}%  "
            f"±{s['std_top1']:.2f}%  "
            f"{s['mean_top5']:7.2f}%  "
            f"{s['mean_macro_f1']:7.2f}%"
        )
    print(f"\n  All results saved to : {args.results_dir}/")
    print(f"  TensorBoard          : "
          f"tensorboard --logdir={args.results_dir}/runs")


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="RQ2: Progressive training data reduction under LOPO"
    )
    parser.add_argument("--data_root",   type=str, default=DATA_ROOT)
    parser.add_argument("--results_dir", type=str, default=RESULTS_DIR)
    parser.add_argument("--reps_start",  type=int, default=REPS_START,
                        help="Max reps per user (default=10, full set)")
    parser.add_argument("--reps_end",    type=int, default=REPS_END,
                        help="Min reps per user (default=1, single rep)")
    args = parser.parse_args()
    main(args)
