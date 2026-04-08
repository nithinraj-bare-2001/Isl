"""
train.py  —  Single-fold training (no early stopping)
=======================================================
  - Fixed epochs — runs ALL epochs no matter what
  - Saves only final epoch checkpoint (final_model.pt)
  - Reports final epoch metrics as primary result
  - MixUp augmentation on batches (length-aware)
  - StochasticDepth in model (configured in model.py)
  - Gradient clipping (max_norm=1.0)
  - CosineAnnealing LR with linear warmup
  - Train accuracy measured on clean inputs (second forward pass)
  - TensorBoard: Loss, Top1_Acc, Top5_Acc (train+test), LR
"""

import os
import sys
import math
import time

import numpy as np
import torch
import torch.nn.functional as F

from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    precision_score,
    recall_score,
    f1_score,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dataset_class import H5SignDataset, compute_normalization_stats
from model import TransformerClassifier, collate_fn_packed


# ==========================================
# Tee Logger
# ==========================================

class Tee:
    def __init__(self, *files):
        self.files = files

    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()

    def flush(self):
        for f in self.files:
            f.flush()


# ==========================================
# MixUp  (length-aware)
# ==========================================

def mixup_batch(inputs, labels, lengths, num_classes, alpha=0.4):
    """
    MixUp: blends two random samples and their one-hot labels.
    Length-aware: frames beyond min(len_a, len_b) are zeroed to avoid
    mixing valid frames with padding zeros from another sequence.
    """
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    B   = inputs.size(0)
    idx = torch.randperm(B, device=inputs.device)

    mixed_inputs = lam * inputs + (1.0 - lam) * inputs[idx]

    # Zero out frames beyond the shorter of each paired sequence
    paired_lengths = torch.minimum(lengths, lengths[idx])
    for i in range(B):
        valid = int(paired_lengths[i].item())
        if valid < mixed_inputs.size(1):
            mixed_inputs[i, valid:] = 0.0

    labels_onehot     = F.one_hot(labels, num_classes).float()
    labels_onehot_idx = F.one_hot(labels[idx], num_classes).float()
    mixed_labels      = lam * labels_onehot + (1.0 - lam) * labels_onehot_idx

    return mixed_inputs, mixed_labels


def mixup_cross_entropy(outputs, soft_labels, label_smoothing=0.0):
    """Cross entropy that works with soft (mixed) labels."""
    log_probs = F.log_softmax(outputs, dim=1)
    if label_smoothing > 0:
        n_classes   = outputs.size(1)
        smooth      = label_smoothing / n_classes
        soft_labels = soft_labels * (1.0 - label_smoothing) + smooth
    loss = -(soft_labels * log_probs).sum(dim=1).mean()
    return loss


# ==========================================
# Collect Predictions
# ==========================================

@torch.no_grad()
def collect_preds(model, loader, device, top_k=5):
    """
    Returns:
        y_true   : (N,)
        y_pred   : (N,)        top-1
        y_topk   : (N, top_k)
        avg_loss : float       mean CE loss (no label smoothing)
    """
    model.eval()
    y_true, y_pred, y_topk = [], [], []
    total_loss, total_n    = 0.0, 0

    for inputs, labels, lengths, padding_mask in loader:
        inputs       = inputs.to(device)
        labels       = labels.to(device)
        padding_mask = padding_mask.to(device)

        outputs = model(inputs, padding_mask=padding_mask)
        loss    = F.cross_entropy(outputs, labels, reduction="sum")

        preds = outputs.argmax(dim=1)
        k     = min(top_k, outputs.size(1))
        topk  = outputs.topk(k, dim=1).indices

        y_true.append(labels.cpu().numpy())
        y_pred.append(preds.cpu().numpy())
        y_topk.append(topk.cpu().numpy())
        total_loss += loss.item()
        total_n    += labels.size(0)

    return (
        np.concatenate(y_true),
        np.concatenate(y_pred),
        np.concatenate(y_topk, axis=0),
        total_loss / max(total_n, 1),
    )


def top_k_accuracy(y_true, y_topk):
    # Vectorised: no Python loop
    correct = (y_topk == y_true[:, None]).any(axis=1).sum()
    return 100.0 * correct / len(y_true)


# ==========================================
# Plot Curves
# ==========================================

def plot_curves(train_vals, test_vals, ylabel, title, save_path,
                train_label="Train", test_label="Test"):
    plt.figure(figsize=(10, 5))
    epochs = list(range(len(train_vals)))
    plt.plot(epochs, train_vals, label=train_label, linewidth=1.5)
    plt.plot(epochs, test_vals,  label=test_label,  linewidth=1.5)
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=100)
    plt.close()


# ==========================================
# Confusion Matrix
# ==========================================

def save_confusion_matrix(cm, class_names, save_path, title="Confusion Matrix"):
    n       = len(class_names)
    figsize = max(14, n // 3)

    fig, ax = plt.subplots(figsize=(figsize, figsize))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.colorbar(im, ax=ax)

    if n <= 60:
        ticks = np.arange(n)
        ax.set_xticks(ticks)
        ax.set_xticklabels(class_names, rotation=90, fontsize=7)
        ax.set_yticks(ticks)
        ax.set_yticklabels(class_names, fontsize=7)

    ax.set_ylabel("True Label")
    ax.set_xlabel("Predicted Label")
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=100)
    plt.close()


# ==========================================
# Top-N Confused Pairs
# ==========================================

def save_top_confused_pairs(cm, class_names, save_path, n=20):
    pairs = []
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            if i != j and cm[i, j] > 0:
                pairs.append((class_names[i], class_names[j], cm[i, j]))

    pairs.sort(key=lambda x: -x[2])
    pairs = pairs[:n]

    with open(save_path.replace(".png", ".csv"), "w") as f:
        f.write("true_class,predicted_as,count\n")
        for true, pred, count in pairs:
            f.write(f"{true},{pred},{count}\n")

    if pairs:
        labels = [f"{t}→{p}" for t, p, _ in pairs]
        counts = [c for _, _, c in pairs]

        plt.figure(figsize=(14, 6))
        plt.barh(labels[::-1], counts[::-1], color="steelblue", edgecolor="black")
        plt.xlabel("Count")
        plt.title(f"Top {n} Most Confused Class Pairs")
        plt.tight_layout()
        plt.savefig(save_path, dpi=100)
        plt.close()


# ==========================================
# Per-Class Accuracy
# ==========================================

def save_per_class_accuracy(y_true, y_pred, class_names, save_dir):
    rows = []
    for i, name in enumerate(class_names):
        mask      = y_true == i
        n_samples = int(mask.sum())
        acc       = (y_pred[mask] == i).mean() * 100.0 if n_samples > 0 else 0.0
        rows.append((name, n_samples, acc))

    rows.sort(key=lambda x: x[2])

    csv_path = os.path.join(save_dir, "per_class_accuracy.csv")
    with open(csv_path, "w") as f:
        f.write("class,n_samples,accuracy(%)\n")
        for name, n, acc in rows:
            f.write(f"{name},{n},{acc:.2f}\n")

    names  = [r[0] for r in rows]
    accs   = [r[2] for r in rows]
    n      = len(names)
    fig_h  = max(8, n * 0.18)

    plt.figure(figsize=(12, fig_h))
    colors = ["#d73027" if a < 50 else "#4575b4" for a in accs]
    plt.barh(names, accs, color=colors, edgecolor="none", height=0.7)
    plt.axvline(x=np.mean(accs), color="black", linestyle="--",
                linewidth=1, label=f"Mean={np.mean(accs):.1f}%")
    plt.xlabel("Accuracy (%)")
    plt.title("Per-Class Accuracy (sorted)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "per_class_accuracy_bar.png"), dpi=80)
    plt.close()

    return rows


def print_worst_best_classes(rows, n=10):
    print(f"\n  --- Bottom {n} classes (worst accuracy) ---")
    for name, n_samples, acc in rows[:n]:
        print(f"    {name:<25s}  n={n_samples:4d}  acc={acc:.1f}%")
    print(f"\n  --- Top {n} classes (best accuracy) ---")
    for name, n_samples, acc in rows[-n:][::-1]:
        print(f"    {name:<25s}  n={n_samples:4d}  acc={acc:.1f}%")


def save_classification_report(y_true, y_pred, class_names, save_path):
    report = classification_report(
        y_true, y_pred,
        target_names=class_names,
        digits=4,
        zero_division=0,
    )
    with open(save_path, "w") as f:
        f.write(report)
    return report


# ==========================================
# LR Lambda  (linear warmup + cosine decay)
# ==========================================

def make_lr_lambda(warmup_epochs, total_epochs, min_lr_ratio=0.05):
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return float(epoch + 1) / float(warmup_epochs + 1)
        # Denominator -1 ensures progress == 1.0 exactly at final epoch
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs - 1)
        return max(min_lr_ratio, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return lr_lambda


# ==========================================
# Train One Fold
# ==========================================

def train_one_fold(
    fold_idx,
    train_paths,
    test_paths,
    label_encoder,
    class_names,
    fold_log_dir,
    device,
    config,
    tb_root="results/runs",
):
    os.makedirs(fold_log_dir, exist_ok=True)

    tb_dir = os.path.join(tb_root, f"fold_{fold_idx}")
    writer = SummaryWriter(log_dir=tb_dir)

    print(f"\n{'='*60}")
    print(f"  FOLD {fold_idx}  |  Train={len(train_paths)}  Test={len(test_paths)}")
    print(f"  TensorBoard : {tb_dir}")
    print(f"{'='*60}")

    # ── Normalization stats (train only) ───────────────────────
    print("  Computing normalization stats...")
    norm_stats = compute_normalization_stats(train_paths)
    np.save(os.path.join(fold_log_dir, "norm_mean.npy"), norm_stats["mean"])
    np.save(os.path.join(fold_log_dir, "norm_std.npy"),  norm_stats["std"])

    # ── Datasets ───────────────────────────────────────────────
    train_set = H5SignDataset(
        train_paths, label_encoder, n=config["max_frames"],
        normalize_stats=norm_stats, augment=True,
    )
    test_set = H5SignDataset(
        test_paths, label_encoder, n=config["max_frames"],
        normalize_stats=norm_stats, augment=False,
    )

    train_loader = DataLoader(
        train_set, batch_size=config["batch_size"], shuffle=True,
        collate_fn=collate_fn_packed,
        num_workers=config["num_workers"], pin_memory=True,
    )
    test_loader = DataLoader(
        test_set, batch_size=config["batch_size"], shuffle=False,
        collate_fn=collate_fn_packed,
        num_workers=config["num_workers"], pin_memory=True,
    )

    # ── Model ──────────────────────────────────────────────────
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

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable parameters: {n_params:,}")

    # ── Optimizer ──────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr           = config["lr"],
        weight_decay = config["weight_decay"],
    )

    total_epochs  = config["epochs"]
    warmup_epochs = config.get("warmup_epochs", 5)
    min_lr_ratio  = config.get("min_lr_ratio", 0.05)

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        make_lr_lambda(warmup_epochs, total_epochs, min_lr_ratio),
    )
    scaler = GradScaler()

    mixup_alpha  = config.get("mixup_alpha", 0.4)
    use_mixup    = config.get("use_mixup", True)
    label_smooth = config.get("label_smoothing", 0.1)

    # ── Per-epoch tracking ─────────────────────────────────────
    train_losses = []
    test_losses  = []
    train_top1s  = []   # train top-1 per epoch (clean inputs, hard labels)
    train_top5s  = []   # train top-5 per epoch (clean inputs, hard labels)
    test_top1s   = []   # test  top-1 per epoch (clean inputs)
    test_top5s   = []   # test  top-5 per epoch (clean inputs)

    print(f"\n  {'Epoch':>5}  {'Time':>6}  {'LR':>8}  "
          f"{'TrLoss':>8}  {'TrTop1':>7}  {'TrTop5':>7}  "
          f"{'TeLoss':>8}  {'TeTop1':>7}  {'TeTop5':>7}")
    print(f"  {'-'*82}")

    for epoch in range(total_epochs):
        t0 = time.time()

        # ── Train ─────────────────────────────────────────────
        model.train()
        tr_loss      = 0.0
        tr_top1_corr = 0
        tr_top5_corr = 0
        tr_total     = 0

        for inputs, labels, lengths, padding_mask in train_loader:
            inputs, labels, lengths, padding_mask = (
                inputs.to(device), labels.to(device),
                lengths.to(device), padding_mask.to(device)
            )
            optimizer.zero_grad()

            if use_mixup:
                mixed_inputs, soft_labels = mixup_batch(
                    inputs, labels, lengths, num_classes, alpha=mixup_alpha
                )
            else:
                mixed_inputs = inputs
                soft_labels  = F.one_hot(labels, num_classes).float()

            with autocast(device_type="cuda", enabled=(device.type == "cuda")):
                outputs = model(mixed_inputs, padding_mask=padding_mask)
                loss    = mixup_cross_entropy(outputs, soft_labels, label_smooth)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            # Train accuracy on clean inputs (original, no MixUp).
            # Second forward pass — measures how well model predicts
            # unblended samples, consistent with test accuracy measurement.
            with torch.no_grad():
                clean_outputs = model(inputs, padding_mask=padding_mask)
                k             = min(5, clean_outputs.size(1))
                topk_preds    = clean_outputs.topk(k, dim=1).indices  # (B, k)

                tr_top1_corr += (topk_preds[:, 0] == labels).sum().item()
                tr_top5_corr += (topk_preds == labels.unsqueeze(1)).any(dim=1).sum().item()
                tr_total     += labels.size(0)

            tr_loss += loss.item() * labels.size(0)

        scheduler.step()

        tr_loss /= tr_total
        tr_top1  = 100.0 * tr_top1_corr / tr_total
        tr_top5  = 100.0 * tr_top5_corr / tr_total

        # ── Test (exact, on clean inputs) ─────────────────────
        te_true, te_pred, te_topk, te_loss = collect_preds(
            model, test_loader, device, top_k=5
        )
        te_top1 = (te_true == te_pred).mean() * 100.0
        te_top5 = top_k_accuracy(te_true, te_topk)

        train_losses.append(tr_loss)
        test_losses.append(te_loss)
        train_top1s.append(tr_top1)
        train_top5s.append(tr_top5)
        test_top1s.append(te_top1)
        test_top5s.append(te_top5)

        current_lr = optimizer.param_groups[0]["lr"]
        dt         = time.time() - t0

        print(
            f"  {epoch:5d}  {dt:5.1f}s  {current_lr:.2e}  "
            f"{tr_loss:8.4f}  {tr_top1:6.2f}%  {tr_top5:6.2f}%  "
            f"{te_loss:8.4f}  {te_top1:6.2f}%  {te_top5:6.2f}%"
        )

        # ── TensorBoard ───────────────────────────────────────
        writer.add_scalars("Loss",     {"train": tr_loss, "test": te_loss},   epoch)
        writer.add_scalars("Top1_Acc", {"train": tr_top1, "test": te_top1},   epoch)
        writer.add_scalars("Top5_Acc", {"train": tr_top5, "test": te_top5},   epoch)
        writer.add_scalar("LR",        current_lr,                            epoch)

    # ── Save final checkpoint ──────────────────────────────────
    torch.save(model.state_dict(), os.path.join(fold_log_dir, "final_model.pt"))

    # ── Per-fold curves ────────────────────────────────────────
    plot_curves(
        train_losses, test_losses,
        ylabel="Loss", title=f"Loss Curve — Fold {fold_idx}",
        save_path=os.path.join(fold_log_dir, "loss_curve.png"),
    )
    plot_curves(
        train_top1s, test_top1s,
        ylabel="Top-1 Accuracy (%)", title=f"Top-1 Accuracy — Fold {fold_idx}",
        save_path=os.path.join(fold_log_dir, "top1_curve.png"),
    )
    plot_curves(
        train_top5s, test_top5s,
        ylabel="Top-5 Accuracy (%)", title=f"Top-5 Accuracy — Fold {fold_idx}",
        save_path=os.path.join(fold_log_dir, "top5_curve.png"),
    )

    # ── Final epoch evaluation ─────────────────────────────────
    print(f"\n  --- Final Epoch Evaluation (epoch {total_epochs - 1}) ---")
    y_true_f, y_pred_f, y_topk_f, final_test_loss = collect_preds(
        model, test_loader, device, top_k=5
    )

    final_top1    = (y_true_f == y_pred_f).mean() * 100.0
    final_top5    = top_k_accuracy(y_true_f, y_topk_f)
    final_macro_p = precision_score(y_true_f, y_pred_f, average="macro", zero_division=0) * 100
    final_macro_r = recall_score(   y_true_f, y_pred_f, average="macro", zero_division=0) * 100
    final_macro_f = f1_score(       y_true_f, y_pred_f, average="macro", zero_division=0) * 100
    final_wtd_f   = f1_score(       y_true_f, y_pred_f, average="weighted", zero_division=0) * 100

    print(f"  Top-1        : {final_top1:.2f}%")
    print(f"  Top-5        : {final_top5:.2f}%")
    print(f"  Macro F1     : {final_macro_f:.2f}%")
    print(f"  Weighted F1  : {final_wtd_f:.2f}%")

    # ── TensorBoard hparams ───────────────────────────────────
    writer.add_hparams(
        hparam_dict={
            "lr"            : config["lr"],
            "batch_size"    : config["batch_size"],
            "model_dim"     : config["model_dim"],
            "num_layers"    : config["num_layers"],
            "dropout"       : config["dropout"],
            "drop_path_rate": config.get("drop_path_rate", 0.1),
            "mixup_alpha"   : mixup_alpha,
        },
        metric_dict={
            "hparam/final_top1"    : final_top1,
            "hparam/final_top5"    : final_top5,
            "hparam/final_macro_f1": final_macro_f,
        },
        run_name=f"fold_{fold_idx}",
    )
    writer.flush()
    writer.close()

    # ── Confusion matrix ──────────────────────────────────────
    cm = confusion_matrix(y_true_f, y_pred_f)
    save_confusion_matrix(
        cm, class_names,
        os.path.join(fold_log_dir, "confusion_matrix.png"),
        title=f"Confusion Matrix — Fold {fold_idx} (final epoch)",
    )
    save_top_confused_pairs(
        cm, class_names,
        os.path.join(fold_log_dir, "top_confused_pairs.png"),
        n=20,
    )

    # ── Per-class accuracy ─────────────────────────────────────
    rows = save_per_class_accuracy(y_true_f, y_pred_f, class_names, fold_log_dir)
    print_worst_best_classes(rows, n=10)

    # ── Classification report ──────────────────────────────────
    report = save_classification_report(
        y_true_f, y_pred_f, class_names,
        os.path.join(fold_log_dir, "classification_report.txt"),
    )
    print("\n  Classification Report (averages):")
    for line in report.strip().split("\n")[-5:]:
        print(f"    {line}")

    print(f"\n   Fold {fold_idx} complete. Saved to: {fold_log_dir}")

    return {
        "fold"            : fold_idx,
        # Final epoch test metrics
        "top1_acc"        : final_top1,
        "top5_acc"        : final_top5,
        "macro_precision" : final_macro_p,
        "macro_recall"    : final_macro_r,
        "macro_f1"        : final_macro_f,
        "weighted_f1"     : final_wtd_f,
        "test_loss"       : final_test_loss,
        # Final epoch train metrics (last value in per-epoch lists)
        "train_top1_acc"  : train_top1s[-1],
        "train_top5_acc"  : train_top5s[-1],
        # Predictions for aggregated confusion matrix in runner.py
        "y_true"          : y_true_f,
        "y_pred"          : y_pred_f,
        # Full per-epoch lists for mean curve plots in runner.py
        "train_losses"    : train_losses,
        "test_losses"     : test_losses,
        "train_top1s"     : train_top1s,
        "train_top5s"     : train_top5s,
        "test_top1s"      : test_top1s,
        "test_top5s"      : test_top5s,
    }
