import os
import numpy as np
import torch
import h5py
import random
import math
from collections import Counter


# ==========================================
# FILE / LABEL UTILS
# ==========================================

def find_h5_files(root_dir):
    paths = []
    for r, _, files in os.walk(root_dir):
        for f in files:
            if f.lower().endswith(".h5"):
                paths.append(os.path.join(r, f))
    return sorted(paths)


# ==========================================
# LABEL MERGE MAP
# ==========================================

LABEL_MERGE_MAP = {
    "byte": "bite",

    "glove": "gloves",

    "lady": "woman",
    "female": "woman",

    "few": "some",

    "after": "next",

    "over": "finish",

    "large": "big",
    "alot": "big",

    "this": "it",
    "that": "it",

    # typo fix
    "photgraph": "photograph",
}


def label_from_filename(path: str) -> str:
    base = os.path.basename(path)
    stem = os.path.splitext(base)[0].lower()

    if "__" in stem:
        first = stem.split("__", 1)[0]

        if first.startswith("_"):
            parts = [p for p in first.split("_") if p]
            label = parts[-1]
        else:
            label = first

    elif "_" in stem:
        label = stem.split("_", 1)[0]

    else:
        label = stem

    label = LABEL_MERGE_MAP.get(label, label)
    return label


def user_from_path(path: str) -> str:
    parts = path.replace("\\", "/").split("/")

    for part in parts:
        if part.lower().startswith("isl_data_user"):
            return part

    for part in parts:
        if part.lower().startswith("user"):
            return part

    return os.path.basename(os.path.dirname(path))


# ==========================================
# ACTIVE FRAME TRIMMING
# ==========================================

def trim_to_active(data, hand_threshold=1e-3):
    T = data.shape[0]

    left_energy  = np.abs(data[:, :21, :]).sum(axis=(1, 2))
    right_energy = np.abs(data[:, 21:, :]).sum(axis=(1, 2))
    any_hand     = (left_energy > hand_threshold) | (right_energy > hand_threshold)

    active = np.where(any_hand)[0]
    if len(active) == 0:
        return data

    return data[active[0]: active[-1] + 1]


# ==========================================
# PER-HAND NORMALIZATION
# ==========================================

def normalize_per_hand(data):
    data = data.copy().astype(np.float32)

    left_energy = np.abs(data[:, :21, :]).sum(axis=(1, 2))
    left_active = left_energy > 1e-3
    if left_active.any():
        left_wrist = data[left_active, 0:1, :]
        data[left_active, :21, :] -= left_wrist

    right_energy = np.abs(data[:, 21:, :]).sum(axis=(1, 2))
    right_active = right_energy > 1e-3
    if right_active.any():
        right_wrist = data[right_active, 21:22, :]
        data[right_active, 21:, :] -= right_wrist

    return data


# ==========================================
# HAND PRESENCE FLAGS
# ==========================================

def compute_hand_flags(data, hand_threshold=1e-3):
    left_present  = (np.abs(data[:, :21, :]).sum(axis=(1, 2)) > hand_threshold).astype(np.float32)
    right_present = (np.abs(data[:, 21:, :]).sum(axis=(1, 2)) > hand_threshold).astype(np.float32)
    return np.stack([left_present, right_present], axis=1)


# ==========================================
# NORMALIZATION STATS
# ==========================================

def compute_normalization_stats(paths, key="intermediate"):
    all_frames = []

    for path in paths:
        try:
            with h5py.File(path, "r") as f:
                data = f[key][:]

            if data.shape[0] == 0:
                continue

            data = data.reshape(data.shape[0], -1, 3).astype(np.float32)
            data = trim_to_active(data)

            if data.shape[0] == 0:
                continue

            data = normalize_per_hand(data)
            all_frames.append(data)

        except Exception as e:
            print(f"[NormStats] Skipping {path}: {e}")

    if not all_frames:
        print("[NormStats] WARNING: No valid files. Returning identity stats.")
        return {
            "mean": np.zeros((1, 42, 3), dtype=np.float32),
            "std":  np.ones((1, 42, 3),  dtype=np.float32),
        }

    concat = np.concatenate(all_frames, axis=0)
    mean   = np.nanmean(concat, axis=0, keepdims=True).astype(np.float32)
    std    = (np.nanstd(concat, axis=0, keepdims=True) + 1e-6).astype(np.float32)

    print(f"[NormStats] {concat.shape[0]} active frames from {len(all_frames)} files.")
    return {"mean": mean, "std": std}


# ==========================================
# DATASET CLASS
# ==========================================

class H5SignDataset(torch.utils.data.Dataset):

    FEATURE_DIM = 128

    def __init__(
        self,
        file_paths,
        label_encoder,
        n=150,
        normalize_stats=None,
        augment=False,
        key="intermediate",
        hand_threshold=1e-3,
    ):
        self.paths           = list(file_paths)
        self.n               = n
        self.normalize_stats = normalize_stats
        self.augment         = augment
        self.key             = key
        self.hand_threshold  = hand_threshold

        if not self.paths:
            raise ValueError("No .h5 file paths provided to H5SignDataset.")

        self.labels_str     = [label_from_filename(p) for p in self.paths]
        self.encoded_labels = label_encoder.transform(self.labels_str)

    def __len__(self):
        return len(self.paths)

    def _make_empty(self, label):
        feat = np.zeros((self.n, self.FEATURE_DIM), dtype=np.float32)
        return (
            torch.tensor(feat,  dtype=torch.float32),
            torch.tensor(label, dtype=torch.long),
            torch.tensor(0,     dtype=torch.long),
        )

    def __getitem__(self, idx):
        path  = self.paths[idx]
        label = int(self.encoded_labels[idx])

        try:
            with h5py.File(path, "r") as f:
                data = f[self.key][:]
            data = data.reshape(data.shape[0], -1, 3).astype(np.float32)
        except Exception as e:
            print(f"[Dataset] Error reading {path}: {e}")
            return self._make_empty(label)

        if data.shape[0] == 0:
            return self._make_empty(label)

        data = trim_to_active(data, self.hand_threshold)
        if data.shape[0] == 0:
            return self._make_empty(label)

        data = normalize_per_hand(data)

        if self.normalize_stats is not None:
            data = (data - self.normalize_stats["mean"]) / self.normalize_stats["std"]

        flags = compute_hand_flags(data, self.hand_threshold)

        if self.augment:
            data, flags = self._augment(data, flags)

        T        = data.shape[0]
        orig_len = min(T, self.n)

        if T > self.n:
            start = (T - self.n) // 2
            data  = data[start : start + self.n]
            flags = flags[start : start + self.n]
        elif T < self.n:
            pad_d  = np.zeros((self.n - T, 42, 3), dtype=np.float32)
            pad_f  = np.zeros((self.n - T, 2),     dtype=np.float32)
            data   = np.concatenate([data,  pad_d], axis=0)
            flags  = np.concatenate([flags, pad_f], axis=0)

        feat = np.concatenate([data.reshape(self.n, -1), flags], axis=1)

        return (
            torch.tensor(feat,     dtype=torch.float32),
            torch.tensor(label,    dtype=torch.long),
            torch.tensor(orig_len, dtype=torch.long),
        )

    def _augment(self, data, flags):
        T = data.shape[0]

        if random.random() < 0.7 and T > 4:
            factor  = random.uniform(0.75, 1.35)
            new_len = max(4, int(T * factor))
            idx     = np.linspace(0, T - 1, new_len).astype(int)
            data    = data[idx]
            flags   = flags[idx]
            T       = new_len

        if random.random() < 0.8:
            xy = data[..., :2].copy()
            z  = data[..., 2:]

            scale = random.uniform(0.85, 1.15)
            xy   *= scale

            angle = random.uniform(-15, 15) * math.pi / 180
            c, s  = math.cos(angle), math.sin(angle)
            R     = np.array([[c, -s], [s, c]])
            xy    = xy @ R.T

            shift = np.random.uniform(-0.03, 0.03, size=(1, 1, 2))
            xy   += shift

            data = np.concatenate([xy, z], axis=-1)

        if random.random() < 0.6:
            data = data + np.random.normal(0, 0.01, data.shape).astype(np.float32)

        if random.random() < 0.4 and T > 8:
            mask_len = random.randint(3, min(12, T // 3))
            start    = random.randint(0, T - mask_len)
            data[start : start + mask_len] = np.random.normal(
                0, 0.05, (mask_len, 42, 3)
            ).astype(np.float32)
            flags[start : start + mask_len] = 0.0

        if random.random() < 0.3:
            left  = data[:, :21, :].copy()
            right = data[:, 21:, :].copy()
            data[:, :21, :] = right
            data[:, 21:, :] = left
            data[..., 0]   *= -1
            flags            = flags[:, [1, 0]]

        if random.random() < 0.5:
            scale = random.uniform(0.85, 1.15)
            data  = data * scale

        return data, flags
