"""
inference.py  —  Load a saved model and run predictions on new data
=====================================================================
Usage:
    from inference import SignInference
    inf = SignInference(fold=0, results_dir="results")
    label, confidence, top5 = inf.predict("path/to/clip.h5")
"""

import os
import numpy as np
import torch

from dataset_class import (
    H5SignDataset,
    trim_to_active,
    normalize_per_hand,
    compute_hand_flags,
)
from model import TransformerClassifier, collate_fn_packed

import h5py


class SignInference:
    """
    Loads the final epoch checkpoint and provides a clean predict() API.

    Parameters
    ----------
    fold        : int   — which fold's model to load
    results_dir : str   — path to results folder (contains fold_X/ subdirs)
    device      : str   — "cuda", "cpu", or "auto"
    config      : dict  — model architecture config (must match training)
    """

    def __init__(
        self,
        fold,
        results_dir="results",
        device="auto",
        config=None,
    ):
        self.results_dir = results_dir
        self.fold        = fold
        self.fold_dir    = os.path.join(results_dir, f"fold_{fold}")

        # ── Device ────────────────────────────────────────────
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # ── Class names ───────────────────────────────────────
        class_names_path = os.path.join(results_dir, "class_names.npy")
        self.class_names = list(np.load(class_names_path, allow_pickle=True))
        self.num_classes = len(self.class_names)

        # ── Normalization stats ────────────────────────────────
        self.norm_mean = np.load(os.path.join(self.fold_dir, "norm_mean.npy"))
        self.norm_std  = np.load(os.path.join(self.fold_dir, "norm_std.npy"))

        # ── Default config ─────────────────────────────────────
        if config is None:
            config = {
                "max_frames"     : 150,
                "model_dim"      : 256,
                "nhead"          : 8,
                "num_layers"     : 4,
                "dim_feedforward": 512,
                "dropout"        : 0.3,
                "drop_path_rate" : 0.1,
            }
        self.config = config

        # ── Model ─────────────────────────────────────────────
        self.model = TransformerClassifier(
            input_size      = H5SignDataset.FEATURE_DIM,
            num_classes     = self.num_classes,
            model_dim       = config["model_dim"],
            nhead           = config["nhead"],
            num_layers      = config["num_layers"],
            dim_feedforward = config["dim_feedforward"],
            dropout         = config["dropout"],
            drop_path_rate  = config.get("drop_path_rate", 0.1),
        ).to(self.device)

        ckpt_path = os.path.join(self.fold_dir, "final_model.pt")
        self.model.load_state_dict(
            torch.load(ckpt_path, map_location=self.device, weights_only=True)
        )
        self.model.eval()
        print(f"[Inference] Loaded final_model.pt from fold {fold} | device={self.device}")

    def _preprocess(self, h5_path, key="intermediate"):
        """Load and preprocess one .h5 file → tensor ready for model."""
        with h5py.File(h5_path, "r") as f:
            data = f[key][:]                              # (T, 2, 21, 3)

        data = data.reshape(data.shape[0], -1, 3).astype(np.float32)  # (T, 42, 3)
        data = trim_to_active(data)
        data = normalize_per_hand(data)

        # Global z-score
        data = (data - self.norm_mean) / self.norm_std

        flags = compute_hand_flags(data)                 # (T, 2)

        n        = self.config["max_frames"]
        T        = data.shape[0]
        orig_len = min(T, n)

        if T > n:
            start = (T - n) // 2
            data  = data[start: start + n]
            flags = flags[start: start + n]
        elif T < n:
            data  = np.concatenate([data,  np.zeros((n - T, 42, 3), dtype=np.float32)], axis=0)
            flags = np.concatenate([flags, np.zeros((n - T, 2),     dtype=np.float32)], axis=0)

        feat = np.concatenate([data.reshape(n, -1), flags], axis=1)  # (n, 128)
        return torch.tensor(feat, dtype=torch.float32), orig_len

    @torch.no_grad()
    def predict(self, h5_path, top_k=5, key="intermediate"):
        """
        Predict class for one .h5 clip.

        Returns
        -------
        top1_label   : str    predicted class name
        confidence   : float  softmax probability of top-1 (0–1)
        top_k_results: list   of (label, prob) tuples for top-k
        """
        feat, orig_len = self._preprocess(h5_path, key=key)
        feat = feat.unsqueeze(0).to(self.device)         # (1, n, 128)

        n    = self.config["max_frames"]
        mask = torch.zeros(1, n, dtype=torch.bool, device=self.device)
        if orig_len < n:
            mask[0, orig_len:] = True

        logits = self.model(feat, padding_mask=mask)     # (1, C)
        probs  = torch.softmax(logits, dim=1)[0]

        k                  = min(top_k, self.num_classes)
        topk_probs, topk_idxs = probs.topk(k)

        top1_label    = self.class_names[topk_idxs[0].item()]
        confidence    = topk_probs[0].item()
        top_k_results = [
            (self.class_names[idx.item()], prob.item())
            for idx, prob in zip(topk_idxs, topk_probs)
        ]

        return top1_label, confidence, top_k_results

    @torch.no_grad()
    def get_embedding(self, h5_path, key="intermediate"):
        """
        Returns the pooled embedding vector (before classifier head).
        Shape: (model_dim,) — useful for t-SNE / UMAP.
        """
        feat, orig_len = self._preprocess(h5_path, key=key)
        feat = feat.unsqueeze(0).to(self.device)

        n    = self.config["max_frames"]
        mask = torch.zeros(1, n, dtype=torch.bool, device=self.device)
        if orig_len < n:
            mask[0, orig_len:] = True

        emb = self.model.get_embeddings(feat, padding_mask=mask)  # (1, model_dim)
        return emb[0].cpu().numpy()


# ── Quick test ────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python inference.py path/to/clip.h5 [fold=0]")
        sys.exit(1)

    h5_path = sys.argv[1]
    fold    = int(sys.argv[2]) if len(sys.argv) > 2 else 0

    inf  = SignInference(fold=fold, results_dir="results")
    top1, conf, topk = inf.predict(h5_path)

    print(f"\nPrediction: {top1}  (confidence: {conf*100:.1f}%)")
    print("Top-5:")
    for label, prob in topk:
        print(f"  {label:<30s} {prob*100:.2f}%")