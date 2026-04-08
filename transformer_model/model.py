"""
model.py  —  Transformer Classifier with StochasticDepth regularization
=========================================================================
Changes vs original:
  - StochasticDepth (DropPath) added per encoder layer
  - drop_path_rate linearly increases across layers (0 → max)
  - collate_fn_packed unchanged
"""

import math
import torch
import torch.nn as nn


# ==========================================
# Collate  (variable-length → padded batch)
# ==========================================

def collate_fn_packed(batch):
    data, labels, lengths = zip(*batch)

    lengths = torch.stack(lengths).long()
    labels  = torch.stack(labels).long()

    max_len  = int(lengths.max().item())
    feat_dim = data[0].shape[-1]

    padded = []
    for seq, L in zip(data, lengths):
        L = int(L.item())
        if seq.shape[0] < max_len:
            pad = torch.zeros(max_len - seq.shape[0], feat_dim, dtype=seq.dtype)
            seq = torch.cat([seq, pad], dim=0)
        else:
            seq = seq[:max_len]
        padded.append(seq)

    data = torch.stack(padded)                           # (B, T, F)

    padding_mask = torch.zeros(data.size(0), data.size(1), dtype=torch.bool)
    for i, L in enumerate(lengths):
        L = int(L.item())
        if L < data.size(1):
            padding_mask[i, L:] = True

    return data, labels, lengths, padding_mask


# ==========================================
# StochasticDepth (DropPath)
# ==========================================

class StochasticDepth(nn.Module):
    """
    Randomly drop entire residual branches during training.
    drop_prob=0.0 → identity (no drop).
    """
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if not self.training or self.drop_prob == 0.0:
            return x
        keep_prob = 1.0 - self.drop_prob
        # Per-sample mask: shape (B, 1, 1) for broadcasting over (B, T, D)
        shape     = (x.shape[0],) + (1,) * (x.ndim - 1)
        noise     = torch.rand(shape, dtype=x.dtype, device=x.device)
        noise     = torch.floor(noise + keep_prob)
        return x * noise / keep_prob

    def extra_repr(self):
        return f"drop_prob={self.drop_prob:.3f}"


# ==========================================
# Transformer Layer with DropPath
# ==========================================

class TransformerLayerWithDropPath(nn.Module):
    """
    Pre-LN Transformer encoder layer with StochasticDepth on both
    the self-attention and feed-forward residual connections.
    """
    def __init__(self, d_model, nhead, dim_feedforward, dropout, drop_path_rate):
        super().__init__()

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )

        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )

        self.drop_path = StochasticDepth(drop_path_rate)

    def forward(self, x, src_key_padding_mask=None):
        # Self-attention with pre-LN
        normed = self.norm1(x)
        attn_out, _ = self.attn(
            normed, normed, normed,
            key_padding_mask=src_key_padding_mask,
        )
        x = x + self.drop_path(attn_out)

        # Feed-forward with pre-LN
        x = x + self.drop_path(self.ff(self.norm2(x)))
        return x


# ==========================================
# Positional Encoding
# ==========================================

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=600):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe       = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe.unsqueeze(0))     # (1, max_len, d_model)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


# ==========================================
# Transformer Classifier
# ==========================================

class TransformerClassifier(nn.Module):
    """
    input_size      : 128  =  42 joints × 3 coords  +  2 hand-presence flags
    drop_path_rate  : max stochastic depth rate (linearly increases per layer)
    """

    def __init__(
        self,
        input_size=128,
        num_classes=100,
        model_dim=256,
        nhead=8,
        num_layers=4,
        dim_feedforward=512,
        dropout=0.3,
        drop_path_rate=0.1,      # StochasticDepth max rate
    ):
        super().__init__()

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_size, model_dim),
            nn.LayerNorm(model_dim),
            nn.GELU(),
        )

        self.pos_encoder = PositionalEncoding(model_dim, dropout=dropout)

        # Linearly increasing drop path rates: layer 0 gets 0, last gets drop_path_rate
        dpr = [
            drop_path_rate * i / max(1, num_layers - 1)
            for i in range(num_layers)
        ]

        self.layers = nn.ModuleList([
            TransformerLayerWithDropPath(
                d_model         = model_dim,
                nhead           = nhead,
                dim_feedforward = dim_feedforward,
                dropout         = dropout,
                drop_path_rate  = dpr[i],
            )
            for i in range(num_layers)
        ])

        self.final_norm = nn.LayerNorm(model_dim)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(model_dim, model_dim // 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(model_dim // 2, num_classes),
        )

    def forward(self, x, lengths=None, padding_mask=None):
        x = self.input_proj(x)                          # (B, T, model_dim)
        x = self.pos_encoder(x)

        for layer in self.layers:
            x = layer(x, src_key_padding_mask=padding_mask)

        x = self.final_norm(x)

        # Masked mean pooling — only valid frames contribute
        if padding_mask is not None:
            valid = (~padding_mask).unsqueeze(-1).float()       # (B, T, 1)
            x     = (x * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1.0)
        else:
            x = x.mean(dim=1)

        return self.classifier(x)                        # (B, num_classes)

    def get_embeddings(self, x, padding_mask=None):
        """
        Returns the pooled embedding BEFORE the classifier head.
        Used for t-SNE / UMAP visualization.
        """
        x = self.input_proj(x)
        x = self.pos_encoder(x)

        for layer in self.layers:
            x = layer(x, src_key_padding_mask=padding_mask)

        x = self.final_norm(x)

        if padding_mask is not None:
            valid = (~padding_mask).unsqueeze(-1).float()
            x     = (x * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1.0)
        else:
            x = x.mean(dim=1)

        return x                                         # (B, model_dim)