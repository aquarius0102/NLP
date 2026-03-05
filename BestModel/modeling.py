from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
from transformers import AutoModel


@dataclass
class Outputs:
    logits_bin: torch.Tensor
    logits_cat: torch.Tensor


class RoBERTaMultiHead(nn.Module):
    def __init__(self, backbone: str = "roberta-base", dropout: float = 0.1):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(backbone)
        hidden = int(self.encoder.config.hidden_size)
        self.drop = nn.Dropout(dropout)
        self.bin_head = nn.Linear(hidden, 1)
        self.cat_head = nn.Linear(hidden, 7)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Outputs:
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = out.last_hidden_state[:, 0, :]
        pooled = self.drop(pooled)
        logits_bin = self.bin_head(pooled).squeeze(-1)
        logits_cat = self.cat_head(pooled)
        return Outputs(logits_bin=logits_bin, logits_cat=logits_cat)


def loss_fn(
    logits_bin: torch.Tensor,
    logits_cat: torch.Tensor,
    y_bin: torch.Tensor,
    y_cat: torch.Tensor,
    pos_weight_bin: Optional[torch.Tensor] = None,
    pos_weight_cat: Optional[torch.Tensor] = None,
    alpha: float = 1.0,
) -> torch.Tensor:
    if pos_weight_bin is not None:
        bce_bin = nn.BCEWithLogitsLoss(pos_weight=pos_weight_bin)
    else:
        bce_bin = nn.BCEWithLogitsLoss()

    if pos_weight_cat is not None:
        bce_cat = nn.BCEWithLogitsLoss(pos_weight=pos_weight_cat)
    else:
        bce_cat = nn.BCEWithLogitsLoss()

    l_bin = bce_bin(logits_bin, y_bin)
    l_cat = bce_cat(logits_cat, y_cat)
    return l_bin + float(alpha) * l_cat