from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import torch
from torch.utils.data import Dataset


@dataclass
class Example:
    par_id: int
    text: str
    label: Optional[int] = None  # binary 0/1


class PCLDataset(Dataset):
    """
    Minimal torch Dataset wrapper. Keeps it simple for coursework.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        text_col: str = "text",
        label_col: str = "binary_label",
        id_col: str = "par_id",
    ):
        self.df = df.reset_index(drop=True)
        self.text_col = text_col
        self.label_col = label_col
        self.id_col = id_col

        if self.text_col not in self.df.columns:
            raise ValueError(f"Missing text column '{self.text_col}'")
        if self.id_col not in self.df.columns:
            raise ValueError(f"Missing id column '{self.id_col}'")

        self.has_labels = self.label_col in self.df.columns

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Example:
        row = self.df.iloc[idx]
        par_id = int(row[self.id_col])
        text = str(row[self.text_col])

        if self.has_labels:
            label = int(row[self.label_col])
        else:
            label = None

        return Example(par_id=par_id, text=text, label=label)


def collate_text_only(batch: List[Example]) -> Dict[str, List]:
    """
    Collate fn for DataLoader (text only).
    """
    return {
        "par_id": [ex.par_id for ex in batch],
        "text": [ex.text for ex in batch],
        "label": [ex.label for ex in batch],
    }


def load_processed_csv(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    df = pd.read_csv(path)
    if "text" in df.columns:
        df["text"] = df["text"].fillna("")
    return df