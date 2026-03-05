from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer


@dataclass
class Batch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    par_id: List[str]
    y_bin: Optional[torch.Tensor] = None
    y_cat: Optional[torch.Tensor] = None


def _parse_multilabel_cell(x: object) -> List[int]:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return [0] * 7
    if isinstance(x, list):
        return [int(v) for v in x]
    s = str(x).strip()
    try:
        v = ast.literal_eval(s)
        if isinstance(v, list):
            return [int(t) for t in v]
    except Exception:
        pass
    s = s.strip("[]")
    parts = [p.strip() for p in s.split(",") if p.strip() != ""]
    if len(parts) == 7:
        return [int(p) for p in parts]
    raise ValueError(f"Cannot parse multilabel cell: {x}")


def load_main_tsv(tsv_path: str) -> pd.DataFrame:
    df = pd.read_csv(tsv_path, sep="\t", skiprows=4, header=None, engine="python")
    df.columns = ["par_id", "art_id", "keyword", "country", "text", "label"]
    df["par_id"] = df["par_id"].astype(str)
    df["text"] = df["text"].astype(str)
    return df


def load_split_labels(path: Union[str, Path]) -> pd.DataFrame:
    path = Path(path)
    df = pd.read_csv(path)
    if "par_id" not in df.columns:
        df = df.rename(columns={df.columns[0]: "par_id"})
    df["par_id"] = df["par_id"].astype(str)

    if "label" in df.columns:
        df["y_cat"] = df["label"].apply(_parse_multilabel_cell)
    elif "labels" in df.columns:
        df["y_cat"] = df["labels"].apply(_parse_multilabel_cell)
    else:
        df["y_cat"] = [[0] * 7 for _ in range(len(df))]

    df["y_cat"] = df["y_cat"].apply(lambda xs: [int(x) for x in xs])
    return df[["par_id", "y_cat"]]


def build_split(main_df: pd.DataFrame, split_df: pd.DataFrame) -> pd.DataFrame:
    main_df = main_df.copy()
    split_df = split_df.copy()
    main_df["par_id"] = main_df["par_id"].astype(str)
    split_df["par_id"] = split_df["par_id"].astype(str)

    out = main_df.merge(split_df, on="par_id", how="inner")
    if "binary_label" not in out.columns:
        out["binary_label"] = out["y_cat"].apply(lambda xs: 1 if sum(xs) > 0 else 0).astype(int)
    out["text"] = out["text"].astype(str)
    return out


class PCLMultiTaskDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer_name: str = "roberta-base",
        max_length: int = 128,
        has_labels: bool = True,
    ):
        self.df = df.reset_index(drop=True)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
        self.max_length = int(max_length)
        self.has_labels = bool(has_labels)

        req = {"par_id", "text"}
        missing = req - set(self.df.columns)
        if missing:
            raise ValueError(f"Missing columns: {missing}")

        self.df["par_id"] = self.df["par_id"].astype(str)
        self.df["text"] = self.df["text"].astype(str)

        if self.has_labels:
            if "binary_label" not in self.df.columns or "y_cat" not in self.df.columns:
                raise ValueError("Labels requested but binary_label or y_cat missing")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        row = self.df.iloc[idx]
        text = str(row["text"])
        enc = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors=None,
        )
        item: Dict[str, object] = {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "par_id": str(row["par_id"]),
        }
        if self.has_labels:
            item["y_bin"] = int(row["binary_label"])
            item["y_cat"] = [int(x) for x in row["y_cat"]]
        return item


def collate_fn(batch: List[Dict[str, object]]) -> Batch:
    max_len = max(len(x["input_ids"]) for x in batch)
    bsz = len(batch)

    input_ids = torch.full((bsz, max_len), fill_value=1, dtype=torch.long)
    attention_mask = torch.zeros((bsz, max_len), dtype=torch.long)
    par_id: List[str] = ["" for _ in range(bsz)]

    y_bin = None
    y_cat = None

    has_labels = "y_bin" in batch[0] and "y_cat" in batch[0]
    if has_labels:
        y_bin = torch.zeros((bsz,), dtype=torch.float)
        y_cat = torch.zeros((bsz, 7), dtype=torch.float)

    for i, ex in enumerate(batch):
        ids = torch.tensor(ex["input_ids"], dtype=torch.long)
        msk = torch.tensor(ex["attention_mask"], dtype=torch.long)
        input_ids[i, : ids.numel()] = ids
        attention_mask[i, : msk.numel()] = msk
        par_id[i] = str(ex["par_id"])
        if has_labels:
            y_bin[i] = float(ex["y_bin"])
            y_cat[i] = torch.tensor(ex["y_cat"], dtype=torch.float)

    return Batch(
        input_ids=input_ids,
        attention_mask=attention_mask,
        par_id=par_id,
        y_bin=y_bin,
        y_cat=y_cat,
    )