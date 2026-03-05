from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoModel

from .data import PCLMultiTaskDataset, collate_fn


def read_test_tsv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", header=None, engine="python")
    df = df.iloc[:, :5]
    df.columns = ["par_id", "art_id", "keyword", "country", "text"]
    df["par_id"] = df["par_id"].astype(str)
    df["text"] = df["text"].astype(str)
    return df[["par_id", "text"]]


def load_model(backbone: str):
    backbone_model = AutoModel.from_pretrained(backbone)
    hidden = backbone_model.config.hidden_size
    head = torch.nn.Linear(hidden, 1)
    model = torch.nn.ModuleDict({"backbone": backbone_model, "head": head})
    return model


@torch.no_grad()
def predict_probs(model, loader, device):
    model.eval()
    par_ids = []
    probs = []

    for batch in loader:
        input_ids = batch.input_ids.to(device)
        attention_mask = batch.attention_mask.to(device)

        out = model["backbone"](input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0, :]
        logits = model["head"](cls).squeeze(-1)

        p = torch.sigmoid(logits).detach().cpu().numpy()

        par_ids.extend(batch.par_id)
        probs.append(p)

    probs = np.concatenate(probs, axis=0)
    return par_ids, probs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir", default="data/raw")
    ap.add_argument("--test_tsv", required=True)
    ap.add_argument("--backbone", default="roberta-base")
    ap.add_argument("--max_length", type=int, default=128)
    ap.add_argument("--ckpt_dir", required=True)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--out_csv", required=True)
    args = ap.parse_args()

    raw_dir = Path(args.raw_dir)
    ckpt_dir = Path(args.ckpt_dir)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    test_path = raw_dir / args.test_tsv
    df = read_test_tsv(test_path)

    ds = PCLMultiTaskDataset(
        df,
        tokenizer_name=args.backbone,
        max_length=args.max_length,
        has_labels=False,
    )

    loader = DataLoader(ds, batch_size=32, shuffle=False, collate_fn=collate_fn)

    model = load_model(args.backbone)

    state = torch.load(ckpt_dir / "pytorch_model.bin", map_location="cpu")
    model.load_state_dict(state, strict=False)

    model.to(device)

    par_ids, probs = predict_probs(model, loader, device)

    preds = (probs >= float(args.threshold)).astype(int)

    out = pd.DataFrame({"par_id": par_ids, "pred": preds, "prob": probs})
    out.to_csv(args.out_csv, index=False)

    print(f"Saved: {args.out_csv}")


if __name__ == "__main__":
    main()