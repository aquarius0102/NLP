from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

from .data import load_main_tsv, load_split_labels, build_split, PCLMultiTaskDataset, collate_fn
from .modeling import RoBERTaMultiHead, loss_fn
from .threshold_tune import tune_threshold
from .utils import set_seed, ensure_dir, save_json, metrics_binary, get_device, to_device


def _pos_weight_binary(y: np.ndarray) -> torch.Tensor:
    pos = float((y == 1).sum())
    neg = float((y == 0).sum())
    if pos == 0:
        return torch.tensor(1.0)
    return torch.tensor(neg / pos)


def _pos_weight_cat(y_cat: np.ndarray) -> torch.Tensor:
    pos = y_cat.sum(axis=0).astype(np.float32)
    neg = (y_cat.shape[0] - pos).astype(np.float32)
    pos = np.maximum(pos, 1.0)
    w = neg / pos
    return torch.tensor(w, dtype=torch.float)


@torch.no_grad()
def eval_loop(
    model: RoBERTaMultiHead,
    loader: DataLoader,
    device: torch.device,
    threshold: float = 0.5,
) -> Dict[str, float]:
    model.eval()
    y_true = []
    y_prob = []

    for batch in loader:
        batch = {
            "input_ids": batch.input_ids,
            "attention_mask": batch.attention_mask,
            "par_id": batch.par_id,
            "y_bin": batch.y_bin,
            "y_cat": batch.y_cat,
        }
        batch = to_device(batch, device)
        out = model(batch["input_ids"], batch["attention_mask"])
        prob = torch.sigmoid(out.logits_bin).detach().cpu().numpy()
        gold = batch["y_bin"].detach().cpu().numpy().astype(int)
        y_true.extend(gold.tolist())
        y_prob.extend(prob.tolist())

    y_pred = (np.asarray(y_prob) >= float(threshold)).astype(int)
    m = metrics_binary(y_true, y_pred.tolist())
    return {k: float(v) for k, v in m.items()}


@torch.no_grad()
def get_probs(
    model: RoBERTaMultiHead,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    y_true = []
    y_prob = []
    for batch in loader:
        batch = {
            "input_ids": batch.input_ids,
            "attention_mask": batch.attention_mask,
            "par_id": batch.par_id,
            "y_bin": batch.y_bin,
            "y_cat": batch.y_cat,
        }
        batch = to_device(batch, device)
        out = model(batch["input_ids"], batch["attention_mask"])
        prob = torch.sigmoid(out.logits_bin).detach().cpu().numpy()
        gold = batch["y_bin"].detach().cpu().numpy().astype(int)
        y_true.append(gold)
        y_prob.append(prob)
    return np.concatenate(y_true), np.concatenate(y_prob)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tsv", type=str, default="dontpatronizeme_pcl.tsv")
    ap.add_argument("--raw_dir", type=str, default="data/raw")
    ap.add_argument("--train_csv", type=str, default="train_semeval_parids-labels.csv")
    ap.add_argument("--dev_csv", type=str, default="dev_semeval_parids-labels.csv")
    ap.add_argument("--backbone", type=str, default="roberta-base")
    ap.add_argument("--max_length", type=int, default=128)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--warmup_ratio", type=float, default=0.1)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_dir", type=str, default="BestModel/artifacts")
    args = ap.parse_args()

    set_seed(args.seed)
    device = get_device()

    tsv_path = Path(args.tsv)
    raw_dir = Path(args.raw_dir)
    train_path = raw_dir / args.train_csv
    dev_path = raw_dir / args.dev_csv

    if not tsv_path.exists():
        raise FileNotFoundError(f"Missing {tsv_path}. Put it in the repo root.")
    if not train_path.exists():
        raise FileNotFoundError(f"Missing {train_path}.")
    if not dev_path.exists():
        raise FileNotFoundError(f"Missing {dev_path}.")

    main_df = load_main_tsv(tsv_path)
    train_split = load_split_labels(train_path)
    dev_split = load_split_labels(dev_path)

    train_df = build_split(main_df, train_split)
    dev_df = build_split(main_df, dev_split)

    train_ds = PCLMultiTaskDataset(train_df, tokenizer_name=args.backbone, max_length=args.max_length, has_labels=True)
    dev_ds = PCLMultiTaskDataset(dev_df, tokenizer_name=args.backbone, max_length=args.max_length, has_labels=True)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    dev_loader = DataLoader(dev_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    y_train = train_df["binary_label"].to_numpy(dtype=int)
    y_train_cat = np.stack(train_df["y_cat"].to_numpy(), axis=0).astype(np.float32)

    pos_w_bin = _pos_weight_binary(y_train).to(device)
    pos_w_cat = _pos_weight_cat(y_train_cat).to(device)

    model = RoBERTaMultiHead(backbone=args.backbone).to(device)

    no_decay = ["bias", "LayerNorm.weight"]
    grouped = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]

    opt = torch.optim.AdamW(grouped, lr=args.lr)

    steps_per_epoch = max(1, len(train_loader))
    total_steps = steps_per_epoch * int(args.epochs)
    warmup_steps = int(total_steps * float(args.warmup_ratio))
    sched = get_linear_schedule_with_warmup(opt, warmup_steps, total_steps)

    out_dir = ensure_dir(args.out_dir)
    best_dir = ensure_dir(out_dir / "best_checkpoint")

    best_f1 = -1.0
    best_threshold = 0.5

    for epoch in range(int(args.epochs)):
        model.train()
        running = 0.0
        n_batches = 0

        for batch in train_loader:
            batch = {
                "input_ids": batch.input_ids,
                "attention_mask": batch.attention_mask,
                "par_id": batch.par_id,
                "y_bin": batch.y_bin,
                "y_cat": batch.y_cat,
            }
            batch = to_device(batch, device)

            opt.zero_grad(set_to_none=True)
            out = model(batch["input_ids"], batch["attention_mask"])

            loss = loss_fn(
                out.logits_bin,
                out.logits_cat,
                batch["y_bin"],
                batch["y_cat"],
                pos_weight_bin=pos_w_bin,
                pos_weight_cat=pos_w_cat,
                alpha=float(args.alpha),
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()

            running += float(loss.item())
            n_batches += 1

        y_true, y_prob = get_probs(model, dev_loader, device)
        tr = tune_threshold(y_true, y_prob)
        best_threshold_epoch = tr.threshold

        m = eval_loop(model, dev_loader, device, threshold=best_threshold_epoch)
        save_json(out_dir / f"dev_metrics_epoch_{epoch+1}.json", {"epoch": epoch + 1, "threshold": best_threshold_epoch, **m})

        if m["f1"] > best_f1:
            best_f1 = float(m["f1"])
            best_threshold = float(best_threshold_epoch)
            torch.save(model.state_dict(), best_dir / "pytorch_model.bin")
            save_json(best_dir / "meta.json", {"backbone": args.backbone, "max_length": args.max_length, "alpha": float(args.alpha)})
            save_json(best_dir / "threshold.json", {"threshold": best_threshold, "dev_f1": best_f1})

    print(f"Best dev F1: {best_f1:.4f} at threshold {best_threshold:.3f}")
    print(f"Saved best checkpoint to: {best_dir}")


if __name__ == "__main__":
    main()