#!/usr/bin/env python3
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"


def _ensure_dirs() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)


def _load_dev_preds() -> pd.DataFrame:
    path = RESULTS / "baseline_dev_preds.csv"
    if not path.exists():
        raise FileNotFoundError("Missing results/baseline_dev_preds.csv. Run train_baseline.py first.")
    df = pd.read_csv(path)
    need = {"par_id", "gold", "pred"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"baseline_dev_preds.csv missing columns: {sorted(missing)}")
    df["par_id"] = pd.to_numeric(df["par_id"], errors="coerce").astype("Int64")
    df["gold"] = pd.to_numeric(df["gold"], errors="coerce").astype("Int64")
    df["pred"] = pd.to_numeric(df["pred"], errors="coerce").astype("Int64")
    if "keyword" not in df.columns:
        df["keyword"] = "unknown"
    df["keyword"] = df["keyword"].astype(str)
    return df.dropna(subset=["par_id", "gold", "pred"]).copy()


def _find_dataset_tsv() -> Path | None:
    candidates = [
        ROOT / "dontpatronizeme_pcl.tsv",
        ROOT / "data" / "dontpatronizeme_pcl.tsv",
        ROOT / "raw" / "dontpatronizeme_pcl.tsv",
        ROOT / "dataset" / "dontpatronizeme_pcl.tsv",
        ROOT / "datasets" / "dontpatronizeme_pcl.tsv",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def _load_full_dataset() -> pd.DataFrame | None:
    p = _find_dataset_tsv()
    if p is None:
        return None
    df = pd.read_csv(p, sep="\t", dtype=str, keep_default_na=False)
    cols = {c.lower(): c for c in df.columns}
    if "par_id" not in cols:
        return None
    par_id_col = cols["par_id"]
    df[par_id_col] = pd.to_numeric(df[par_id_col], errors="coerce").astype("Int64")
    text_col = None
    for k in ["text", "paragraph", "par_text", "paragraph_text"]:
        if k in cols:
            text_col = cols[k]
            break
    if text_col is None:
        return df[[par_id_col]].rename(columns={par_id_col: "par_id"})
    out = df[[par_id_col, text_col]].rename(columns={par_id_col: "par_id", text_col: "text"})
    out["text"] = out["text"].astype(str)
    return out


def save_flowchart() -> None:
    fig, ax = plt.subplots(figsize=(12, 3.6))
    ax.axis("off")

    boxes = {
        "input": (0.06, 0.5, "Input paragraph"),
        "enc": (0.30, 0.5, "RoBERTa-base\n(shared encoder)"),
        "bin": (0.56, 0.72, "Binary head\n(PCL vs No PCL)"),
        "cat": (0.56, 0.28, "Category head\n(7 labels)"),
        "thr": (0.78, 0.5, "Threshold tuning\n(on dev F1)"),
        "out": (0.94, 0.5, "Final label"),
    }

    for _, (x, y, text) in boxes.items():
        ax.text(
            x, y, text,
            ha="center", va="center",
            fontsize=14,
            bbox=dict(boxstyle="round,pad=0.35", linewidth=1.5)
        )

    def arrow(a, b):
        x1, y1, _ = boxes[a]
        x2, y2, _ = boxes[b]
        ax.annotate(
            "",
            xy=(x2 - 0.06, y2),
            xytext=(x1 + 0.06, y1),
            arrowprops=dict(arrowstyle="->", linewidth=2),
        )

    arrow("input", "enc")
    arrow("enc", "bin")
    arrow("enc", "cat")
    arrow("bin", "thr")
    arrow("cat", "thr")
    arrow("thr", "out")

    fig.tight_layout()
    out = FIGURES / "stage3_flowchart.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def save_confusion_matrix(df: pd.DataFrame) -> None:
    y = df["gold"].astype(int).to_numpy()
    p = df["pred"].astype(int).to_numpy()

    tn = int(np.sum((y == 0) & (p == 0)))
    fp = int(np.sum((y == 0) & (p == 1)))
    fn = int(np.sum((y == 1) & (p == 0)))
    tp = int(np.sum((y == 1) & (p == 1)))

    cm = np.array([[tn, fp], [fn, tp]], dtype=int)

    fig, ax = plt.subplots(figsize=(6.5, 5))
    im = ax.imshow(cm)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["no PCL", "PCL"])
    ax.set_yticklabels(["no PCL", "PCL"])
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("Baseline confusion matrix (dev)")

    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")

    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    out = FIGURES / "stage3_confusion_matrix.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"Saved: {out}")


def _metrics(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) else 0.0
    return prec, rec, f1


def save_threshold_curve_if_available(df: pd.DataFrame) -> None:
    score_col = None
    for c in ["score", "prob", "p1", "proba", "p_pcl"]:
        if c in df.columns:
            score_col = c
            break
    if score_col is None:
        return

    y = df["gold"].astype(int).to_numpy()
    s = pd.to_numeric(df[score_col], errors="coerce").to_numpy()
    m = np.isfinite(s)
    y = y[m]
    s = s[m]
    if len(s) == 0:
        return

    thresholds = np.linspace(0.0, 1.0, 101)
    f1s = []
    for t in thresholds:
        pred = (s >= t).astype(int)
        tp = int(np.sum((y == 1) & (pred == 1)))
        fp = int(np.sum((y == 0) & (pred == 1)))
        fn = int(np.sum((y == 1) & (pred == 0)))
        _, _, f1 = _metrics(tp, fp, fn)
        f1s.append(f1)

    best_i = int(np.argmax(f1s))
    best_t = float(thresholds[best_i])
    best_f1 = float(f1s[best_i])

    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.plot(thresholds, f1s)
    ax.scatter([best_t], [best_f1])
    ax.set_xlabel("Decision threshold")
    ax.set_ylabel("F1 (PCL class)")
    ax.set_title("F1 vs threshold on dev set")
    ax.text(best_t, best_f1, f"  best={best_t:.2f}, F1={best_f1:.3f}", va="center")
    fig.tight_layout()

    out = FIGURES / "stage3_threshold_curve.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"Saved: {out}")

    pd.DataFrame({"threshold": thresholds, "f1": f1s}).to_csv(RESULTS / "stage3_threshold_curve.csv", index=False)
    print(f"Saved: {RESULTS / 'stage3_threshold_curve.csv'}")


def save_error_examples_if_possible(dev: pd.DataFrame) -> None:
    full = _load_full_dataset()
    if full is None or "text" not in full.columns:
        return

    work = dev.merge(full, on="par_id", how="left")
    work["text"] = work["text"].fillna("").astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
    y = work["gold"].astype(int)
    p = work["pred"].astype(int)
    work["error_type"] = np.where((y == 1) & (p == 0), "FN", np.where((y == 0) & (p == 1), "FP", "OK"))
    work = work[work["error_type"].isin(["FN", "FP"])].copy()
    work["text"] = work["text"].apply(lambda s: (s[:220] + "…") if len(s) > 221 else s)

    rows = pd.concat([work[work["error_type"] == "FN"].head(6), work[work["error_type"] == "FP"].head(6)], ignore_index=True)
    rows = rows[["error_type", "par_id", "keyword", "gold", "pred", "text"]].copy()

    out_csv = RESULTS / "stage3_error_examples.csv"
    rows.to_csv(out_csv, index=False)
    print(f"Saved: {out_csv}")

    fig, ax = plt.subplots(figsize=(12, 3.0 + 0.45 * len(rows)))
    ax.axis("off")
    table = ax.table(
        cellText=rows.values,
        colLabels=rows.columns,
        loc="upper left",
        cellLoc="left",
        colLoc="left",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.25)
    fig.tight_layout()

    out_png = FIGURES / "stage3_error_examples.png"
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_png}")


def main() -> None:
    _ensure_dirs()
    dev = _load_dev_preds()
    save_flowchart()
    save_confusion_matrix(dev)
    save_threshold_curve_if_available(dev)
    save_error_examples_if_possible(dev)


if __name__ == "__main__":
    main()