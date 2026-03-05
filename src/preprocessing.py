from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd


PCL_TSV_COLUMNS = ["par_id", "art_id", "keyword", "country", "text", "label"]


def load_pcl_tsv(path: str | Path) -> pd.DataFrame:
    """
    Loads dontpatronizeme_pcl.tsv (skips the disclaimer header).
    """
    path = Path(path)
    df = pd.read_csv(
        path,
        sep="\t",
        skiprows=4,  # README says data starts at line 5
        names=PCL_TSV_COLUMNS,
        quoting=3,  # avoid weird quote handling
        dtype={"par_id": int, "art_id": int, "keyword": str, "country": str, "text": str, "label": int},
    )
    df["text"] = df["text"].fillna("")
    return df


def add_binary_label(df: pd.DataFrame, label_col: str = "label") -> pd.DataFrame:
    """
    SemEval Task 4 (subtask 1) setup:
      {0,1} -> No PCL (0)
      {2,3,4} -> PCL (1)
    """
    out = df.copy()
    out["binary_label"] = (out[label_col] >= 2).astype(int)
    return out


_whitespace_re = re.compile(r"\s+")
_html_entity_re = re.compile(r"&amp;|&lt;|&gt;|&quot;|&#39;")


def basic_clean_text(text: str) -> str:
    """
    Lightweight cleaning only. Don't over-clean or you lose signal.
    """
    if not isinstance(text, str):
        text = "" if text is None else str(text)

    # common HTML entities -> plain equivalents
    text = text.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
    text = text.replace("&quot;", '"').replace("&#39;", "'")

    # normalise whitespace
    text = _whitespace_re.sub(" ", text).strip()
    return text


def apply_basic_cleaning(df: pd.DataFrame, text_col: str = "text") -> pd.DataFrame:
    out = df.copy()
    out[text_col] = out[text_col].apply(basic_clean_text)
    return out


def load_split_csv(path: str | Path) -> pd.DataFrame:
    """
    Loads train_semeval_parids-labels.csv / dev_semeval_parids-labels.csv.
    Expected columns: par_id,label
    """
    path = Path(path)
    df = pd.read_csv(path)
    # be forgiving about column naming
    df.columns = [c.strip() for c in df.columns]
    if "par_id" not in df.columns:
        raise ValueError(f"{path} missing 'par_id' column. Found: {df.columns.tolist()}")
    if "label" not in df.columns:
        raise ValueError(f"{path} missing 'label' column. Found: {df.columns.tolist()}")
    df["par_id"] = df["par_id"].astype(int)
    df["label"] = df["label"].astype(int)
    return df


def merge_text_with_split(
    pcl_df: pd.DataFrame,
    split_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Join split labels onto the master text table by par_id.
    """
    merged = split_df.merge(
        pcl_df[["par_id", "art_id", "keyword", "country", "text"]],
        on="par_id",
        how="left",
        validate="one_to_one",
    )
    if merged["text"].isna().any():
        missing = int(merged["text"].isna().sum())
        raise ValueError(f"Missing text for {missing} rows after merge (par_id mismatch?)")
    return merged


@dataclass(frozen=True)
class Paths:
    root: Path

    @property
    def data_raw(self) -> Path:
        return self.root / "data" / "raw"

    @property
    def data_processed(self) -> Path:
        return self.root / "data" / "processed"

    @property
    def figures(self) -> Path:
        return self.root / "figures"


def prepare_train_dev(
    repo_root: str | Path,
    clean: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Produces train/dev tables with text + binary_label, using the official split CSVs.
    """
    paths = Paths(Path(repo_root))

    pcl = load_pcl_tsv(paths.data_raw / "dontpatronizeme_pcl.tsv")

    train_split = load_split_csv(paths.data_raw / "train_semeval_parids-labels.csv")
    dev_split = load_split_csv(paths.data_raw / "dev_semeval_parids-labels.csv")

    train = merge_text_with_split(pcl, train_split)
    dev = merge_text_with_split(pcl, dev_split)

    train = add_binary_label(train, label_col="label")
    dev = add_binary_label(dev, label_col="label")

    if clean:
        train = apply_basic_cleaning(train, text_col="text")
        dev = apply_basic_cleaning(dev, text_col="text")

    return train, dev


def save_processed_splits(
    train_df: pd.DataFrame,
    dev_df: pd.DataFrame,
    repo_root: str | Path,
    prefix: str = "pcl",
) -> Tuple[Path, Path]:
    """
    Saves to data/processed/{prefix}_train.csv and {prefix}_dev.csv.
    """
    root = Path(repo_root)
    out_dir = root / "data" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)

    train_path = out_dir / f"{prefix}_train.csv"
    dev_path = out_dir / f"{prefix}_dev.csv"

    train_df.to_csv(train_path, index=False)
    dev_df.to_csv(dev_path, index=False)

    return train_path, dev_path


def load_test_tsv(path: str | Path) -> pd.DataFrame:
    """
    Loads task4_test.tsv (no labels).
    Expected columns match SemEval release: par_id, keyword, country_code, text (and sometimes art_id).
    We'll just keep what exists.
    """
    path = Path(path)
    df = pd.read_csv(path, sep="\t")
    df.columns = [c.strip() for c in df.columns]
    if "text" in df.columns:
        df["text"] = df["text"].fillna("")
    return df