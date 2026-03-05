from pathlib import Path
import json

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


REPO = Path(__file__).resolve().parent
RAW = REPO / "data" / "raw"
FIG = REPO / "figures"
RES = REPO / "results"

FIG.mkdir(exist_ok=True)
RES.mkdir(exist_ok=True)


def load_dataset():
    df = pd.read_csv(
        RAW / "dontpatronizeme_pcl.tsv",
        sep="\t",
        skiprows=4,
        names=["par_id", "art_id", "keyword", "country", "text", "label"],
    )
    df["text"] = df["text"].fillna("")
    df["par_id"] = df["par_id"].astype(int)
    df["label"] = df["label"].astype(int)
    df["binary_label"] = (df["label"] >= 2).astype(int)
    return df


def load_ids(path):
    df = pd.read_csv(path)

    cols = {c.lower().strip(): c for c in df.columns}
    if "par_id" not in cols:
        raise ValueError(f"{path} needs a par_id column")

    ids = df[cols["par_id"]].astype(int).tolist()
    return set(ids)


def make_splits(df):
    train_ids = load_ids(RAW / "train_semeval_parids-labels.csv")
    dev_ids = load_ids(RAW / "dev_semeval_parids-labels.csv")

    train = df[df["par_id"].isin(train_ids)].copy()
    dev = df[df["par_id"].isin(dev_ids)].copy()

    return train, dev


def train_model(X_train, y_train):
    pipe = Pipeline(
        [
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.95)),
            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced")),
        ]
    )
    pipe.fit(X_train, y_train)
    return pipe


def evaluate(model, X, y):
    preds = model.predict(X)

    metrics = {
        "accuracy": float(accuracy_score(y, preds)),
        "precision": float(precision_score(y, preds, zero_division=0)),
        "recall": float(recall_score(y, preds, zero_division=0)),
        "f1": float(f1_score(y, preds, zero_division=0)),
    }

    return preds, metrics


def save_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["no PCL", "PCL"],
    )

    disp.plot(values_format="d")
    plt.title("Baseline confusion matrix")
    plt.tight_layout()
    plt.savefig(FIG / "baseline_confusion_matrix.png", dpi=200)
    plt.close()


def main():
    df = load_dataset()
    train, dev = make_splits(df)

    print("Train size:", len(train))
    print("Dev size:", len(dev))
    print("Train PCL rate:", round(train["binary_label"].mean(), 4))
    print("Dev PCL rate:", round(dev["binary_label"].mean(), 4))

    X_train = train["text"].tolist()
    y_train = train["binary_label"].tolist()

    X_dev = dev["text"].tolist()
    y_dev = dev["binary_label"].tolist()

    model = train_model(X_train, y_train)
    preds, metrics = evaluate(model, X_dev, y_dev)

    print("\nDev metrics")
    for k, v in metrics.items():
        print(k, round(v, 4))

    save_confusion_matrix(y_dev, preds)

    pd.DataFrame(
        {
            "par_id": dev["par_id"].astype(int),
            "gold": y_dev,
            "pred": preds,
            "keyword": dev["keyword"].astype(str),
        }
    ).to_csv(RES / "baseline_dev_preds.csv", index=False)

    with open(RES / "baseline_dev_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("\nSaved results to results/ and figures/")


if __name__ == "__main__":
    main()