from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_selection import chi2


REPO = Path(__file__).resolve().parent
RAW = REPO / "data" / "raw"
FIG = REPO / "figures"
RES = REPO / "results"

FIG.mkdir(parents=True, exist_ok=True)
RES.mkdir(parents=True, exist_ok=True)


def load_data():
    df = pd.read_csv(
        RAW / "dontpatronizeme_pcl.tsv",
        sep="\t",
        skiprows=4,
        names=["par_id", "art_id", "keyword", "country", "text", "label"],
    )
    df["text"] = df["text"].fillna("")
    df["binary_label"] = (df["label"] >= 2).astype(int)
    df["token_len"] = df["text"].apply(lambda x: len(str(x).split()))
    return df


def eda_basic_stats(df):
    print("\n=== EDA 1: basic stats ===")

    counts = df["binary_label"].value_counts().sort_index()
    pos_rate = 100.0 * counts.get(1, 0) / len(df)

    print("\nClass counts (0=no PCL, 1=PCL):")
    print(counts.to_string())
    print(f"\nPCL rate: {pos_rate:.2f}%")

    desc = df["token_len"].describe(percentiles=[0.5, 0.9, 0.95, 0.99])
    print("\nToken length stats:")
    print(desc.to_string())

    plt.figure()
    sns.countplot(x="binary_label", data=df)
    plt.title("Class distribution")
    plt.xlabel("Class (0=no PCL, 1=PCL)")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(FIG / "eda_class_distribution.png", dpi=200)
    plt.close()

    plt.figure()
    sns.histplot(df["token_len"], bins=60)
    plt.title("Token lengths (all)")
    plt.xlabel("Tokens")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(FIG / "eda_token_length_hist.png", dpi=200)
    plt.close()

    plt.figure()
    sns.histplot(df[df["token_len"] <= 200]["token_len"], bins=60)
    plt.title("Token lengths (<=200)")
    plt.xlabel("Tokens")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(FIG / "eda_token_length_hist_zoom.png", dpi=200)
    plt.close()


def eda_keyword_and_bigrams(df):
    print("\n=== EDA 2: keyword + bigram signal ===")

    kw = df["keyword"].value_counts()
    print("\nKeyword counts:")
    print(kw.to_string())

    plt.figure(figsize=(10, 4))
    sns.barplot(x=kw.index, y=kw.values)
    plt.title("Paragraphs per keyword")
    plt.xlabel("Keyword")
    plt.ylabel("Count")
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig(FIG / "eda_keyword_distribution.png", dpi=200)
    plt.close()

    vec = CountVectorizer(stop_words="english", ngram_range=(2, 2), min_df=3)
    X0 = vec.fit_transform(df[df["binary_label"] == 0]["text"])
    ngrams = vec.get_feature_names_out()
    c0 = X0.sum(axis=0).A1

    vec2 = CountVectorizer(
        stop_words="english", ngram_range=(2, 2), min_df=3, vocabulary=ngrams
    )
    X1 = vec2.fit_transform(df[df["binary_label"] == 1]["text"])
    c1 = X1.sum(axis=0).A1

    f0 = c0 / max(c0.sum(), 1)
    f1 = c1 / max(c1.sum(), 1)

    eps = 1e-12
    ratio = (f1 + eps) / (f0 + eps)

    out = (
        pd.DataFrame(
            {
                "bigram": ngrams,
                "count_no_pcl": c0,
                "count_pcl": c1,
                "freq_no_pcl": f0,
                "freq_pcl": f1,
                "ratio_pcl_over_no_pcl": ratio,
            }
        )
        .sort_values("ratio_pcl_over_no_pcl", ascending=False)
        .reset_index(drop=True)
    )

    out_path = RES / "eda_bigram_salience.csv"
    out.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")

    top = out.head(20)
    plt.figure(figsize=(10, 4))
    sns.barplot(x=top["bigram"], y=top["ratio_pcl_over_no_pcl"])
    plt.title("Top bigrams (PCL vs No PCL ratio)")
    plt.xlabel("Bigram")
    plt.ylabel("Ratio")
    plt.xticks(rotation=60, ha="right")
    plt.tight_layout()
    plt.savefig(FIG / "eda_top_pcl_bigrams.png", dpi=200)
    plt.close()


def eda_pcl_rate_by_keyword(df):
    print("\n=== Extra EDA: PCL rate by keyword ===")

    g = (
        df.groupby("keyword")["binary_label"]
        .agg(["count", "sum", "mean"])
        .rename(columns={"sum": "pcl_count", "mean": "pcl_rate"})
        .sort_values("pcl_rate", ascending=False)
        .reset_index()
    )

    out_path = RES / "eda_pcl_rate_by_keyword.csv"
    g.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")

    print("\nTop keywords by PCL rate:")
    print(g.head(10).to_string(index=False))

    plt.figure(figsize=(10, 4))
    sns.barplot(x=g["keyword"], y=g["pcl_rate"])
    plt.title("PCL rate by keyword")
    plt.xlabel("Keyword")
    plt.ylabel("PCL rate")
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig(FIG / "eda_pcl_rate_by_keyword.png", dpi=200)
    plt.close()


def eda_discriminative_unigrams(df, top_k=20):
    print("\n=== Extra EDA: discriminative unigrams (TF-IDF + chi2) ===")

    y = df["binary_label"].values
    vec = TfidfVectorizer(stop_words="english", min_df=3, ngram_range=(1, 1))
    X = vec.fit_transform(df["text"])
    feats = vec.get_feature_names_out()

    chi_vals, _ = chi2(X, y)
    chi_df = pd.DataFrame({"token": feats, "chi2": chi_vals}).sort_values(
        "chi2", ascending=False
    )

    out_path = RES / "eda_top_unigrams_chi2.csv"
    chi_df.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")

    pcl_mask = y == 1
    mean_pcl = X[pcl_mask].mean(axis=0).A1
    mean_nopcl = X[~pcl_mask].mean(axis=0).A1

    top = chi_df.head(300).copy()
    idx = [vec.vocabulary_[t] for t in top["token"].tolist()]
    top["mean_tfidf_pcl"] = mean_pcl[idx]
    top["mean_tfidf_no_pcl"] = mean_nopcl[idx]
    top["delta_pcl_minus_no_pcl"] = top["mean_tfidf_pcl"] - top["mean_tfidf_no_pcl"]

    top_pcl = top.sort_values("delta_pcl_minus_no_pcl", ascending=False).head(top_k)
    top_no = top.sort_values("delta_pcl_minus_no_pcl", ascending=True).head(top_k)

    print("\nTop tokens leaning PCL:")
    print(top_pcl[["token", "delta_pcl_minus_no_pcl"]].to_string(index=False))

    print("\nTop tokens leaning No PCL:")
    print(top_no[["token", "delta_pcl_minus_no_pcl"]].to_string(index=False))

    plt.figure(figsize=(10, 4))
    sns.barplot(x=top_pcl["token"], y=top_pcl["delta_pcl_minus_no_pcl"])
    plt.title("Top tokens leaning PCL (TF-IDF delta)")
    plt.xlabel("Token")
    plt.ylabel("Mean TF-IDF(PCL) - Mean TF-IDF(No PCL)")
    plt.xticks(rotation=60, ha="right")
    plt.tight_layout()
    plt.savefig(FIG / "eda_top_tokens_pcl.png", dpi=200)
    plt.close()

    plt.figure(figsize=(10, 4))
    sns.barplot(x=top_no["token"], y=-top_no["delta_pcl_minus_no_pcl"])
    plt.title("Top tokens leaning No PCL (TF-IDF delta)")
    plt.xlabel("Token")
    plt.ylabel("Mean TF-IDF(No PCL) - Mean TF-IDF(PCL)")
    plt.xticks(rotation=60, ha="right")
    plt.tight_layout()
    plt.savefig(FIG / "eda_top_tokens_no_pcl.png", dpi=200)
    plt.close()

    both_out = pd.concat(
        [
            top_pcl.assign(side="pcl").rename(columns={"delta_pcl_minus_no_pcl": "delta"}),
            top_no.assign(side="no_pcl").rename(columns={"delta_pcl_minus_no_pcl": "delta"}),
        ],
        ignore_index=True,
    )
    both_out_path = RES / "eda_top_tokens_delta.csv"
    both_out.to_csv(both_out_path, index=False)
    print(f"Saved: {both_out_path}")


def quick_noise_checks(df):
    print("\n=== Quick checks (just in case) ===")
    print("Exact duplicate texts:", int(df["text"].duplicated().sum()))
    print("Very short (<=3 tokens):", int((df["token_len"] <= 3).sum()))
    print("Very long (>=300 tokens):", int((df["token_len"] >= 300).sum()))
    print(
        "HTML/entity/newline-ish:",
        int(df["text"].str.contains(r"&amp;|&lt;|&gt;|<br>|\\n", regex=True).sum()),
    )


def main():
    df = load_data()
    print("Loaded rows:", len(df))

    eda_basic_stats(df)
    eda_keyword_and_bigrams(df)
    eda_pcl_rate_by_keyword(df)
    eda_discriminative_unigrams(df, top_k=20)
    quick_noise_checks(df)

    print("\nFigures saved in:", FIG)
    print("Tables saved in:", RES)


if __name__ == "__main__":
    main()