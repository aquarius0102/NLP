# PCL Detection with RoBERTa

## Overview

This repository contains an implementation for **SemEval Task 4: Patronizing and Condescending Language (PCL) Detection**.

The model is a **RoBERTa-based binary classifier** fine-tuned on the provided dataset to detect patronizing or condescending language.

The repository includes:

- Training code
- Prediction scripts
- A trained model checkpoint
- Development and test predictions used for submission

---

# Repository Structure

    .
    ├── BestModel/
    │   ├── train.py
    │   ├── predict.py
    │   ├── data.py
    │   ├── modeling.py
    │   ├── utils.py
    │   ├── threshold_tune.py
    │   └── artifacts/
    │       └── best_checkpoint/
    │           ├── pytorch_model.bin
    │           ├── threshold.json
    │           └── meta.json
    │
    ├── data/
    │   └── raw/
    │       ├── dontpatronizeme_pcl.tsv
    │       ├── dontpatronizeme_categories.tsv
    │       ├── train_semeval_parids-labels.csv
    │       ├── dev_semeval_parids-labels.csv
    │       └── task4_test.tsv
    │
    ├── dev.txt
    ├── test.txt
    ├── requirements.txt
    └── README.md

---

# Environment Setup

Install dependencies:

    pip install -r requirements.txt

Python **3.10+** is recommended.

---

# Training the Model

Run the following command to train the model:

    python3 -m BestModel.train \
      --tsv data/raw/dontpatronizeme_pcl.tsv \
      --raw_dir data/raw \
      --epochs 3 \
      --batch_size 16 \
      --max_length 128

The best model checkpoint will be saved to:

    BestModel/artifacts/best_checkpoint/

The model used in this submission achieved:

    Best dev F1: 0.5877

---

# Generating Development Predictions

Run:

    python3 -m BestModel.predict \
      --tsv data/raw/dontpatronizeme_pcl.tsv \
      --raw_dir data/raw \
      --split_csv dev_semeval_parids-labels.csv \
      --ckpt_dir BestModel/artifacts/best_checkpoint \
      --threshold 0.380 \
      --out_csv dev_preds.csv

Convert predictions to the required submission format:

    cut -d',' -f3 dev_preds.csv | tail -n +2 > dev.txt

---

# Generating Test Predictions

Run:

    python3 -m BestModel.predict \
      --raw_dir data/raw \
      --test_tsv task4_test.tsv \
      --ckpt_dir BestModel/artifacts/best_checkpoint \
      --threshold 0.380 \
      --out_csv test_preds.csv

Convert predictions to the submission format:

    cut -d',' -f2 test_preds.csv | tail -n +2 > test.txt

---

# Submission Files

The repository includes the required submission outputs:

    dev.txt
    test.txt

Each file contains **one prediction per line (0 or 1)**.

---

# Model Details

- Backbone: `roberta-base`
- Maximum sequence length: 128
- Batch size: 16
- Training epochs: 3
- Optimizer: AdamW
- Loss function: Binary Cross Entropy
- Decision threshold: 0.380

---

# Notes for Marker

All code required to reproduce training and predictions is located in the `BestModel/` directory.

The included checkpoint allows predictions to be generated directly without retraining.

Important files:

| File | Description |
|-----|-----|
| train.py | Model training pipeline |
| predict.py | Generates predictions for dev/test |
| data.py | Dataset loading and preprocessing |
| modeling.py | Model architecture |
| threshold_tune.py | Threshold tuning utilities |

The repository contains both the **source code** and the **generated prediction files** required for submission.
