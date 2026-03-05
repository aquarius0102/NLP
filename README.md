# NLP (SemEval 2022 Task 4: PCL Detection)

Binary classification: PCL vs No PCL using the Don't Patronize Me dataset.

## Folder structure

- data/raw/ : original files (don't edit)
- data/processed/ : cleaned/split csvs saved from preprocessing
- notebooks/ : analysis notebooks (I used notebooks/notebook.ipynb)
- src/ : reusable python code
- figures/ : plots for report
- results/ : metrics, predictions, etc.

## Setup

From repo root:

```bash
python -m venv .venv
source .venv/bin/activate   # (Windows: .venv\Scripts\activate)
pip install -r requirements.txt
python -m ipykernel install --user --name nlp --display-name "NLP"