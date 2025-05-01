# Fake News Detection and Source Credibility Analysis

## Goals

Fake news undermines public trust and can sway social, political and economic outcomes.  Our goal is to build a *transparent* pipeline that

1. Accurately classifies news articles as **Real** or **Fake**.
2. Provides **explanatory signals** (named-entity counts, headline-body stance) so journalists can inspect why an article was flagged.
3. Studies **source credibility** across multiple corpora (LIAR, GossipCop) to understand domain transfer.

## Methods

* **Data** Kaggle ["Fake News Detection"](https://www.kaggle.com/datasets/emineyetm/fake-news-detection-datasets) corpus (12.6 k real Reuters + 12.6 k fake articles).  80 : 20 split into train/test.  Additional [LIAR](https://www.kaggle.com/datasets/khandalaryan/liar-preprocessed-dataset) & [GossipCop](https://www.kaggle.com/datasets/subodh7300/gossipcop) sets are used for a credibility probe.
* **Pre-processing** spaCy tokenisation & lemmatisation → `processed_text`, `processed_title`.
* **Embeddings** `bert-base-uncased` frozen CLS vectors (768-D) 
* **NER features** `dbmdz/bert-base-cased-conll03` counts of PER / ORG / LOC / MISC tokens (4-D)
* **Headline-Body Stance** Zero-shot `roberta-large-mnli`; one-hot \[contradicts, supports\] appended as 2-D feature.
* **Classifier** PCA (100 comps) → Logistic Regression.  For credibility we train a linear SVM and a small 2-layer MLP on CLS embeddings.
* **Caching** All heavy tensors (`*.npy`) are cached; trained heads are saved as `bert_ner_model.pkl` and `bert_ner_stance_model.pkl` (≤1 MB each).

## Using this repository

All project tasks are exposed through the consolidated command-line interface in `nlp/cli.py`:

```bash
python -m nlp.cli <command> [options]
```

| Command | Description |
|---------|-------------|
| `dl-data` | Download the corpora (Fake/True, GossipCop, LIAR) from Kaggle into `nlp/data/`. Requires a valid `~/.kaggle/kaggle.json`. |
| `preprocess` | Clean & split the Fake/True news set with spaCy; saves train/test splits to `nlp/data/preprocessed/`. |

Standalone research scripts:

* **BERT + NER evaluation** – frozen CLS embeddings, optional stance feature
  ```bash
  python -m nlp.BERT_model_eval \
      --data_dir nlp/data/preprocessed \
      --device auto            # auto / cuda / mps / cpu
  # Extra flags: --with_stance  --add_stance_feat  --skip_source
  ```

Available flags:
- `--data_dir`: Directory containing preprocessed data (default: `data/preprocessed`)
- `--device`: Compute device for NER/stance models
  - `auto`: Use CUDA if available, else MPS, else CPU
  - `cuda`: NVIDIA GPU (if available)
  - `mps`: Apple Silicon GPU (if available)
  - `cpu`: Force CPU execution
- `--with_stance`: Run stance detection between headline and article body
- `--add_stance_feat`: Add stance as a feature for classification (takes ~2.5h on first run)
- `--skip_source`: Skip LIAR+GossipCop source credibility analysis
- `--pca_dim`: Number of PCA components (default: 100)
- `--max_iter`: Max iterations for LogisticRegression (default: 1000)

Recommended command for full evaluation (includes stance features and analysis):
```bash
python -m nlp.BERT_model_eval \
    --data_dir nlp/data/preprocessed \
    --device auto \
    --with_stance \
    --add_stance_feat
```

* **Word2Vec + LSTM pipeline**
  1. Feature extraction:    `python -m nlp.LSTM.word2vec_feature_extraction`
  2. Enhanced BiLSTM:       `python -m nlp.LSTM.lstm_model`
  3. Simplified BiLSTM:     `python -m nlp.LSTM.simplified_lstm_model`

### Quick-start (fresh machine)

```bash
# 1. (Optional) create and activate a virtual environment
python -m venv .venv && source .venv/bin/activate

# 2. Install Python dependencies
pip install -r requirements.txt

# 3. Add your Kaggle API token so the downloader works
mkdir -p ~/.kaggle && cp <path>/kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json

# 4. Fetch raw data
python -m nlp.cli dl-data

# 5. Pre-process & split
python -m nlp.cli preprocess

# 6. Extract BERT CLS embeddings (≈8‒10 min)
cd nlp/helper_script && python feature_extraction.py && cd -

# 7. Run the main evaluation (builds NER & stance caches on first run)
python -m nlp.BERT_model_eval --device auto
```

**Note:** For convenience, the repository includes pre-computed caches:
- NER counts (`train_ner.npy`, `test_ner.npy`)
- Stance predictions (`test_stance.npy`)

**Note:** BERT CLS embeddings (`train_features.pt`, `test_features.pt`) are not pre-stored. You must run `feature_extraction.py` to generate them.

If you want to perform a completely fresh run (e.g., to regenerate all caches), expect the following times on MPS:
- NER feature extraction: ~9 min (train: 7:21, test: 1:47)
- Full stance detection: ~2.5 hours (train: 2:04:32, test: 30:55)

## Contents

- [docs](docs): template to create slides for project presentations
- [nlp](nlp): Python project code
- [notebooks](notebooks): Jupyter notebooks for project development and experimentation
- [report](report): LaTeX report
- [tests](tests): unit tests for project code

## Conclusions

* **BERT + NER** achieves **94 % accuracy / 0.99 ROC-AUC** on the held-out test set—on par with a TF-IDF logistic baseline but with better calibration.
* Adding the stance one-hot lifts accuracy to **94.3 %** and F1 to **0.946**, confirming headline-body contradictions provide complementary signal.
* Source-credibility transfer is hard: the linear SVM scores ~0.60 accuracy on LIAR and 0.59 on GossipCop; the MLP head boosts GossipCop accuracy but lowers F1/AUC.  Frozen CLS embeddings need domain-specific fine-tuning for robust cross-source performance.
* The full pipeline—including NER, stance and caching—runs in <10 min on MPS once caches are built, and subsequent runs load instantly.

See `report/report.tex` for further details.
