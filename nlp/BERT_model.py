# ================================
#   Disable TensorFlow inside Transformers (forces PyTorch backend)
# ================================
import os
os.environ["TRANSFORMERS_NO_TF"] = "1"  # must be set before importing transformers

import argparse
import sys
import warnings

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from transformers import pipeline
import matplotlib.pyplot as plt
from tqdm import tqdm  # Progress bar

# -----------------------------
# Utility functions
# -----------------------------

def extract_entity_features(texts, ner_pipe):
    """Return a NumPy array with PER/ORG/LOC/MISC counts for *texts*.

    Parameters
    ----------
    texts : Iterable[str]
        Raw text documents.
    ner_pipe : transformers.pipelines.NamedEntityRecognitionPipeline
        Pre-initialised Hugging Face NER pipeline.
    """
    features = []
    for text in tqdm(texts, desc="NER", unit="doc"):
        if not isinstance(text, str) or not text.strip():
            features.append([0, 0, 0, 0])
            continue
        ents = ner_pipe(text[:512])  # BERT token limit safeguard
        counts = {"PER": 0, "ORG": 0, "LOC": 0, "MISC": 0}
        for ent in ents:
            label = ent.get("entity_group")
            if label in counts:
                counts[label] += 1
        features.append([counts["PER"], counts["ORG"], counts["LOC"], counts["MISC"]])
    return np.asarray(features, dtype=np.float32)


# -----------------------------
# Main routine
# -----------------------------

def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description="Fake-news detection with BERT embeddings and NER features")
    parser.add_argument(
        "--data_dir",
        default="data/preprocessed",
        help="Directory containing train_features.pt, test_features.pt, train.csv, test.csv",
    )
    parser.add_argument(
        "--cuda",
        action="store_true",
        help="Use CUDA device for NER inference if available (defaults to CPU). This only affects the pipeline step; the rest of the script works on CPUs.",
    )
    args = parser.parse_args(argv)

    data_dir = args.data_dir.rstrip("/")

    # ---------------
    # Load features & labels
    # ---------------
    print("Loading pre-computed BERT embeddings …", file=sys.stderr)
    train_features = torch.load(f"{data_dir}/train_features.pt", map_location="cpu").to(torch.float32)
    test_features = torch.load(f"{data_dir}/test_features.pt", map_location="cpu").to(torch.float32)

    print("Loading CSV label files …", file=sys.stderr)
    train_df = pd.read_csv(f"{data_dir}/train.csv")
    test_df = pd.read_csv(f"{data_dir}/test.csv")

    # Sanity-check columns
    required_col = "processed_text"
    if required_col not in train_df.columns:
        raise ValueError(f"Expected column '{required_col}' in train.csv; found {train_df.columns.tolist()}")

    # ---------------
    # NER feature extraction
    # ---------------
    device = 0 if args.cuda and torch.cuda.is_available() else -1
    print(f"Initialising NER pipeline on device {device} …", file=sys.stderr)
    ner_pipe = pipeline(
        "ner",
        model="dbmdz/bert-base-cased-finetuned-conll03-english",
        tokenizer="dbmdz/bert-base-cased-finetuned-conll03-english",
        aggregation_strategy="simple",
        framework="pt",
        device=device,
    )

    print("Extracting NER features (train set) …", file=sys.stderr)
    train_ner = extract_entity_features(train_df["processed_text"].tolist(), ner_pipe)
    print("Extracting NER features (test set) …", file=sys.stderr)
    test_ner = extract_entity_features(test_df["processed_text"].tolist(), ner_pipe)

    # ---------------
    # Combine BERT + NER features
    # ---------------
    X_train_combo = np.concatenate([train_features.numpy(), train_ner], axis=1)
    X_test_combo = np.concatenate([test_features.numpy(), test_ner], axis=1)

    # ---------------
    # Dimensionality reduction
    # ---------------
    print("Applying PCA (BERT+NER) …", file=sys.stderr)
    pca = PCA(n_components=100)
    X_train_pca = pca.fit_transform(X_train_combo)
    X_test_pca = pca.transform(X_test_combo)

    # ---------------
    # Label preparation
    # ---------------
    y_train = train_df["label"].fillna(0).values
    y_test = test_df["label"].fillna(0).values

    # ---------------
    # Logistic Regression on combined features
    # ---------------
    print("Training Logistic Regression (BERT+NER) …", file=sys.stderr)
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train_pca, y_train)
    y_pred = clf.predict(X_test_pca)

    # Evaluation
    print("\n=== Evaluation: Logistic Regression (BERT + NER) ===")
    print(f"Accuracy : {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall   : {recall_score(y_test, y_pred):.4f}")
    print(f"F1 Score : {f1_score(y_test, y_pred):.4f}\n")
    print(classification_report(y_test, y_pred, target_names=["REAL", "FAKE"]))

    y_prob = clf.predict_proba(X_test_pca)[:, 1]
    roc_auc = roc_auc_score(y_test, y_prob)
    print(f"ROC-AUC (BERT+NER): {roc_auc:.4f}\n")

    # ROC curve plot
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"BERT+NER (area={roc_auc:.4f})", color="tab:blue")
    plt.plot([0, 1], [0, 1], linestyle="--", color="grey")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve – BERT Embeddings + NER Features")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.show()

    # ---------------
    # Baseline: BERT embeddings only
    # ---------------
    print("\nComputing BERT-only baseline …", file=sys.stderr)
    pca_base = PCA(n_components=100)
    X_train_base = pca_base.fit_transform(train_features.numpy())
    X_test_base = pca_base.transform(test_features.numpy())

    clf_base = LogisticRegression(max_iter=1000)
    clf_base.fit(X_train_base, y_train)
    y_pred_base = clf_base.predict(X_test_base)

    print("\n=== Evaluation: BERT Only (Baseline) ===")
    print(f"Accuracy : {accuracy_score(y_test, y_pred_base):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred_base):.4f}")
    print(f"Recall   : {recall_score(y_test, y_pred_base):.4f}")
    print(f"F1 Score : {f1_score(y_test, y_pred_base):.4f}\n")
    print(classification_report(y_test, y_pred_base, target_names=["REAL", "FAKE"]))

    y_prob_base = clf_base.predict_proba(X_test_base)[:, 1]
    roc_auc_base = roc_auc_score(y_test, y_prob_base)
    print(f"ROC-AUC (BERT only): {roc_auc_base:.4f}\n")

    fpr_b, tpr_b, _ = roc_curve(y_test, y_prob_base)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr_b, tpr_b, label=f"BERT Only (area={roc_auc_base:.4f})", color="tab:green")
    plt.plot([0, 1], [0, 1], linestyle="--", color="grey")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve – BERT Only Baseline")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)
    main() 