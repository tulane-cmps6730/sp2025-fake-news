#!/usr/bin/env python
"""BERT_model_eval.py – command-line evaluation of fake-news detection
using frozen BERT CLS embeddings ± NER counts.

Run:
    python -m nlp.BERT_model_eval --data_dir data/preprocessed --device auto
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
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
from tqdm.auto import tqdm
from sklearn.model_selection import cross_val_score, cross_val_predict
import string
import nltk
from nltk.corpus import stopwords
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, label_binarize
import gc
from transformers import BertTokenizer, BertModel
from sklearn.neural_network import MLPClassifier
import joblib

# Stance detection helpers
from nlp.stance_detection import (
    load_classifier as load_stance_clf,
    batch_predict as stance_batch_predict,
)

STANCE_LABELS = ["contradicts", "supports", "neutral"]

# ---------------------------------------------------------------------------
# Utility: NER-count extractor (PER / ORG / LOC / MISC)
# ---------------------------------------------------------------------------


def extract_entity_features(texts: list[str], ner_pipe) -> np.ndarray:  # noqa: D401
    """Return a float32 (n_docs, 4) array of simple NER token counts."""
    features: list[list[int]] = []
    for text in tqdm(texts, desc="NER", unit="doc"):
        if not isinstance(text, str) or not text.strip():
            features.append([0, 0, 0, 0])
            continue
        ents = ner_pipe(text[:512])  # BERT 512-token guard
        counts = {"PER": 0, "ORG": 0, "LOC": 0, "MISC": 0}
        for ent in ents:
            label = ent.get("entity_group")
            if label in counts:
                counts[label] += 1
        features.append([counts[c] for c in ("PER", "ORG", "LOC", "MISC")])
    return np.asarray(features, dtype=np.float32)


# ---------------------------------------------------------------------------
# Device helper (CPU / CUDA / MPS)
# ---------------------------------------------------------------------------


def _select_device(choice: str = "auto") -> int | str:
    choice = choice.lower()
    if choice == "cuda":
        return 0 if torch.cuda.is_available() else -1
    if choice == "mps":
        return "mps" if torch.backends.mps.is_available() else -1
    if choice == "cpu":
        return -1
    # auto mode
    if torch.cuda.is_available():
        return 0
    if torch.backends.mps.is_available():
        return "mps"
    return -1


# ---------------------------------------------------------------------------
# Main routine
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:  # noqa: D401
    parser = argparse.ArgumentParser(
        description="Evaluate logistic regression on BERT embeddings with optional NER counts.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data_dir",
        default="data/preprocessed",
        help="Folder containing train_features.pt, test_features.pt, train.csv, test.csv",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda", "mps"],
        default="auto",
        help="Compute device for the Hugging Face NER pipeline.",
    )
    parser.add_argument(
        "--pca_dim", type=int, default=100, help="Number of PCA components."
    )
    parser.add_argument(
        "--max_iter",
        type=int,
        default=1000,
        help="Max iterations for LogisticRegression.",
    )
    parser.add_argument(
        "--with_stance",
        action="store_true",
        help="Run stance-detection validation on processed_title vs processed_text.",
    )
    parser.add_argument(
        "--skip_source",
        action="store_true",
        help="Skip LIAR+GossipCop source-credibility analysis.",
    )
    parser.add_argument(
        "--add_stance_feat",
        action="store_true",
        help="Append one-hot stance (headline vs body) to feature vector and evaluate a second classifier.",
    )
    args = parser.parse_args(argv)

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        sys.exit(f"error: data_dir not found → {data_dir}")

    # -------------------- 1. Load BERT features --------------------
    print("Loading BERT feature tensors …", file=sys.stderr)
    train_features = torch.load(data_dir / "train_features.pt", map_location="cpu").to(
        torch.float32
    )
    test_features = torch.load(data_dir / "test_features.pt", map_location="cpu").to(
        torch.float32
    )

    # -------------------- 2. Load CSV splits -----------------------
    train_df = pd.read_csv(data_dir / "train.csv")
    test_df = pd.read_csv(data_dir / "test.csv")

    # -------------------- 3. NER pipeline --------------------------
    device_id = _select_device(args.device)
    print(f"Initialising NER pipeline on device={device_id} …", file=sys.stderr)
    ner_pipe = pipeline(
        "ner",
        model="dbmdz/bert-base-cased-finetuned-conll03-english",
        tokenizer="dbmdz/bert-base-cased-finetuned-conll03-english",
        aggregation_strategy="simple",
        framework="pt",
        device=device_id,
    )

    # -------------------- 3a. Cache NER counts ----------------------

    def _cached_ner_counts(texts: pd.Series, split_name: str) -> np.ndarray:
        """Load pre-computed NER counts from .npy if present, else compute and cache."""
        cache_path = data_dir / f"{split_name}_ner.npy"
        if cache_path.exists():
            print(f"Loaded cached NER counts → {cache_path.name}", file=sys.stderr)
            return np.load(cache_path)

        print(f"Computing NER counts for {split_name} …", file=sys.stderr)
        arr = extract_entity_features(texts.tolist(), ner_pipe)
        np.save(cache_path, arr)
        print(f"Saved NER counts cache → {cache_path.name}", file=sys.stderr)
        return arr

    # -------------------- 4. NER counts (cached) --------------------
    train_ner = _cached_ner_counts(train_df["processed_text"], "train")
    test_ner = _cached_ner_counts(test_df["processed_text"], "test")

    # -------------------- 5. Combine features ---------------------
    X_train_combo = np.concatenate([train_features.numpy(), train_ner], axis=1)
    X_test_combo = np.concatenate([test_features.numpy(), test_ner], axis=1)

    # -------------------- 6. PCA + LR (BERT+NER) ------------------
    pca = PCA(n_components=args.pca_dim, random_state=42)
    X_train_pca = pca.fit_transform(X_train_combo)
    X_test_pca = pca.transform(X_test_combo)

    clf = LogisticRegression(max_iter=args.max_iter, C=0.1, random_state=42)
    clf.fit(X_train_pca, train_df["label"].fillna(0).values)
    y_pred = clf.predict(X_test_pca)

    y_test = test_df["label"].fillna(0).values

    print("\n=== Evaluation: Logistic Regression (BERT + NER) ===")
    print(f"Accuracy : {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall   : {recall_score(y_test, y_pred):.4f}")
    print(f"F1 Score : {f1_score(y_test, y_pred):.4f}\n")
    print(classification_report(y_test, y_pred, target_names=["REAL", "FAKE"]))

    # -------------------- Save trained PCA + LR -------------------
    model_path = data_dir / "bert_ner_model.pkl"
    joblib.dump({"pca": pca, "clf": clf}, model_path)
    print(f"Saved model → {model_path}", file=sys.stderr)

    y_prob = clf.predict_proba(X_test_pca)[:, 1]
    roc_auc = roc_auc_score(y_test, y_prob)
    fpr, tpr, _ = roc_curve(y_test, y_prob)

    # -------------------- 6b. Cross-validation --------------------
    print("\n--- 5-Fold Cross-Validation (training split) ---")
    cv_scores = cross_val_score(
        clf, X_train_pca, train_df["label"].values, cv=5, scoring="accuracy"
    )
    print(f"Fold accuracies : {cv_scores}")
    print(f"Mean accuracy   : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    y_pred_cv = cross_val_predict(clf, X_train_pca, train_df["label"].values, cv=5)
    f1_cv = f1_score(train_df["label"].values, y_pred_cv)
    roc_auc_cv = roc_auc_score(
        train_df["label"].values, clf.predict_proba(X_train_pca)[:, 1]
    )
    print(f"F1 (CV)         : {f1_cv:.4f}")
    print(f"ROC-AUC (CV)    : {roc_auc_cv:.4f}\n")

    # -------------------- 7. Baseline (BERT-only) ------------------
    pca_base = PCA(n_components=args.pca_dim, random_state=42)
    X_train_base = pca_base.fit_transform(train_features.numpy())
    X_test_base = pca_base.transform(test_features.numpy())

    clf_base = LogisticRegression(max_iter=args.max_iter, random_state=42)
    clf_base.fit(X_train_base, train_df["label"].values)
    y_pred_base = clf_base.predict(X_test_base)

    print("\n=== Evaluation: BERT Only (Baseline) ===")
    print(f"Accuracy : {accuracy_score(y_test, y_pred_base):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred_base):.4f}")
    print(f"Recall   : {recall_score(y_test, y_pred_base):.4f}")
    print(f"F1 Score : {f1_score(y_test, y_pred_base):.4f}\n")
    print(classification_report(y_test, y_pred_base, target_names=["REAL", "FAKE"]))

    y_prob_base = clf_base.predict_proba(X_test_base)[:, 1]
    roc_auc_base = roc_auc_score(y_test, y_prob_base)
    fpr_b, tpr_b, _ = roc_curve(y_test, y_prob_base)

    # -------------------- 8. Plot ROC curves -----------------------
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"BERT+NER (area={roc_auc:.4f})", color="tab:blue")
    plt.plot(
        fpr_b, tpr_b, label=f"BERT Only (area={roc_auc_base:.4f})", color="tab:green"
    )
    plt.plot([0, 1], [0, 1], linestyle="--", color="grey")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve – Logistic Regression vs. Baseline")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.show()

    # -------------------- 9. Source-Credibility Analysis ----------
    if not args.skip_source:
        run_source_credibility_analysis()

    # -------------------- 10. Optional stance detection -----------
    if args.with_stance:
        run_stance_validation(test_df, device=args.device, cache_dir=data_dir)

    # -------------------- 7b. Add stance feature evaluation ------
    if args.add_stance_feat:
        print("\n=== Building features with stance one-hot ===", file=sys.stderr)

        train_stance_feat = _cached_stance_onehot(train_df, "train", data_path=data_dir, device=args.device)
        test_stance_feat = _cached_stance_onehot(test_df, "test", data_path=data_dir, device=args.device)

        X_train_combo_s = np.concatenate([train_features.numpy(), train_ner, train_stance_feat], axis=1)
        X_test_combo_s = np.concatenate([test_features.numpy(), test_ner, test_stance_feat], axis=1)

        pca_s = PCA(n_components=args.pca_dim, random_state=42)
        X_train_pca_s = pca_s.fit_transform(X_train_combo_s)
        X_test_pca_s = pca_s.transform(X_test_combo_s)

        clf_s = LogisticRegression(max_iter=args.max_iter, C=0.1, random_state=42)
        clf_s.fit(X_train_pca_s, train_df["label"].fillna(0).values)
        y_pred_s = clf_s.predict(X_test_pca_s)

        print("\n=== Evaluation: BERT + NER + Stance ===")
        print(f"Accuracy : {accuracy_score(y_test, y_pred_s):.4f}")
        print(f"Precision: {precision_score(y_test, y_pred_s):.4f}")
        print(f"Recall   : {recall_score(y_test, y_pred_s):.4f}")
        print(f"F1 Score : {f1_score(y_test, y_pred_s):.4f}\n")
        print(classification_report(y_test, y_pred_s, target_names=["REAL", "FAKE"]))

        stance_model_path = data_dir / "bert_ner_stance_model.pkl"
        joblib.dump({"pca": pca_s, "clf": clf_s}, stance_model_path)
        print(f"Saved stance model → {stance_model_path}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Source-Credibility: LIAR + GossipCop evaluation
# ---------------------------------------------------------------------------


def run_source_credibility_analysis() -> None:
    """Evaluate an SVM on LIAR & GossipCop statements using fresh BERT CLS embeddings."""

    print(
        "\n===== Source-Credibility Analysis (LIAR & GossipCop) =====", file=sys.stderr
    )

    nltk.download("stopwords", quiet=True)
    stop_words = set(stopwords.words("english"))

    data_root = Path("nlp/data")

    # Load datasets
    gossip_df = pd.read_csv(data_root / "gossipcop.csv")

    liar_dir = data_root / "liar_dataset"
    train_liar = pd.read_csv(
        liar_dir / "train.tsv",
        sep="\t",
        header=None,
        names=[
            "ID",
            "Label",
            "Statement",
            "Subjects",
            "Speaker",
            "SpeakerJob",
            "State",
            "PartyAffiliation",
            "BarelyTrueCount",
            "FalseCount",
            "HalfTrueCount",
            "MostlyTrueCount",
            "PantsOnFireCount",
            "Context",
        ],
    )
    test_liar = pd.read_csv(
        liar_dir / "test.tsv", sep="\t", header=None, names=train_liar.columns
    )
    valid_liar = pd.read_csv(
        liar_dir / "valid.tsv", sep="\t", header=None, names=train_liar.columns
    )

    # Drop rows without statements
    for df in (train_liar, test_liar, valid_liar):
        df.dropna(subset=["Statement"], inplace=True)

    # Simple cleaning
    def clean(text: str) -> str:
        text = text.lower().translate(str.maketrans("", "", string.punctuation))
    return text

    train_liar["clean"] = train_liar["Statement"].map(clean)
    test_liar["clean"] = test_liar["Statement"].map(clean)
    valid_liar["clean"] = valid_liar["Statement"].map(clean)
    gossip_df["clean"] = gossip_df["text"].map(clean)

    # BERT tokenizer / model (CPU for safety)
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    # Prefer GPU/MPS if present
    device = (
        0
        if torch.cuda.is_available()
        else ("mps" if torch.backends.mps.is_available() else -1)
    )
    model = BertModel.from_pretrained("bert-base-uncased")
    if device != -1:
        model = model.to(device)

    def bert_embed(texts: list[str], batch_size: int = 32) -> np.ndarray:
        """Return CLS embeddings batched for speed."""
        vecs: list[np.ndarray] = []
        for i in tqdm(range(0, len(texts), batch_size), desc="BERT", unit="batch"):
            batch = texts[i : i + batch_size]
            inputs = tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            )
            if device != -1:
                inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
            cls = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            vecs.append(cls)
        del inputs, outputs
        gc.collect()
        return np.vstack(vecs)

    CACHE_DIR = data_root / "bert_cache"  # e.g. nlp/data/bert_cache
    CACHE_DIR.mkdir(exist_ok=True)

    def cached_embed(texts: pd.Series, name: str) -> np.ndarray:
        path = CACHE_DIR / f"{name}.npy"
        if path.exists():
            print(f"Loaded cached embeddings → {path.name}", file=sys.stderr)
            return np.load(path)
        print(f"Computing embeddings for {name} …", file=sys.stderr)
        arr = bert_embed(texts.tolist())
        np.save(path, arr)
        print(f"Saved cache → {path.name}", file=sys.stderr)
        return arr

    X_train = cached_embed(train_liar["clean"], "liar_train")
    X_test = cached_embed(test_liar["clean"], "liar_test")
    X_valid = cached_embed(valid_liar["clean"], "liar_valid")
    X_gossip = cached_embed(gossip_df["clean"], "gossipcop")

    # ------------------------------------------------------------------
    # Fix A: unify all datasets to *binary* labels → 0 (real) / 1 (fake)
    # ------------------------------------------------------------------

    def _to_binary(lbl: str) -> int:  # 1 → fake, 0 → real/true
        lbl = str(lbl).lower()
        return 1 if lbl in {"fake", "false", "barely-true", "pants-fire"} else 0

    y_train = train_liar["Label"].map(_to_binary).values
    y_test = test_liar["Label"].map(_to_binary).values
    y_valid = valid_liar["Label"].map(_to_binary).values
    y_gossip = gossip_df["label"].map(_to_binary).values

    # Linear SVM
    clf_svm = SVC(kernel="linear", probability=True)
    clf_svm.fit(X_train, y_train)

    def evaluate(name: str, y_true: np.ndarray, X: np.ndarray):
        """Print Accuracy, F1 and ROC-AUC for a binary classifier."""
        y_pred = clf_svm.predict(X)
        y_score = clf_svm.predict_proba(X)[:, 1]
        roc_auc = roc_auc_score(y_true, y_score)

        print(f"\n{name}:")
        print(f"  Accuracy : {accuracy_score(y_true, y_pred):.4f}")
        print(f"  F1 Score : {f1_score(y_true, y_pred, average='binary'):.4f}")
        print(f"  ROC-AUC  : {roc_auc:.4f}")
        print(classification_report(y_true, y_pred, target_names=["REAL", "FAKE"]))

    evaluate("GossipCop", y_gossip, X_gossip)
    evaluate("LIAR – Test", y_test, X_test)
    evaluate("LIAR – Validation", y_valid, X_valid)

    # ------------------ Neural Baseline: MLP ------------------
    print(
        "\n>>> Training MLP (2-layer feed-forward) on CLS embeddings …", file=sys.stderr
    )

    mlp = MLPClassifier(
        hidden_layer_sizes=(256, 128),
        activation="relu",
        solver="adam",
        batch_size=128,
        learning_rate_init=1e-3,
        max_iter=1,  # single epoch per .fit call
        warm_start=True,  # continue training across calls
        random_state=42,
    )

    epochs = 20
    for _ in tqdm(range(epochs), desc="MLP Epochs", unit="epoch"):
        mlp.fit(X_train, y_train)

    def eval_mlp(name: str, y_true: np.ndarray, X: np.ndarray):
        y_pred = mlp.predict(X)
        y_score = mlp.predict_proba(X)[:, 1]
        roc_auc = roc_auc_score(y_true, y_score)

        print(f"\n{name} – MLP:")
        print(f"  Accuracy : {accuracy_score(y_true, y_pred):.4f}")
        print(f"  F1 Score : {f1_score(y_true, y_pred, average='binary'):.4f}")
        print(f"  ROC-AUC  : {roc_auc:.4f}")
        print(classification_report(y_true, y_pred, target_names=["REAL", "FAKE"]))

    eval_mlp("GossipCop", y_gossip, X_gossip)
    eval_mlp("LIAR – Test", y_test, X_test)
    eval_mlp("LIAR – Validation", y_valid, X_valid)

    gossip_df["Label"] = np.where(gossip_df["label"] == "fake", "false", "true")


# ---------------------------------------------------------------------------
# Stance validation (title ≈ claim vs. text)
# ---------------------------------------------------------------------------


def run_stance_validation(
    test_df: pd.DataFrame, *, device: str, cache_dir: Path
) -> None:
    """Print stance distribution on the test split.

    Uses processed_title as the *claim* and processed_text as the *article*.
    Predictions are cached to <cache_dir>/test_stance.npy for speed.
    """

    print("\n===== Stance-Detection Validation (test split) =====", file=sys.stderr)

    titles = test_df["processed_title"].fillna("").astype(str)
    texts = test_df["processed_text"].fillna("").astype(str)

    mask = (titles.str.strip() != "") & (texts.str.strip() != "")
    if mask.sum() == 0:
        print(
            "No non-empty title/text pairs available for stance detection.",
            file=sys.stderr,
        )
        return

    titles = titles[mask]
    texts = texts[mask]

    cache_path = cache_dir / "test_stance.npy"
    if cache_path.exists():
        print(f"Loaded cached stance predictions → {cache_path.name}", file=sys.stderr)
        stances = np.load(cache_path, allow_pickle=True)
    else:
        clf = load_stance_clf(device=device)
        df_tmp = pd.DataFrame({"text": texts, "claim": titles})
        df_tmp = stance_batch_predict(
            df_tmp, clf, text_col="text", claim_col="claim", show_progress=True
        )
        stances = df_tmp["stance"].values
        np.save(cache_path, stances)
        print(f"Saved stance cache → {cache_path.name}", file=sys.stderr)

    test_df = test_df.loc[mask].copy()
    test_df["stance"] = stances

    print("\nStance frequency (non-empty rows):")
    print(test_df["stance"].value_counts())

    print("\nStance vs. Fake-News label cross-tab:")
    print(pd.crosstab(test_df["label"].map({0: "REAL", 1: "FAKE"}), test_df["stance"]))

    # Simple baseline: CONTRADICTS → FAKE else REAL
    rule_pred = np.where(test_df["stance"] == "contradicts", 1, 0)
    acc = accuracy_score(test_df["label"], rule_pred)
    print(f"\nHeuristic (contradict ⇒ fake) accuracy : {acc:.4f}")

    # ----------------------- Visualization -----------------------
    stance_order = ["contradicts", "supports", "neutral"]
    ct = pd.crosstab(test_df["label"].map({0: "REAL", 1: "FAKE"}), test_df["stance"])
    # ensure columns order
    for s in stance_order:
        if s not in ct.columns:
            ct[s] = 0
    ct = ct[stance_order]

    prop = ct.div(ct.sum(axis=1), axis=0)

    prop.plot(
        kind="bar",
        stacked=True,
        color=["tab:red", "tab:green", "tab:gray"],
        figsize=(6, 4),
    )
    plt.ylabel("Proportion")
    plt.title("Stance distribution by label (test split)")
    plt.legend(title="Stance", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.show()


def _cached_stance_onehot(df: pd.DataFrame, split: str, *, data_path: Path, device: str) -> np.ndarray:
    """Return (n,2) array [contradicts, supports] with caching."""
    cache_path = data_path / f"{split}_stance_feat.npy"
    if cache_path.exists():
        return np.load(cache_path)

    titles = df["processed_title"].fillna("").astype(str)
    texts = df["processed_text"].fillna("").astype(str)
    preds: list[str] = []
    # load classifier once
    clf_sd = load_stance_clf(device=device)
    for lbl, art in tqdm(zip(titles, texts), total=len(df), desc=f"Stance {split}"):  # type: ignore
        if lbl.strip() == "" or art.strip() == "":
            preds.append("neutral")
        else:
            preds.append(
                clf_sd(art, STANCE_LABELS, hypothesis_template=f"The article {{}} the claim: '{lbl}'.")[
                    "labels"
                ][0]
            )

    # one-hot encode (contradicts, supports)
    oh = np.zeros((len(preds), 2), dtype=np.float32)
    for i, p in enumerate(preds):
        if p == "contradicts":
            oh[i, 0] = 1
        elif p == "supports":
            oh[i, 1] = 1
    np.save(cache_path, oh)
    return oh


if __name__ == "__main__":
    main()
# End of CLI script
