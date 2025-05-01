import os
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")  # force Transformers to skip TensorFlow/Keras

from typing import List

import pandas as pd
from tqdm.auto import tqdm

from transformers import pipeline  # type: ignore
import torch

LABELS = ["supports", "contradicts", "neutral"]
TEMPLATE = "The article {} the claim: '{}'."


# --------------------------------
# Device helpers
# --------------------------------


def _select_device(choice: str = "auto"):
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


def load_classifier(
    model_name: str = "roberta-large-mnli", *, device: str = "auto"
) -> "transformers.pipelines.Pipeline":  # pragma: no cover
    """Return a Hugging-Face zero-shot classification pipeline.

    Parameters
    ----------
    model_name : str, optional
        HF model checkpoint (default ``roberta-large-mnli``).
    device : {"auto", "cpu", "cuda", "mps"}, optional
        Compute backend for inference. "auto" prefers CUDA → MPS → CPU.
    """

    device_id = _select_device(device)
    clf = pipeline(
        "zero-shot-classification",
        model=model_name,
        device=device_id,
        framework="pt",  # ensure PyTorch is used
    )
    return clf


def _truncate(text: str, max_len: int = 2000) -> str:
    """Crude truncate to keep inference under 512 BPE tokens."""
    return text[:max_len]


def predict_stance(article: str, claim: str, clf) -> str:
    """Return the top stance label for *article* w.r.t. *claim*."""
    article = _truncate(article)
    hypothesis = TEMPLATE.format("{}", claim)
    result = clf(article, LABELS, hypothesis_template=hypothesis)
    return result["labels"][0]


def batch_predict(
    df: pd.DataFrame,
    clf,
    text_col: str = "text",
    claim_col: str = "claim",
    show_progress: bool = True,
) -> pd.DataFrame:
    """Append a `stance` column to *df* using *clf* for predictions."""
    iterator = tqdm(df.iterrows(), total=len(df)) if show_progress else df.iterrows()
    stances: List[str] = []
    for _, row in iterator:
        stances.append(predict_stance(row[text_col], row[claim_col], clf))
    df = df.copy()
    df["stance"] = stances
    return df


__all__ = [
    "load_classifier",
    "predict_stance",
    "batch_predict",
] 