"""
Evaluation helpers: ROC-AUC, precision-recall, confusion matrix,
feature importance, and optional chart output.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline


def compute_metrics(y_true, y_pred, y_prob) -> Dict[str, float]:
    """Return a dict of scalar evaluation metrics."""
    return {
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "avg_precision": float(average_precision_score(y_true, y_prob)),
        "f1": float(f1_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
    }


def print_evaluation(model_name: str, y_true, y_pred, y_prob) -> Dict[str, float]:
    """Print a full evaluation report and return metric dict."""
    metrics = compute_metrics(y_true, y_pred, y_prob)
    print(f"\n{'='*60}")
    print(f"  {model_name}")
    print(f"{'='*60}")
    print(f"  ROC-AUC:          {metrics['roc_auc']:.4f}")
    print(f"  Avg Precision:    {metrics['avg_precision']:.4f}")
    print(f"  F1 (binary):      {metrics['f1']:.4f}")
    print(f"  F1 (macro):       {metrics['f1_macro']:.4f}")
    print()
    print(classification_report(y_true, y_pred, target_names=["Retained", "Churned"]))
    cm = confusion_matrix(y_true, y_pred)
    print("Confusion matrix:")
    print(f"  TN={cm[0,0]:4d}  FP={cm[0,1]:4d}")
    print(f"  FN={cm[1,0]:4d}  TP={cm[1,1]:4d}")
    return metrics


def get_feature_importance(pipeline: Pipeline, top_n: int = 20) -> pd.DataFrame:
    """
    Extract feature importance from the final estimator in a Pipeline.
    Works for tree-based models (feature_importances_) and linear models
    (coef_).
    """
    prep = pipeline.named_steps["prep"]
    model = pipeline.named_steps["model"]

    # Build feature names from the ColumnTransformer
    names: list[str] = []
    for name, transformer, cols in prep.transformers_:
        if name == "remainder":
            continue
        if hasattr(transformer, "get_feature_names_out"):
            names.extend(transformer.get_feature_names_out(cols).tolist())
        else:
            # Pipeline — delegate to last step
            last = transformer[-1] if hasattr(transformer, "__iter__") else transformer
            if hasattr(last, "get_feature_names_out"):
                names.extend(last.get_feature_names_out(cols).tolist())
            else:
                names.extend(cols)

    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_[0])
    else:
        return pd.DataFrame()

    df = pd.DataFrame({"feature": names[: len(importances)], "importance": importances})
    return df.sort_values("importance", ascending=False).head(top_n).reset_index(drop=True)


def save_charts(
    pipeline: Pipeline,
    model_name: str,
    X_test,
    y_test,
    out_dir: Path = Path("experiments/charts"),
) -> None:
    """Save ROC curve + confusion matrix PNGs (optional, skips if matplotlib unavailable)."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    slug = model_name.replace(" ", "_").lower()
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    y_pred = pipeline.predict(X_test)

    # ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], "k--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curve — {model_name}")
    ax.legend()
    fig.savefig(out_dir / f"roc_{slug}.png", dpi=100, bbox_inches="tight")
    plt.close(fig)

    # Confusion matrix
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred, display_labels=["Retained", "Churned"], ax=ax
    )
    ax.set_title(f"Confusion Matrix — {model_name}")
    fig.savefig(out_dir / f"cm_{slug}.png", dpi=100, bbox_inches="tight")
    plt.close(fig)
