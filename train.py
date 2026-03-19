#!/usr/bin/env python3
"""
End-to-end training script.

Trains a Logistic Regression baseline and an XGBoost model on the
synthetic churn dataset, evaluates both, logs runs to the experiment
tracker, and saves the best model to models/.

Usage:
    python train.py [--data data/raw/churn_data.csv] [--model-dir models]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Optional XGBoost — fall back to GradientBoosting if not installed
try:
    from xgboost import XGBClassifier as _GBM

    def _make_gbm():
        return _GBM(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
        )

    GBM_NAME = "XGBoost"
except ImportError:
    from sklearn.ensemble import GradientBoostingClassifier as _GBM  # type: ignore

    def _make_gbm():
        return _GBM(n_estimators=200, max_depth=4, learning_rate=0.05, random_state=42)

    GBM_NAME = "GradientBoosting"

from churn.evaluate import get_feature_importance, print_evaluation, save_charts
from churn.features import build_pipeline, load_data
from churn.tracker import ExperimentTracker

RANDOM_STATE = 42
TEST_SIZE = 0.20


def train(data_path: str, model_dir: str) -> None:
    model_dir_path = Path(model_dir)
    model_dir_path.mkdir(parents=True, exist_ok=True)

    print(f"Loading data from {data_path} …")
    X, y = load_data(data_path)
    print(f"  {len(X):,} samples | churn rate: {y.mean():.1%}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"  Train: {len(X_train):,}  |  Test: {len(X_test):,}")

    tracker = ExperimentTracker()

    # ── Model 1: Logistic Regression (baseline) ───────────────────────────────
    print("\n[1/2] Training Logistic Regression baseline …")
    lr_model = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs", random_state=RANDOM_STATE)
    lr_pipeline = build_pipeline(lr_model)
    lr_pipeline.fit(X_train, y_train)

    y_pred_lr = lr_pipeline.predict(X_test)
    y_prob_lr = lr_pipeline.predict_proba(X_test)[:, 1]
    lr_metrics = print_evaluation("Logistic Regression (baseline)", y_test, y_pred_lr, y_prob_lr)

    lr_run_id = tracker.log_run(
        model_name="LogisticRegression",
        params={"C": 1.0, "max_iter": 1000, "solver": "lbfgs"},
        metrics=lr_metrics,
        tags={"type": "baseline"},
    )
    print(f"  Logged run: {lr_run_id}")

    joblib.dump(lr_pipeline, model_dir_path / "logreg_pipeline.pkl")
    save_charts(lr_pipeline, "Logistic Regression", X_test, y_test)

    # ── Model 2: XGBoost / GradientBoosting ──────────────────────────────────
    print(f"\n[2/2] Training {GBM_NAME} …")
    gbm_pipeline = build_pipeline(_make_gbm())
    gbm_pipeline.fit(X_train, y_train)

    y_pred_gbm = gbm_pipeline.predict(X_test)
    y_prob_gbm = gbm_pipeline.predict_proba(X_test)[:, 1]
    gbm_metrics = print_evaluation(GBM_NAME, y_test, y_pred_gbm, y_prob_gbm)

    gbm_run_id = tracker.log_run(
        model_name=GBM_NAME,
        params={"n_estimators": 300, "max_depth": 5, "learning_rate": 0.05},
        metrics=gbm_metrics,
        tags={"type": "champion"},
    )
    print(f"  Logged run: {gbm_run_id}")

    joblib.dump(gbm_pipeline, model_dir_path / "gbm_pipeline.pkl")
    save_charts(gbm_pipeline, GBM_NAME, X_test, y_test)

    # ── Feature importance (best model) ──────────────────────────────────────
    print(f"\nTop-10 feature importances ({GBM_NAME}):")
    fi = get_feature_importance(gbm_pipeline, top_n=10)
    if not fi.empty:
        for _, row in fi.iterrows():
            bar = "█" * int(row["importance"] / fi["importance"].max() * 30)
            print(f"  {row['feature']:<45s} {bar}")

    # ── Experiment summary ────────────────────────────────────────────────────
    print("\n" + tracker.summary())

    best = tracker.best_run("roc_auc")
    if best:
        print(f"\nBest run: {best['run_id']} → {best['model']}  ROC-AUC={best['metrics']['roc_auc']:.4f}")

    print(f"\nModels saved to {model_dir_path.resolve()}/")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train churn prediction models")
    parser.add_argument("--data", default="data/raw/churn_data.csv")
    parser.add_argument("--model-dir", default="models")
    args = parser.parse_args()

    if not Path(args.data).exists():
        print(f"Data file not found: {args.data}")
        print("Run:  python -m churn.generate_data  to generate it first.")
        sys.exit(1)

    train(args.data, args.model_dir)


if __name__ == "__main__":
    main()
