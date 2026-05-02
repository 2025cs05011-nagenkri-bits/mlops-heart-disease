"""
train.py

Train and compare two classifiers on the Heart Disease dataset
(Logistic Regression baseline and Random Forest ensemble), evaluate
on the held-out test set, and persist the better-performing pipeline.

Each candidate model is logged as a separate MLflow run under the
'heart_disease_classification' experiment, capturing hyperparameters,
test metrics, the fitted Pipeline, and a confusion-matrix plot.

The selected model is chosen by ROC-AUC on the test split. Metrics
are written to reports/metrics.json and the fitted Pipeline (preprocessor
+ classifier) is dumped to models/heart_model.pkl.

Usage:
  python -m src.training.train
  python -m src.training.train --experiment heart_disease_classification \
                               --tracking-uri ./mlruns
"""

import argparse
import json
import os
import tempfile

import joblib
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, f1_score,
    precision_score, recall_score, roc_auc_score,
)
from sklearn.pipeline import Pipeline

from src.data.preprocess import build_preprocessor, split_features_target

DEFAULT_TRAIN_PATH = "data/processed/train.csv"
DEFAULT_TEST_PATH = "data/processed/test.csv"
DEFAULT_MODEL_PATH = "models/heart_model.pkl"
DEFAULT_METRICS_PATH = "reports/metrics.json"
DEFAULT_EXPERIMENT = "heart_disease_classification"
DEFAULT_TRACKING_URI = "./mlruns"


def build_pipeline(classifier):
    return Pipeline([
        ("preprocess", build_preprocessor()),
        ("classifier", classifier),
    ])


def evaluate(pipeline, X_test, y_test):
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    return {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred)),
        "recall": float(recall_score(y_test, y_pred)),
        "f1": float(f1_score(y_test, y_pred)),
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "classification_report": classification_report(y_test, y_pred, output_dict=True),
    }


def _log_confusion_matrix(cm, name):
    """Render a confusion-matrix heatmap and log it to the active MLflow run."""
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", ax=ax,
        xticklabels=["no_disease", "disease"],
        yticklabels=["no_disease", "disease"],
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"{name} - confusion matrix")
    fig.tight_layout()

    tmp_dir = tempfile.mkdtemp()
    tmp_path = os.path.join(tmp_dir, f"cm_{name}.png")
    fig.savefig(tmp_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    mlflow.log_artifact(tmp_path, artifact_path="plots")
    os.remove(tmp_path)
    os.rmdir(tmp_dir)


def train_candidates(X_train, y_train, X_test, y_test, random_state=42):
    candidates = {
        "logistic_regression": LogisticRegression(
            max_iter=1000, random_state=random_state,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=200, max_depth=None, random_state=random_state, n_jobs=-1,
        ),
    }
    results = {}
    fitted = {}
    run_ids = {}
    for name, clf in candidates.items():
        with mlflow.start_run(run_name=name) as run:
            mlflow.set_tag("model_family", name)
            mlflow.log_params(clf.get_params())
            mlflow.log_param("random_state", random_state)
            mlflow.log_param("n_train", len(X_train))
            mlflow.log_param("n_test", len(X_test))

            pipe = build_pipeline(clf)
            pipe.fit(X_train, y_train)
            metrics = evaluate(pipe, X_test, y_test)

            scalar_metrics = {k: v for k, v in metrics.items()
                              if isinstance(v, (int, float))}
            mlflow.log_metrics(scalar_metrics)
            _log_confusion_matrix(metrics["confusion_matrix"], name)
            mlflow.sklearn.log_model(sk_model=pipe, name="model")

            results[name] = metrics
            fitted[name] = pipe
            run_ids[name] = run.info.run_id
    return fitted, results, run_ids


def select_best(results, metric="roc_auc"):
    return max(results, key=lambda name: results[name][metric])


def save_artifacts(pipeline, results, best_name, model_path, metrics_path):
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
    joblib.dump(pipeline, model_path)
    payload = {
        "best_model": best_name,
        "selection_metric": "roc_auc",
        "results": results,
    }
    with open(metrics_path, "w") as f:
        json.dump(payload, f, indent=2)


def print_summary(results, best_name):
    print("Model comparison (test set)")
    print("=" * 70)
    header = f"{'model':<22}{'accuracy':>10}{'precision':>11}{'recall':>9}{'f1':>9}{'roc_auc':>10}"
    print(header)
    print("-" * 70)
    for name, m in results.items():
        marker = "*" if name == best_name else " "
        print(f"{marker} {name:<20}{m['accuracy']:>10.4f}{m['precision']:>11.4f}"
              f"{m['recall']:>9.4f}{m['f1']:>9.4f}{m['roc_auc']:>10.4f}")
    print("=" * 70)
    print(f"Selected (highest roc_auc): {best_name}")


def main():
    parser = argparse.ArgumentParser(description="Train Heart Disease classifier")
    parser.add_argument("--train", type=str, default=DEFAULT_TRAIN_PATH)
    parser.add_argument("--test", type=str, default=DEFAULT_TEST_PATH)
    parser.add_argument("--model-path", type=str, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--metrics-path", type=str, default=DEFAULT_METRICS_PATH)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--experiment", type=str, default=DEFAULT_EXPERIMENT)
    parser.add_argument("--tracking-uri", type=str, default=DEFAULT_TRACKING_URI)
    args = parser.parse_args()

    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.experiment)

    train_df = pd.read_csv(args.train)
    test_df = pd.read_csv(args.test)
    X_train, y_train = split_features_target(train_df)
    X_test, y_test = split_features_target(test_df)

    fitted, results, run_ids = train_candidates(
        X_train, y_train, X_test, y_test, args.random_state,
    )
    best_name = select_best(results, metric="roc_auc")
    save_artifacts(fitted[best_name], results, best_name, args.model_path, args.metrics_path)

    client = mlflow.MlflowClient(tracking_uri=args.tracking_uri)
    for name, run_id in run_ids.items():
        client.set_tag(run_id, "selected", "true" if name == best_name else "false")
    client.log_artifact(run_ids[best_name], args.metrics_path)

    print_summary(results, best_name)
    print(f"Saved model   -> {args.model_path}")
    print(f"Saved metrics -> {args.metrics_path}")
    print(f"MLflow store  -> {args.tracking_uri}  (experiment: {args.experiment})")
    print(f"Best run ID   -> {run_ids[best_name]}")


if __name__ == "__main__":
    main()
