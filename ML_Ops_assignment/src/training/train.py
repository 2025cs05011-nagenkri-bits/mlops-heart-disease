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
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline

from src.data.preprocess import build_preprocessor, split_features_target

DEFAULT_TRAIN_PATH = "data/processed/train.csv"
DEFAULT_TEST_PATH = "data/processed/test.csv"
DEFAULT_MODEL_PATH = "models/heart_model.pkl"
DEFAULT_METRICS_PATH = "reports/metrics.json"
DEFAULT_EXPERIMENT = "heart_disease_classification"
DEFAULT_TRACKING_URI = "./mlruns"
CV_FOLDS = 5
CV_SCORING = ["accuracy", "precision", "recall", "f1", "roc_auc"]

CANDIDATES = {
    "logistic_regression": {
        "estimator": LogisticRegression(max_iter=1000),
        "param_grid": {
            "classifier__C": [0.1, 1.0, 10.0],
        },
    },
    "random_forest": {
        "estimator": RandomForestClassifier(n_jobs=-1),
        "param_grid": {
            "classifier__n_estimators": [100, 200],
            "classifier__max_depth": [None, 5, 10],
        },
    },
}


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


def cross_validate_summary(pipeline, X, y, cv):
    """Run cross_validate over CV_SCORING and return a flat mean/std dict."""
    cvr = cross_validate(
        pipeline, X, y, cv=cv, scoring=CV_SCORING,
        n_jobs=-1, return_train_score=False,
    )
    summary = {}
    for scorer in CV_SCORING:
        scores = cvr[f"test_{scorer}"]
        summary[f"cv_{scorer}_mean"] = float(scores.mean())
        summary[f"cv_{scorer}_std"] = float(scores.std())
    return summary


def train_candidates(X_train, y_train, X_test, y_test, random_state=42):
    """For each candidate: GridSearchCV-tune, 5-fold CV-evaluate, then test-set evaluate."""
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=random_state)
    results = {}
    fitted = {}
    run_ids = {}
    for name, spec in CANDIDATES.items():
        with mlflow.start_run(run_name=name) as run:
            mlflow.set_tag("model_family", name)
            mlflow.log_param("random_state", random_state)
            mlflow.log_param("n_train", len(X_train))
            mlflow.log_param("n_test", len(X_test))
            mlflow.log_param("cv_folds", CV_FOLDS)

            estimator = spec["estimator"]
            if hasattr(estimator, "random_state"):
                estimator.set_params(random_state=random_state)
            base_pipe = build_pipeline(estimator)

            search = GridSearchCV(
                base_pipe, spec["param_grid"], cv=cv,
                scoring="roc_auc", n_jobs=-1, refit=True,
            )
            search.fit(X_train, y_train)
            best_pipe = search.best_estimator_

            for k, v in search.best_params_.items():
                mlflow.log_param(f"best_{k}", v)
            mlflow.log_metric("cv_best_roc_auc_search", float(search.best_score_))

            cv_summary = cross_validate_summary(best_pipe, X_train, y_train, cv)
            mlflow.log_metrics(cv_summary)

            metrics = evaluate(best_pipe, X_test, y_test)
            scalar_test = {f"test_{k}": v for k, v in metrics.items()
                           if isinstance(v, (int, float))}
            mlflow.log_metrics(scalar_test)
            _log_confusion_matrix(metrics["confusion_matrix"], name)
            mlflow.sklearn.log_model(sk_model=best_pipe, name="model")

            metrics["best_params"] = search.best_params_
            metrics["cv"] = cv_summary

            results[name] = metrics
            fitted[name] = best_pipe
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
    print("=" * 78)
    header = (
        f"{'model':<22}{'accuracy':>10}{'precision':>11}{'recall':>9}"
        f"{'f1':>9}{'roc_auc':>10}{'cv_roc_auc':>12}"
    )
    print(header)
    print("-" * 78)
    for name, m in results.items():
        marker = "*" if name == best_name else " "
        cv_mean = m["cv"]["cv_roc_auc_mean"]
        cv_std = m["cv"]["cv_roc_auc_std"]
        print(f"{marker} {name:<20}{m['accuracy']:>10.4f}{m['precision']:>11.4f}"
              f"{m['recall']:>9.4f}{m['f1']:>9.4f}{m['roc_auc']:>10.4f}"
              f"  {cv_mean:.4f}±{cv_std:.4f}")
    print("=" * 78)
    print(f"Selected (highest test roc_auc): {best_name}")
    print(f"Best params : {results[best_name]['best_params']}")


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
