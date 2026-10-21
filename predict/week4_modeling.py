#!/usr/bin/env python3
"""
Week 4: Classification models for DP2 (review rounds) and DP4 (stale PRs).

Models:
  - Baselines: DummyClassifier (majority), LogisticRegression, size-only CatBoost
  - Main: CatBoost, LightGBM

Validation:
  - 5-fold stratified CV (intra-project)
  - Leave-one-repo-out CV (cross-project)

Output: predict/output/
"""

import csv
import os
import warnings

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap

from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, precision_score, recall_score,
    average_precision_score, confusion_matrix,
)
from sklearn.preprocessing import StandardScaler

from catboost import CatBoostClassifier
import lightgbm as lgb

warnings.filterwarnings("ignore")

PROCESSED_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "processed")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight"})

# ── Features ────────────────────────────────────────────────────────────────

FEATURE_COLS = [
    # Group 1: PR-level
    "lines_added", "lines_deleted", "files_changed", "num_commits",
    "has_tests", "has_docs", "touches_ci",
    "desc_length", "title_length", "has_screenshots", "has_code_blocks",
    "references_issues", "is_draft", "label_count",
    "day_of_week", "hour_of_day",
    # Group 2: Developer history
    "author_merge_rate", "author_avg_review_rounds",
    "author_commit_frequency", "author_tenure_days", "author_pr_count",
    # Group 3: Project context
    "open_pr_count", "merge_velocity_7d", "merge_velocity_30d",
]

SIZE_ONLY_COLS = ["lines_added", "lines_deleted", "files_changed", "num_commits"]

BOOL_COLS = {"has_tests", "has_docs", "touches_ci", "has_screenshots",
             "has_code_blocks", "references_issues", "is_draft"}


def to_float(val):
    if val in ("", "None", "none", None):
        return np.nan
    if val in ("True", "true"):
        return 1.0
    if val in ("False", "false"):
        return 0.0
    try:
        return float(val)
    except (ValueError, TypeError):
        return np.nan


# ── Data loading ────────────────────────────────────────────────────────────

def load_data(path, feature_cols, target_fn):
    """Load CSV, extract features and target.

    Returns: X (np.array), y (np.array), repos (list[str]), valid_indices
    """
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    X_list, y_list, repos = [], [], []
    for row in rows:
        target = target_fn(row)
        if target is None:
            continue
        features = [to_float(row.get(c, "")) for c in feature_cols]
        X_list.append(features)
        y_list.append(target)
        repos.append(row.get("repo", ""))

    X = np.array(X_list, dtype=np.float64)
    y = np.array(y_list, dtype=np.int32)

    # Impute NaN with column median
    for j in range(X.shape[1]):
        col = X[:, j]
        mask = np.isnan(col)
        if mask.any():
            median_val = np.nanmedian(col)
            col[mask] = median_val

    return X, y, repos


def dp2_target(row):
    """Binary: 1 round (0) vs 2+ rounds (1)."""
    val = to_float(row.get("review_rounds", ""))
    if np.isnan(val):
        return None
    return 1 if val >= 2 else 0


def dp4_target(row):
    """Binary: stale (1) vs not stale (0)."""
    val = row.get("is_stale", "")
    if val in ("True", "true", "1"):
        return 1
    if val in ("False", "false", "0"):
        return 0
    return None


# ── Evaluation ──────────────────────────────────────────────────────────────

def evaluate(y_true, y_pred, y_prob=None):
    """Compute classification metrics."""
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
    }
    if y_prob is not None and len(np.unique(y_true)) > 1:
        metrics["auc_roc"] = roc_auc_score(y_true, y_prob)
        metrics["auc_pr"] = average_precision_score(y_true, y_prob)
    return metrics


def format_metrics(metrics):
    return " | ".join(f"{k}={v:.3f}" for k, v in metrics.items())


# ── Models ──────────────────────────────────────────────────────────────────

def train_eval_models(X, y, repos, task_name, feature_cols):
    """Train all models with 5-fold CV and leave-one-repo-out."""
    results = {}

    print(f"\n{'=' * 70}")
    print(f"  {task_name}")
    print(f"  Samples: {len(y)}, Positive: {y.sum()} ({100 * y.mean():.1f}%)")
    print(f"{'=' * 70}")

    # ── 5-fold Stratified CV ────────────────────────────────────────────
    print(f"\n--- 5-fold Stratified CV ---")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    models_config = {
        "Dummy (majority)": lambda: DummyClassifier(strategy="most_frequent"),
        "Logistic Regression": lambda: LogisticRegression(max_iter=1000, random_state=42),
        "CatBoost (size-only)": lambda: CatBoostClassifier(
            iterations=300, depth=6, learning_rate=0.1,
            verbose=0, random_seed=42, auto_class_weights="Balanced",
        ),
        "CatBoost": lambda: CatBoostClassifier(
            iterations=500, depth=6, learning_rate=0.05,
            verbose=0, random_seed=42, auto_class_weights="Balanced",
        ),
        "LightGBM": lambda: lgb.LGBMClassifier(
            n_estimators=500, max_depth=6, learning_rate=0.05,
            is_unbalance=True, verbose=-1, random_state=42, n_jobs=-1,
        ),
    }

    for model_name, model_fn in models_config.items():
        fold_metrics = []
        is_size_only = "size-only" in model_name

        for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
            if is_size_only:
                size_col_idx = [feature_cols.index(c) for c in SIZE_ONLY_COLS]
                X_train = X[train_idx][:, size_col_idx]
                X_test = X[test_idx][:, size_col_idx]
            else:
                X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            model = model_fn()

            if model_name == "Logistic Regression":
                scaler = StandardScaler()
                X_train = scaler.fit_transform(X_train)
                X_test = scaler.transform(X_test)

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            y_prob = None
            if hasattr(model, "predict_proba"):
                y_prob = model.predict_proba(X_test)[:, 1]

            fold_metrics.append(evaluate(y_test, y_pred, y_prob))

        avg = {k: np.mean([m[k] for m in fold_metrics]) for k in fold_metrics[0]}
        results[f"5fold_{model_name}"] = avg
        print(f"  {model_name:30s} | {format_metrics(avg)}")

    # ── Leave-one-repo-out CV ───────────────────────────────────────────
    print(f"\n--- Leave-one-repo-out CV ---")
    unique_repos = sorted(set(repos))
    repos_arr = np.array(repos)

    for model_name in ["CatBoost", "LightGBM"]:
        model_fn = models_config[model_name]
        repo_metrics = {}

        for repo in unique_repos:
            test_mask = repos_arr == repo
            train_mask = ~test_mask

            if test_mask.sum() < 10 or train_mask.sum() < 10:
                continue
            if len(np.unique(y[test_mask])) < 2:
                continue

            model = model_fn()
            model.fit(X[train_mask], y[train_mask])
            y_pred = model.predict(X[test_mask])
            y_prob = model.predict_proba(X[test_mask])[:, 1] if hasattr(model, "predict_proba") else None

            repo_metrics[repo] = evaluate(y[test_mask], y_pred, y_prob)

        if repo_metrics:
            avg = {k: np.mean([m[k] for m in repo_metrics.values()]) for k in list(repo_metrics.values())[0]}
            results[f"LORO_{model_name}"] = avg
            results[f"LORO_{model_name}_per_repo"] = repo_metrics
            print(f"  {model_name:30s} | {format_metrics(avg)}")
            for repo, m in sorted(repo_metrics.items()):
                repo_short = repo.split("/")[1] if "/" in repo else repo
                print(f"    {repo_short:25s} | f1={m['f1']:.3f} auc={m.get('auc_roc', 0):.3f}")

    # ── Train final model for SHAP ──────────────────────────────────────
    print(f"\n--- Training final CatBoost for SHAP ---")
    final_model = CatBoostClassifier(
        iterations=500, depth=6, learning_rate=0.05,
        verbose=0, random_seed=42, auto_class_weights="Balanced",
    )
    final_model.fit(X, y)
    results["final_model"] = final_model
    results["feature_importance"] = dict(zip(feature_cols, final_model.feature_importances_))

    return results


# ── SHAP analysis ───────────────────────────────────────────────────────────

def shap_analysis(model, X, feature_cols, task_name):
    """Generate SHAP summary plot."""
    print(f"\n--- SHAP analysis for {task_name} ---")
    # Use a sample for speed
    sample_size = min(2000, len(X))
    idx = np.random.RandomState(42).choice(len(X), sample_size, replace=False)
    X_sample = X[idx]

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    # For binary classification, shap_values may be a list of 2 arrays
    if isinstance(shap_values, list):
        shap_values = shap_values[1]  # positive class

    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(shap_values, X_sample, feature_names=feature_cols, show=False, max_display=20)
    plt.title(f"SHAP Feature Importance — {task_name}")
    fig = plt.gcf()
    fig.savefig(os.path.join(OUTPUT_DIR, f"shap_{task_name.lower().replace(' ', '_')}.png"))
    plt.close("all")
    print(f"  Saved SHAP plot")

    # Feature importance bar chart
    fi = dict(zip(feature_cols, np.abs(shap_values).mean(axis=0)))
    fi_sorted = sorted(fi.items(), key=lambda x: x[1], reverse=True)

    fig, ax = plt.subplots(figsize=(10, 7))
    names = [f[0] for f in fi_sorted[:15]]
    vals = [f[1] for f in fi_sorted[:15]]
    ax.barh(range(len(names)), vals[::-1], color="#2196F3", alpha=0.8)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names[::-1])
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title(f"Top 15 Features — {task_name}")
    fig.savefig(os.path.join(OUTPUT_DIR, f"feature_importance_{task_name.lower().replace(' ', '_')}.png"))
    plt.close(fig)
    print(f"  Saved feature importance plot")

    return fi_sorted


# ── Confusion matrix plot ───────────────────────────────────────────────────

def plot_confusion_matrix(y_true, y_pred, task_name, labels=None):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if cm[i, j] > cm.max() / 2 else "black"
            ax.text(j, i, f"{cm[i, j]}", ha="center", va="center", color=color, fontsize=14)
    if labels:
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix — {task_name}")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.savefig(os.path.join(OUTPUT_DIR, f"confusion_matrix_{task_name.lower().replace(' ', '_')}.png"))
    plt.close(fig)


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    # ── DP2: Review Rounds ──────────────────────────────────────────────
    print("Loading DP2 data (merged PRs)...")
    merged_path = os.path.join(PROCESSED_DIR, "all_merged_features.csv")
    X_dp2, y_dp2, repos_dp2 = load_data(merged_path, FEATURE_COLS, dp2_target)

    dp2_results = train_eval_models(X_dp2, y_dp2, repos_dp2, "DP2 Review Rounds", FEATURE_COLS)

    # Confusion matrix for final model
    final_dp2 = dp2_results["final_model"]
    y_pred_dp2 = final_dp2.predict(X_dp2)
    plot_confusion_matrix(y_dp2, y_pred_dp2, "DP2", labels=["1 round", "2+ rounds"])

    # SHAP
    fi_dp2 = shap_analysis(final_dp2, X_dp2, FEATURE_COLS, "DP2")

    # ── DP4: Stale PRs ─────────────────────────────────────────────────
    print("\nLoading DP4 data (open PRs)...")
    open_path = os.path.join(PROCESSED_DIR, "all_open_features.csv")
    X_dp4, y_dp4, repos_dp4 = load_data(open_path, FEATURE_COLS, dp4_target)

    dp4_results = train_eval_models(X_dp4, y_dp4, repos_dp4, "DP4 Stale PRs", FEATURE_COLS)

    # Confusion matrix
    final_dp4 = dp4_results["final_model"]
    y_pred_dp4 = final_dp4.predict(X_dp4)
    plot_confusion_matrix(y_dp4, y_pred_dp4, "DP4", labels=["Active", "Stale"])

    # SHAP
    fi_dp4 = shap_analysis(final_dp4, X_dp4, FEATURE_COLS, "DP4")

    # ── Save results report ─────────────────────────────────────────────
    lines = ["# Week 4 — Modeling Results", ""]

    for task, res in [("DP2 (Review Rounds)", dp2_results), ("DP4 (Stale PRs)", dp4_results)]:
        lines.append(f"## {task}")
        lines.append("")
        lines.append("### 5-fold Stratified CV")
        lines.append("")
        lines.append("| Model | Accuracy | F1 | Precision | Recall | AUC-ROC | AUC-PR |")
        lines.append("|---|---|---|---|---|---|---|")
        for key, metrics in res.items():
            if key.startswith("5fold_"):
                name = key.replace("5fold_", "")
                auc = metrics.get("auc_roc", "—")
                auc_pr = metrics.get("auc_pr", "—")
                if isinstance(auc, float):
                    auc = f"{auc:.3f}"
                if isinstance(auc_pr, float):
                    auc_pr = f"{auc_pr:.3f}"
                lines.append(
                    f"| {name} | {metrics['accuracy']:.3f} | {metrics['f1']:.3f} | "
                    f"{metrics['precision']:.3f} | {metrics['recall']:.3f} | {auc} | {auc_pr} |"
                )
        lines.append("")

        lines.append("### Leave-one-repo-out CV")
        lines.append("")
        for key, metrics in res.items():
            if key.startswith("LORO_") and not key.endswith("_per_repo"):
                name = key.replace("LORO_", "")
                auc = metrics.get("auc_roc", "—")
                if isinstance(auc, float):
                    auc = f"{auc:.3f}"
                lines.append(f"**{name}:** F1={metrics['f1']:.3f}, AUC-ROC={auc}")
        lines.append("")

        # Feature importance
        fi = res.get("feature_importance", {})
        if fi:
            lines.append("### Top 10 Features (CatBoost importance)")
            lines.append("")
            fi_sorted = sorted(fi.items(), key=lambda x: x[1], reverse=True)
            for i, (feat, imp) in enumerate(fi_sorted[:10]):
                lines.append(f"{i+1}. **{feat}**: {imp:.1f}")
            lines.append("")

    report_path = os.path.join(OUTPUT_DIR, "modeling_results.md")
    with open(report_path, "w") as f:
        f.write("\n".join(lines))
    print(f"\nReport saved: {report_path}")
    print(f"All outputs in: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
