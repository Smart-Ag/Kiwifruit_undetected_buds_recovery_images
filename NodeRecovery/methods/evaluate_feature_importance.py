"""Evaluate the four MGC features with grouped ablation and test-set permutation."""

import os
from pathlib import Path
import pickle

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

# ---------------- USER INPUT ----------------
DATA_DIR = Path(os.environ.get("NR_WORK_DIR", Path(__file__).resolve().parent / "work"))
MASK_DIR = DATA_DIR / "res_out" / "ml_masks"
MODEL_DIR = DATA_DIR / "weights" / "grouped_cv"
OUT_DIR = MODEL_DIR / "feature_importance"
MISSING_PROBABILITIES = (0.3, 0.4, 0.5, 0.6)
RANDOM_SEED = int(os.environ.get("NR_RANDOM_SEED", "42"))
N_FOLDS = 5
N_PERMUTATIONS = 100
# --------------------------------------------

FEATURES = [
    "gap_length_m",
    "mean_IL_m",
    "std_IL_m",
    "rel_position",
]
TARGET = "label_missing"
GROUP = "Branch_id"
GAP_TYPE = "gap_type"
VALID_GAP_TYPES = {"internal", "basal", "apical", "basal_apical"}
REQUIRED_COLUMNS = set(FEATURES + [TARGET, GROUP, GAP_TYPE, "mask_probability"])


def load_mask_data(
    path: Path, expected_probability: float
) -> tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """Load one masked dataset and retain missing feature values for its pipeline."""
    data = pd.read_csv(path)
    missing_columns = REQUIRED_COLUMNS.difference(data.columns)
    if missing_columns:
        raise ValueError(
            f"{path.name}: missing required column(s): "
            + ", ".join(sorted(missing_columns))
        )

    probabilities = pd.to_numeric(data["mask_probability"], errors="coerce").dropna().unique()
    if len(probabilities) != 1 or not np.isclose(probabilities[0], expected_probability):
        raise ValueError(
            f"{path.name}: expected mask_probability={expected_probability}, "
            f"found {probabilities.tolist()}."
        )

    X = data[FEATURES].astype(float)
    if np.isinf(X.to_numpy()).any():
        raise ValueError(f"{path.name}: feature values must not be infinite.")

    y = data[TARGET].astype(int)
    groups = data[GROUP]
    if set(y.unique()) != {0, 1}:
        raise ValueError(f"{path.name}: {TARGET} must contain both 0 and 1.")
    if groups.isna().any():
        raise ValueError(f"{path.name}: {GROUP} contains missing values.")
    gap_types = data[GAP_TYPE]
    if gap_types.isna().any() or not set(gap_types).issubset(VALID_GAP_TYPES):
        raise ValueError(
            f"{path.name}: {GAP_TYPE} must contain only "
            f"{sorted(VALID_GAP_TYPES)}. Regenerate the boundary-aware masks."
        )
    return X, y, groups, gap_types


def prediction_scores(model, X: pd.DataFrame) -> np.ndarray:
    """Return continuous positive-class scores for ROC-AUC and PR-AUC."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        return model.decision_function(X)
    raise TypeError("The fitted model supplies neither predict_proba nor decision_function.")


def auc_metrics(y_true: pd.Series, scores: np.ndarray) -> dict[str, float]:
    """PR-AUC is calculated as average precision, the standard AUPRC estimator."""
    return {
        "roc_auc": roc_auc_score(y_true, scores),
        "pr_auc": average_precision_score(y_true, scores),
    }


def grouped_ablation(
    fitted_model,
    model_name: str,
    probability: float,
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
) -> pd.DataFrame:
    """Measure loss of grouped-CV discrimination after dropping one feature."""
    splitter = StratifiedGroupKFold(
        n_splits=N_FOLDS,
        shuffle=True,
        random_state=RANDOM_SEED,
    )
    folds = list(splitter.split(X, y, groups))
    rows = []
    feature_sets = [("none", FEATURES)] + [
        (feature, [candidate for candidate in FEATURES if candidate != feature])
        for feature in FEATURES
    ]

    for removed_feature, used_features in feature_sets:
        for fold_number, (train_index, validation_index) in enumerate(folds, start=1):
            model = clone(fitted_model)
            model.fit(X.iloc[train_index][used_features], y.iloc[train_index])
            scores = prediction_scores(model, X.iloc[validation_index][used_features])
            metrics = auc_metrics(y.iloc[validation_index], scores)

            rows.append(
                {
                    "mask_probability": probability,
                    "model": model_name,
                    "removed_feature": removed_feature,
                    "features_used": ", ".join(used_features),
                    "fold": fold_number,
                    "train_samples": len(train_index),
                    "validation_samples": len(validation_index),
                    "train_branches": groups.iloc[train_index].nunique(),
                    "validation_branches": groups.iloc[validation_index].nunique(),
                    **metrics,
                }
            )
    return pd.DataFrame(rows)


def summarise_ablation(fold_results: pd.DataFrame) -> pd.DataFrame:
    """Summarise feature-removal performance relative to the full feature set."""
    summary = (
        fold_results.groupby(
            ["mask_probability", "model", "removed_feature", "features_used"],
            as_index=False,
        )
        .agg(
            roc_auc_mean=("roc_auc", "mean"),
            roc_auc_sd=("roc_auc", "std"),
            pr_auc_mean=("pr_auc", "mean"),
            pr_auc_sd=("pr_auc", "std"),
        )
    )
    baseline = (
        summary.loc[summary["removed_feature"].eq("none")]
        .set_index("mask_probability")[["roc_auc_mean", "pr_auc_mean"]]
        .rename(
            columns={
                "roc_auc_mean": "full_roc_auc_mean",
                "pr_auc_mean": "full_pr_auc_mean",
            }
        )
    )
    summary = summary.join(baseline, on="mask_probability")
    summary["roc_auc_drop"] = summary["full_roc_auc_mean"] - summary["roc_auc_mean"]
    summary["pr_auc_drop"] = summary["full_pr_auc_mean"] - summary["pr_auc_mean"]
    return summary.sort_values(["mask_probability", "pr_auc_drop"], ascending=[True, False])


def held_out_permutation_importance(
    fitted_model,
    model_name: str,
    probability: float,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Quantify held-out performance loss after permuting each feature."""
    baseline_scores = prediction_scores(fitted_model, X_test)
    baseline_metrics = auc_metrics(y_test, baseline_scores)
    repeat_rows = []

    for feature in FEATURES:
        for repeat in range(1, N_PERMUTATIONS + 1):
            permuted_X = X_test.copy()
            permuted_X[feature] = rng.permutation(permuted_X[feature].to_numpy())
            permuted_scores = prediction_scores(fitted_model, permuted_X)
            permuted_metrics = auc_metrics(y_test, permuted_scores)
            repeat_rows.append(
                {
                    "mask_probability": probability,
                    "model": model_name,
                    "feature": feature,
                    "repeat": repeat,
                    "permuted_roc_auc": permuted_metrics["roc_auc"],
                    "permuted_pr_auc": permuted_metrics["pr_auc"],
                    "roc_auc_drop": baseline_metrics["roc_auc"] - permuted_metrics["roc_auc"],
                    "pr_auc_drop": baseline_metrics["pr_auc"] - permuted_metrics["pr_auc"],
                }
            )

    repeats = pd.DataFrame(repeat_rows)
    summary = (
        repeats.groupby(["mask_probability", "model", "feature"], as_index=False)
        .agg(
            roc_auc_drop_mean=("roc_auc_drop", "mean"),
            roc_auc_drop_sd=("roc_auc_drop", "std"),
            roc_auc_drop_ci_low=("roc_auc_drop", lambda values: values.quantile(0.025)),
            roc_auc_drop_ci_high=("roc_auc_drop", lambda values: values.quantile(0.975)),
            pr_auc_drop_mean=("pr_auc_drop", "mean"),
            pr_auc_drop_sd=("pr_auc_drop", "std"),
            pr_auc_drop_ci_low=("pr_auc_drop", lambda values: values.quantile(0.025)),
            pr_auc_drop_ci_high=("pr_auc_drop", lambda values: values.quantile(0.975)),
        )
    )
    summary["baseline_roc_auc"] = baseline_metrics["roc_auc"]
    summary["baseline_pr_auc"] = baseline_metrics["pr_auc"]
    summary["test_samples"] = len(y_test)
    summary["test_positive_samples"] = int(y_test.sum())
    summary = summary.sort_values(["mask_probability", "pr_auc_drop_mean"], ascending=[True, False])
    return repeats, summary


selection_path = MODEL_DIR / "held_out_test_results.csv"
selection = pd.read_csv(selection_path).set_index("mask_probability")
OUT_DIR.mkdir(parents=True, exist_ok=True)

all_ablation_folds = []
all_permutation_repeats = []
all_permutation_summaries = []
composition_rows = []

for probability_index, probability in enumerate(MISSING_PROBABILITIES):
    train_path = MASK_DIR / f"train_mask_{probability:.1f}.csv"
    test_path = MASK_DIR / f"test_mask_{probability:.1f}.csv"
    model_path = MODEL_DIR / f"selected_model_mask_{probability:.1f}.pkl"

    X_train, y_train, train_groups, train_gap_types = load_mask_data(train_path, probability)
    X_test, y_test, test_groups, test_gap_types = load_mask_data(test_path, probability)
    if set(train_groups).intersection(test_groups):
        raise ValueError(f"p={probability}: train and test datasets share branches.")

    for partition, labels, groups, gap_types in (
        ("train", y_train, train_groups, train_gap_types),
        ("test", y_test, test_groups, test_gap_types),
    ):
        composition = pd.DataFrame(
            {"gap_type": gap_types, "label_missing": labels, "Branch_id": groups}
        )
        composition = (
            composition.groupby(["gap_type", "label_missing"], as_index=False)
            .agg(samples=("Branch_id", "size"), branches=("Branch_id", "nunique"))
        )
        composition.insert(0, "partition", partition)
        composition.insert(1, "mask_probability", probability)
        composition_rows.extend(composition.to_dict("records"))

    with model_path.open("rb") as model_file:
        selected_model = pickle.load(model_file)
    model_name = selection.loc[probability, "selected_model"]

    ablation_folds = grouped_ablation(
        fitted_model=selected_model,
        model_name=model_name,
        probability=probability,
        X=X_train,
        y=y_train,
        groups=train_groups,
    )
    all_ablation_folds.append(ablation_folds)

    seed_sequence = np.random.SeedSequence([RANDOM_SEED, probability_index])
    rng = np.random.default_rng(seed_sequence)
    permutation_repeats, permutation_summary = held_out_permutation_importance(
        fitted_model=selected_model,
        model_name=model_name,
        probability=probability,
        X_test=X_test,
        y_test=y_test,
        rng=rng,
    )
    all_permutation_repeats.append(permutation_repeats)
    all_permutation_summaries.append(permutation_summary)

    print(
        f"p={probability:.1f} | {model_name} | "
        f"test ROC-AUC={permutation_summary['baseline_roc_auc'].iloc[0]:.4f} | "
        f"test PR-AUC={permutation_summary['baseline_pr_auc'].iloc[0]:.4f}"
    )

ablation_folds = pd.concat(all_ablation_folds, ignore_index=True)
ablation_summary = summarise_ablation(ablation_folds)
permutation_repeats = pd.concat(all_permutation_repeats, ignore_index=True)
permutation_summary = pd.concat(all_permutation_summaries, ignore_index=True)

ablation_folds.to_csv(OUT_DIR / "grouped_cv_feature_ablation_folds.csv", index=False)
ablation_summary.to_csv(OUT_DIR / "grouped_cv_feature_ablation_summary.csv", index=False)
permutation_repeats.to_csv(OUT_DIR / "held_out_permutation_importance_repeats.csv", index=False)
permutation_summary.to_csv(OUT_DIR / "held_out_permutation_importance_summary.csv", index=False)
pd.DataFrame(composition_rows).to_csv(
    OUT_DIR / "boundary_gap_composition.csv", index=False
)

pd.DataFrame(
    {
        "setting": [
            "importance_method",
            "cross_validation",
            "permutation_repeats",
            "roc_auc",
            "pr_auc",
            "boundary_gap_metadata",
        ],
        "value": [
            "Grouped leave-one-feature-out ablation plus held-out permutation importance",
            f"{N_FOLDS}-fold StratifiedGroupKFold grouped by Branch_id",
            N_PERMUTATIONS,
            "Area under the receiver operating characteristic curve",
            "Average precision (standard PR-AUC/AUPRC estimator)",
            "gap_type is retained for sample-composition reporting and is not a classifier feature",
        ],
    }
).to_csv(OUT_DIR / "feature_importance_methods.csv", index=False)

print(f"\nSaved feature-importance results to: {OUT_DIR}")
