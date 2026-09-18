"""Grouped model selection on masked training FCs and final held-out testing."""

import os
from pathlib import Path
import pickle

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

# ---------------- USER INPUT ----------------
DATA_DIR = Path(os.environ.get("NR_WORK_DIR", Path(__file__).resolve().parent / "work"))
MASK_DIR = DATA_DIR / "res_out" / "ml_masks"
RESULTS_DIR = DATA_DIR / "weights" / "grouped_cv"
MISSING_PROBABILITIES = (0.3, 0.4, 0.5, 0.6)
RANDOM_SEED = int(os.environ.get("NR_RANDOM_SEED", "42"))
N_FOLDS = 5
PRIMARY_SELECTION_METRIC = "f1"
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


def validate_mask_file(
    data: pd.DataFrame,
    path: Path,
    expected_probability: float,
) -> None:
    """Confirm that a file contains one valid mask-probability dataset."""
    missing_columns = REQUIRED_COLUMNS.difference(data.columns)
    if missing_columns:
        raise ValueError(
            f"{path.name}: missing required column(s): "
            + ", ".join(sorted(missing_columns))
        )

    if data[GROUP].isna().any():
        raise ValueError(f"{path.name}: {GROUP} contains missing values.")
    if data[TARGET].isna().any():
        raise ValueError(f"{path.name}: {TARGET} contains missing values.")
    if data[GAP_TYPE].isna().any() or not set(data[GAP_TYPE]).issubset(VALID_GAP_TYPES):
        raise ValueError(
            f"{path.name}: {GAP_TYPE} must contain only "
            f"{sorted(VALID_GAP_TYPES)}. Regenerate the boundary-aware masks."
        )

    observed_probabilities = pd.to_numeric(
        data["mask_probability"], errors="coerce"
    ).dropna().unique()
    if (
        len(observed_probabilities) != 1
        or not np.isclose(observed_probabilities[0], expected_probability)
    ):
        raise ValueError(
            f"{path.name}: expected mask_probability={expected_probability}, "
            f"found {observed_probabilities.tolist()}."
        )

    try:
        numeric_features = data[FEATURES].astype(float)
    except ValueError as error:
        raise ValueError(f"{path.name}: all model features must be numeric.") from error

    if np.isinf(numeric_features.to_numpy()).any():
        raise ValueError(f"{path.name}: feature values must not be infinite.")


def prepare_xy_groups(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Return model inputs while preserving missing feature values for imputation."""
    X = data[FEATURES].astype(float)
    y = data[TARGET].astype(int)
    groups = data[GROUP]

    if set(y.unique()) != {0, 1}:
        raise ValueError("label_missing must contain both binary classes (0 and 1).")
    if groups.nunique() < N_FOLDS:
        raise ValueError(
            f"At least {N_FOLDS} unique branches are required for grouped CV; "
            f"found {groups.nunique()}."
        )
    return X, y, groups


def build_models(y_train: pd.Series) -> dict[str, Pipeline]:
    """Create fresh model templates with class balancing based on this dataset."""
    positive_count = int((y_train == 1).sum())
    negative_count = int((y_train == 0).sum())
    scale_pos_weight = negative_count / positive_count

    return {
        "Logistic Regression": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                ("scaler", StandardScaler()),
                ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced")),
            ]
        ),
        "XGBoost": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                (
                    "classifier",
                    XGBClassifier(
                        n_estimators=150,
                        max_depth=3,
                        learning_rate=0.1,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        eval_metric="logloss",
                        scale_pos_weight=scale_pos_weight,
                        random_state=RANDOM_SEED,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "SVM": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    SVC(
                        kernel="rbf",
                        C=1.0,
                        gamma="scale",
                        class_weight="balanced",
                        probability=False,
                        random_state=RANDOM_SEED,
                    ),
                ),
            ]
        ),
        "Random Forest": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=300,
                        max_depth=None,
                        min_samples_leaf=1,
                        class_weight="balanced",
                        random_state=RANDOM_SEED,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
    }


def metric_row(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }


def scoped_metric_rows(
    y_true: pd.Series,
    y_pred: np.ndarray,
    gap_types: pd.Series,
) -> list[dict[str, float | int | str]]:
    """Report performance overall and for internal and boundary gaps separately.

    Gap type remains metadata only: the classifier is still fitted with the
    four predefined morphometric features and not with a boundary indicator.
    """
    masks = {
        "all": np.ones(len(y_true), dtype=bool),
        "internal": gap_types.eq("internal").to_numpy(),
        "boundary": gap_types.ne("internal").to_numpy(),
        "basal": gap_types.eq("basal").to_numpy(),
        "apical": gap_types.eq("apical").to_numpy(),
        "basal_apical": gap_types.eq("basal_apical").to_numpy(),
    }
    rows = []
    y_array = y_true.to_numpy()
    for scope, mask in masks.items():
        if not np.any(mask):
            continue
        subset_y = y_array[mask]
        rows.append(
            {
                "gap_scope": scope,
                "samples": int(np.sum(mask)),
                "positive_samples": int(np.sum(subset_y)),
                **metric_row(subset_y, y_pred[mask]),
            }
        )
    return rows


def grouped_cross_validation(
    models: dict[str, Pipeline],
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    gap_types: pd.Series,
    probability: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Evaluate models with branch-disjoint validation folds."""
    splitter = StratifiedGroupKFold(
        n_splits=N_FOLDS,
        shuffle=True,
        random_state=RANDOM_SEED,
    )
    folds = list(splitter.split(X, y, groups))
    fold_rows = []
    scoped_fold_rows = []

    for model_name, model_template in models.items():
        for fold_number, (train_index, validation_index) in enumerate(folds, start=1):
            fold_model = clone(model_template)
            fold_model.fit(X.iloc[train_index], y.iloc[train_index])
            validation_prediction = fold_model.predict(X.iloc[validation_index])

            metrics = metric_row(y.iloc[validation_index], validation_prediction)
            fold_rows.append(
                {
                    "mask_probability": probability,
                    "model": model_name,
                    "fold": fold_number,
                    "train_samples": len(train_index),
                    "validation_samples": len(validation_index),
                    "train_branches": groups.iloc[train_index].nunique(),
                    "validation_branches": groups.iloc[validation_index].nunique(),
                    **metrics,
                }
            )
            for scoped_metrics in scoped_metric_rows(
                y.iloc[validation_index],
                validation_prediction,
                gap_types.iloc[validation_index],
            ):
                scoped_fold_rows.append(
                    {
                        "mask_probability": probability,
                        "model": model_name,
                        "fold": fold_number,
                        "train_samples": len(train_index),
                        "validation_samples": len(validation_index),
                        "train_branches": groups.iloc[train_index].nunique(),
                        "validation_branches": groups.iloc[validation_index].nunique(),
                        **scoped_metrics,
                    }
                )

    fold_results = pd.DataFrame(fold_rows)
    summary = (
        fold_results.groupby(["mask_probability", "model"], as_index=False)
        .agg(
            cv_precision_mean=("precision", "mean"),
            cv_precision_sd=("precision", "std"),
            cv_recall_mean=("recall", "mean"),
            cv_recall_sd=("recall", "std"),
            cv_f1_mean=("f1", "mean"),
            cv_f1_sd=("f1", "std"),
        )
    )
    return fold_results, summary, pd.DataFrame(scoped_fold_rows)


def select_model(cv_summary: pd.DataFrame) -> str:
    """Choose the best model using training-partition CV only."""
    metric_column = f"cv_{PRIMARY_SELECTION_METRIC}_mean"
    return (
        cv_summary.sort_values(
            by=[metric_column, "cv_precision_mean", "model"],
            ascending=[False, False, True],
        )
        .iloc[0]["model"]
    )


RESULTS_DIR.mkdir(parents=True, exist_ok=True)
all_fold_results = []
all_scoped_fold_results = []
all_cv_summaries = []
test_rows = []
test_scope_rows = []

for probability in MISSING_PROBABILITIES:
    train_path = MASK_DIR / f"train_mask_{probability:.1f}.csv"
    test_path = MASK_DIR / f"test_mask_{probability:.1f}.csv"
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)

    validate_mask_file(train_data, train_path, probability)
    validate_mask_file(test_data, test_path, probability)

    train_branches = set(train_data[GROUP])
    test_branches = set(test_data[GROUP])
    shared_branches = train_branches.intersection(test_branches)
    if shared_branches:
        raise ValueError(
            f"p={probability}: train/test branch leakage detected: "
            f"{sorted(shared_branches)}"
        )

    X_train, y_train, train_groups = prepare_xy_groups(train_data)
    X_test, y_test, test_groups = prepare_xy_groups(test_data)
    models = build_models(y_train)

    print(
        f"\np={probability:.1f}: {len(train_data)} training samples from "
        f"{train_groups.nunique()} branches; {len(test_data)} held-out samples from "
        f"{test_groups.nunique()} branches"
    )

    fold_results, cv_summary, scoped_fold_results = grouped_cross_validation(
        models=models,
        X=X_train,
        y=y_train,
        groups=train_groups,
        gap_types=train_data[GAP_TYPE],
        probability=probability,
    )
    best_model_name = select_model(cv_summary)
    best_model = clone(models[best_model_name])
    best_model.fit(X_train, y_train)

    # The held-out test partition is touched only after model selection above.
    test_prediction = best_model.predict(X_test)
    test_metrics = metric_row(y_test, test_prediction)
    model_path = RESULTS_DIR / f"selected_model_mask_{probability:.1f}.pkl"
    with model_path.open("wb") as model_file:
        pickle.dump(best_model, model_file)

    cv_summary["selected_by_cv"] = cv_summary["model"].eq(best_model_name)
    all_fold_results.append(fold_results)
    all_scoped_fold_results.append(scoped_fold_results)
    all_cv_summaries.append(cv_summary)
    test_rows.append(
        {
            "mask_probability": probability,
            "selected_model": best_model_name,
            "selection_metric": PRIMARY_SELECTION_METRIC,
            "cv_f1_mean": cv_summary.loc[
                cv_summary["model"].eq(best_model_name), "cv_f1_mean"
            ].iloc[0],
            "cv_f1_sd": cv_summary.loc[
                cv_summary["model"].eq(best_model_name), "cv_f1_sd"
            ].iloc[0],
            "test_samples": len(test_data),
            "test_branches": test_groups.nunique(),
            "test_positive_samples": int(y_test.sum()),
            **test_metrics,
            "model_path": str(model_path),
        }
    )
    for scoped_metrics in scoped_metric_rows(
        y_test, test_prediction, test_data[GAP_TYPE]
    ):
        test_scope_rows.append(
            {
                "mask_probability": probability,
                "selected_model": best_model_name,
                "selection_metric": PRIMARY_SELECTION_METRIC,
                "test_branches": test_groups.nunique(),
                **scoped_metrics,
            }
        )

    print(
        f"  Selected by grouped CV: {best_model_name}; "
        f"held-out test F1={test_metrics['f1']:.4f}, "
        f"precision={test_metrics['precision']:.4f}, "
        f"recall={test_metrics['recall']:.4f}"
    )

pd.concat(all_fold_results, ignore_index=True).to_csv(
    RESULTS_DIR / "grouped_cv_fold_results.csv", index=False
)
pd.concat(all_scoped_fold_results, ignore_index=True).to_csv(
    RESULTS_DIR / "grouped_cv_fold_results_by_gap_scope.csv", index=False
)
pd.concat(all_cv_summaries, ignore_index=True).to_csv(
    RESULTS_DIR / "grouped_cv_model_selection.csv", index=False
)
pd.DataFrame(test_rows).to_csv(
    RESULTS_DIR / "held_out_test_results.csv", index=False
)
pd.DataFrame(test_scope_rows).to_csv(
    RESULTS_DIR / "held_out_test_results_by_gap_scope.csv", index=False
)

print(f"\nSaved grouped-CV and held-out-test reports to: {RESULTS_DIR}")
