"""Visibility-anchored, stage-wise evaluation of reconstructed fruiting-cane nodes.

Detected nodes are summarized by counts only; no detected-node correspondence
or detection tolerance is assumed. Recovery stages are evaluated at the fixed
configured positional tolerance inside gaps defined from the ground-truth
Recon_visibility annotations. This is an oracle-gap assessment: it isolates
recovery localization from errors in detected-node positions.

Optional PNM input
------------------
To score MGC and PNM gap-level performance, provide pnm_gap_results.xlsx or
pnm_gap_results.csv in res_out. Required columns are:

    cane_id, gap_id, gap_index, start_arc_m, end_arc_m, mgc_prediction,
    pnm_map_count, pnm_lower_count, pnm_upper_count

The file must contain every observed gap, including MGC-negative gaps. Each PNM
row is paired with the visibility-defined reference gap using cane_id and
gap_index; true_missing_count is taken from that reference gap. A count obtained
by applying the reconstructed arc interval directly to ground-truth arcs is
retained only as a coordinate-alignment diagnostic.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


# --------------------------- USER SETTINGS ---------------------------
CODE_DIR = Path(__file__).resolve().parent.parent
RUN_DIR = Path(os.environ.get("NR_RUN_DIR", CODE_DIR / "runs"))
GROUND_TRUTH_PATH = Path(os.environ.get("NR_GROUND_TRUTH_PATH", RUN_DIR / "ground_truth.xlsx"))
PREDICTION_XLSX_PATH = Path(os.environ.get("NR_PREDICTION_XLSX_PATH", RUN_DIR / "recon_recovered.xlsx"))
PREDICTION_CSV_PATH = Path(os.environ.get("NR_PREDICTION_CSV_PATH", RUN_DIR / "recon_recovered.csv"))
PNM_GAP_XLSX_PATH = Path(os.environ.get("NR_PNM_GAP_XLSX_PATH", RUN_DIR / "pnm_gap_results.xlsx"))
PNM_GAP_CSV_PATH = Path(os.environ.get("NR_PNM_GAP_CSV_PATH", RUN_DIR / "pnm_gap_results.csv"))
OUTPUT_DIR = Path(os.environ.get("NR_EVAL_OUTPUT_DIR", RUN_DIR / "evaluation"))

# Fixed recovery localization tolerance selected by the user.
RECOVERY_TOLERANCE_MM = float(os.environ.get("NR_RECOVERY_TOLERANCE_MM", "30.0"))

ACCEPTED_TYPES = {"detected", "bulge", "guessed"}
GPLA_TYPES = {"bulge", "candidate"}
# ---------------------------------------------------------------------


GROUND_TRUTH_REQUIRED = {"cane_id", "node_id", "arc_m", "diameter_mm"}
PREDICTION_REQUIRED = {"cane_id", "predicted_id", "arc_m", "diameter_mm", "type"}
PNM_REQUIRED = {
    "cane_id",
    "gap_id",
    "gap_index",
    "start_arc_m",
    "end_arc_m",
    "mgc_prediction",
    "pnm_map_count",
    "pnm_lower_count",
    "pnm_upper_count",
}


def read_table(path: Path) -> pd.DataFrame:
    """Read an Excel or CSV table while preserving the source file unchanged."""
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported input format: {path.suffix}")


def locate_pnm_gap_table() -> Path | None:
    """Prefer the automatically updated CSV when both input forms are present."""
    if PNM_GAP_CSV_PATH.exists():
        return PNM_GAP_CSV_PATH
    if PNM_GAP_XLSX_PATH.exists():
        return PNM_GAP_XLSX_PATH
    return None


def locate_prediction_table() -> Path:
    """Prefer the automatically updated recovery CSV over a legacy Excel input."""
    if PREDICTION_CSV_PATH.exists():
        return PREDICTION_CSV_PATH
    return PREDICTION_XLSX_PATH


def require_columns(data: pd.DataFrame, required: set[str], name: str) -> None:
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"{name} is missing required column(s): {', '.join(sorted(missing))}")


def prepare_ground_truth(data: pd.DataFrame) -> pd.DataFrame:
    require_columns(data, GROUND_TRUTH_REQUIRED, "Ground-truth table")
    result = data.copy()
    visibility_columns = [
        column
        for column in result.columns
        if str(column).strip().lower() == "recon_visibility"
    ]
    if len(visibility_columns) != 1:
        raise ValueError(
            "Ground-truth table must contain exactly one recon_visibility column. "
            "Mark each ground-truth bud detected by reconstruction as 'Detected' "
            "and leave the remaining buds blank."
        )
    visibility = (
        result[visibility_columns[0]]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
    )
    allowed_visibility = {"", "detected", "undetected", "not_detected", "not detected"}
    unexpected_visibility = set(visibility).difference(allowed_visibility)
    if unexpected_visibility:
        raise ValueError(
            "recon_visibility contains unsupported value(s): "
            + ", ".join(sorted(unexpected_visibility))
        )
    result["recon_visibility"] = visibility
    result["is_recon_detected"] = visibility.eq("detected")
    result["cane_id"] = pd.to_numeric(result["cane_id"], errors="raise").astype(int)
    result["node_id"] = result["node_id"].astype(str)
    result["arc_m"] = pd.to_numeric(result["arc_m"], errors="raise")
    result["diameter_mm"] = pd.to_numeric(result["diameter_mm"], errors="coerce")

    if result[["cane_id", "node_id", "arc_m"]].isna().any().any():
        raise ValueError("Ground-truth cane_id, node_id, and arc_m must be non-missing.")
    if result.duplicated(["cane_id", "node_id"]).any():
        raise ValueError("Ground-truth node_id values must be unique within each cane.")

    result = result.sort_values(["cane_id", "arc_m", "node_id"]).reset_index(drop=True)
    arc_differences = result.groupby("cane_id", sort=False)["arc_m"].diff()
    if (arc_differences.dropna() <= 0).any():
        raise ValueError("Ground-truth arc_m values must be strictly increasing within each cane.")
    return result


def prepare_predictions(data: pd.DataFrame) -> pd.DataFrame:
    require_columns(data, PREDICTION_REQUIRED, "Prediction table")
    result = data.copy()
    result["cane_id"] = pd.to_numeric(result["cane_id"], errors="raise").astype(int)
    result["predicted_id"] = result["predicted_id"].astype(str)
    result["arc_m"] = pd.to_numeric(result["arc_m"], errors="raise")
    result["diameter_mm"] = pd.to_numeric(result["diameter_mm"], errors="coerce")
    result["type"] = result["type"].astype(str).str.strip().str.lower()

    if result[["cane_id", "predicted_id", "arc_m", "type"]].isna().any().any():
        raise ValueError("Prediction cane_id, predicted_id, arc_m, and type must be non-missing.")
    if result.duplicated(["cane_id", "predicted_id"]).any():
        raise ValueError("predicted_id values must be unique within each cane.")

    result = result.sort_values(["cane_id", "arc_m", "predicted_id"]).reset_index(drop=True)
    arc_differences = result.groupby("cane_id", sort=False)["arc_m"].diff()
    if (arc_differences.dropna() <= 0).any():
        raise ValueError("Prediction arc_m values must be strictly increasing within each cane.")
    return result


def is_better(candidate: tuple[int, float, str], current: tuple[int, float, str] | None) -> bool:
    """Prefer more matches, then lower total location error, then direct matches."""
    if current is None:
        return True
    candidate_matches, candidate_cost, candidate_action = candidate
    current_matches, current_cost, current_action = current
    if candidate_matches != current_matches:
        return candidate_matches > current_matches
    if not np.isclose(candidate_cost, current_cost):
        return candidate_cost < current_cost
    action_priority = {"match": 0, "skip_gt": 1, "skip_prediction": 2}
    return action_priority[candidate_action] < action_priority[current_action]


def ordered_match(
    ground_truth: pd.DataFrame,
    predictions: pd.DataFrame,
    tolerance_m: float,
) -> list[tuple[int, int, float]]:
    """Return maximum-cardinality, minimum-error, one-to-one monotonic matches."""
    n_ground_truth, n_predictions = len(ground_truth), len(predictions)
    matched_count = np.zeros((n_ground_truth + 1, n_predictions + 1), dtype=int)
    total_error = np.zeros((n_ground_truth + 1, n_predictions + 1), dtype=float)
    action = np.full((n_ground_truth + 1, n_predictions + 1), "", dtype=object)

    for i in range(1, n_ground_truth + 1):
        action[i, 0] = "skip_gt"
    for j in range(1, n_predictions + 1):
        action[0, j] = "skip_prediction"

    gt_arcs = ground_truth["arc_m"].to_numpy(dtype=float)
    prediction_arcs = predictions["arc_m"].to_numpy(dtype=float)

    for i in range(1, n_ground_truth + 1):
        for j in range(1, n_predictions + 1):
            choices: list[tuple[int, float, str]] = [
                (matched_count[i - 1, j], total_error[i - 1, j], "skip_gt"),
                (matched_count[i, j - 1], total_error[i, j - 1], "skip_prediction"),
            ]
            error = abs(gt_arcs[i - 1] - prediction_arcs[j - 1])
            if error <= tolerance_m:
                choices.append(
                    (matched_count[i - 1, j - 1] + 1, total_error[i - 1, j - 1] + error, "match")
                )

            best: tuple[int, float, str] | None = None
            for choice in choices:
                if is_better(choice, best):
                    best = choice
            assert best is not None
            matched_count[i, j], total_error[i, j], action[i, j] = best

    matches: list[tuple[int, int, float]] = []
    i, j = n_ground_truth, n_predictions
    while i > 0 or j > 0:
        current_action = action[i, j]
        if current_action == "match":
            error = abs(gt_arcs[i - 1] - prediction_arcs[j - 1])
            matches.append((i - 1, j - 1, error))
            i -= 1
            j -= 1
        elif current_action == "skip_gt":
            i -= 1
        elif current_action == "skip_prediction":
            j -= 1
        else:
            raise RuntimeError("Matching backtrack encountered an undefined action.")
    return list(reversed(matches))


def count_only_metrics(
    ground_truth: pd.DataFrame,
    predictions: pd.DataFrame,
    stage: str,
    shared_canes: list[int],
) -> pd.DataFrame:
    """Report node totals and count error without implying node correspondence."""
    rows: list[dict[str, Any]] = []
    total_target = 0
    total_prediction = 0
    for cane_id in shared_canes:
        cane_gt = ground_truth.loc[ground_truth["cane_id"].eq(cane_id)]
        cane_predictions = predictions.loc[predictions["cane_id"].eq(cane_id)]
        if stage == "detected_count":
            target_count = int(cane_gt["is_recon_detected"].sum())
            prediction_count = int(cane_predictions["type"].eq("detected").sum())
            target_definition = "ground_truth_rows_marked_recon_visibility_detected"
        elif stage == "overall_node_count":
            target_count = len(cane_gt)
            prediction_count = int(cane_predictions["type"].isin(ACCEPTED_TYPES).sum())
            target_definition = "all_ground_truth_rows"
        else:
            raise ValueError(f"Unsupported count-only stage: {stage}")

        total_target += target_count
        total_prediction += prediction_count
        count_error = prediction_count - target_count
        rows.append(
            {
                "stage": stage,
                "cane_id": cane_id,
                "ground_truth_count": target_count,
                "ground_truth_count_definition": target_definition,
                "predicted_count": prediction_count,
                "count_error_predicted_minus_ground_truth": count_error,
                "absolute_count_error": abs(count_error),
                "relative_count_error_percent": (
                    100 * count_error / target_count if target_count else np.nan
                ),
                "positional_matching_performed": False,
            }
        )

    pooled_count_error = total_prediction - total_target
    rows.append(
        {
            "stage": stage,
            "cane_id": "all_canes_pooled",
            "ground_truth_count": total_target,
            "ground_truth_count_definition": target_definition,
            "predicted_count": total_prediction,
            "count_error_predicted_minus_ground_truth": pooled_count_error,
            "absolute_count_error": abs(pooled_count_error),
            "relative_count_error_percent": (
                100 * pooled_count_error / total_target if total_target else np.nan
            ),
            "positional_matching_performed": False,
        }
    )
    return pd.DataFrame(rows)


def assignment_rows(
    stage: str,
    cane_id: int,
    tolerance_mm: float,
    ground_truth: pd.DataFrame,
    predictions: pd.DataFrame,
    matches: list[tuple[int, int, float]],
    matching_basis: str = "arc_tolerance",
) -> list[dict[str, Any]]:
    """Create auditable matched, false-negative, and false-positive node rows."""
    matched_gt = {gt_index for gt_index, _, _ in matches}
    matched_predictions = {prediction_index for _, prediction_index, _ in matches}
    rows: list[dict[str, Any]] = []

    for gt_index, prediction_index, error in matches:
        gt = ground_truth.iloc[gt_index]
        prediction = predictions.iloc[prediction_index]
        rows.append(
            {
                "stage": stage,
                "cane_id": cane_id,
                "tolerance_mm": tolerance_mm,
                "match_status": "matched",
                "ground_truth_node_id": gt["node_id"],
                "ground_truth_arc_m": gt["arc_m"],
                "ground_truth_diameter_mm": gt["diameter_mm"],
                "predicted_id": prediction["predicted_id"],
                "predicted_arc_m": prediction["arc_m"],
                "predicted_diameter_mm": prediction["diameter_mm"],
                "predicted_type": prediction["type"],
                "absolute_error_mm": error * 1000,
            }
        )

    for gt_index, gt in ground_truth.iterrows():
        if gt_index not in matched_gt:
            rows.append(
                {
                    "stage": stage,
                    "cane_id": cane_id,
                    "tolerance_mm": tolerance_mm,
                    "match_status": "false_negative",
                    "ground_truth_node_id": gt["node_id"],
                    "ground_truth_arc_m": gt["arc_m"],
                    "ground_truth_diameter_mm": gt["diameter_mm"],
                    "predicted_id": np.nan,
                    "predicted_arc_m": np.nan,
                    "predicted_diameter_mm": np.nan,
                    "predicted_type": np.nan,
                    "absolute_error_mm": np.nan,
                }
            )

    for prediction_index, prediction in predictions.iterrows():
        if prediction_index not in matched_predictions:
            rows.append(
                {
                    "stage": stage,
                    "cane_id": cane_id,
                    "tolerance_mm": tolerance_mm,
                    "match_status": "false_positive",
                    "ground_truth_node_id": np.nan,
                    "ground_truth_arc_m": np.nan,
                    "ground_truth_diameter_mm": np.nan,
                    "predicted_id": prediction["predicted_id"],
                    "predicted_arc_m": prediction["arc_m"],
                    "predicted_diameter_mm": prediction["diameter_mm"],
                    "predicted_type": prediction["type"],
                    "absolute_error_mm": np.nan,
                }
            )
    for row in rows:
        row["matching_basis"] = matching_basis
    return rows


def ground_truth_total_length(ground_truth: pd.DataFrame) -> float:
    """Return the recorded FC length when available, otherwise the last node arc."""
    total_length_columns = [
        column
        for column in ground_truth.columns
        if str(column).strip().lower() == "total_length_m"
    ]
    if total_length_columns:
        values = pd.to_numeric(
            ground_truth[total_length_columns[0]], errors="coerce"
        ).dropna()
        if len(values) and values.iloc[0] > 0:
            return float(values.iloc[0])
    return float(ground_truth["arc_m"].max())


def stage_summary_from_counts(
    stage: str,
    cane_id: int,
    recovery_tolerance_mm: float,
    target_count: int,
    prediction_count: int,
    match_errors_m: list[float],
    matching_basis: str,
    evaluable_gaps: int,
    excluded_gaps: int,
) -> dict[str, Any]:
    """Create the standard metric row when matching occurs independently per gap."""
    true_positive = len(match_errors_m)
    false_negative = target_count - true_positive
    false_positive = prediction_count - true_positive
    precision = true_positive / prediction_count if prediction_count else np.nan
    recall = true_positive / target_count if target_count else np.nan
    f1 = (
        2 * precision * recall / (precision + recall)
        if np.isfinite(precision) and np.isfinite(recall) and precision + recall > 0
        else np.nan
    )
    errors_mm = np.asarray(match_errors_m, dtype=float) * 1000
    return {
        "stage": stage,
        "cane_id": cane_id,
        "tolerance_mm": recovery_tolerance_mm,
        "matching_basis": matching_basis,
        "evaluable_gaps": evaluable_gaps,
        "excluded_gaps": excluded_gaps,
        "target_nodes": target_count,
        "predicted_nodes": prediction_count,
        "TP": true_positive,
        "FP": false_positive,
        "FN": false_negative,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "count_error": prediction_count - target_count,
        "count_absolute_error": abs(prediction_count - target_count),
        "matched_nodes": true_positive,
        "mean_localisation_error_mm": errors_mm.mean() if len(errors_mm) else np.nan,
        "median_localisation_error_mm": np.median(errors_mm) if len(errors_mm) else np.nan,
        "rmse_localisation_error_mm": np.sqrt(np.mean(errors_mm**2)) if len(errors_mm) else np.nan,
        "p95_localisation_error_mm": np.quantile(errors_mm, 0.95) if len(errors_mm) else np.nan,
    }


def build_visibility_gap_records(
    cane_id: int,
    ground_truth: pd.DataFrame,
    predictions: pd.DataFrame,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Build oracle gaps from ground-truth visibility labels, without detection matching."""
    ground_truth = ground_truth.reset_index(drop=True)
    predictions = predictions.reset_index(drop=True)
    full_length_m = ground_truth_total_length(ground_truth)

    visible_indices = np.flatnonzero(
        ground_truth["is_recon_detected"].to_numpy()
    ).tolist()
    gt_anchors = [-1, *visible_indices, len(ground_truth)]

    rows: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    for gap_index, (start_gt_index, end_gt_index) in enumerate(
        zip(gt_anchors[:-1], gt_anchors[1:]), start=1
    ):
        is_basal = start_gt_index == -1
        is_apical = end_gt_index == len(ground_truth)
        if is_basal and is_apical:
            gap_type = "basal_apical"
        elif is_basal:
            gap_type = "basal"
        elif is_apical:
            gap_type = "apical"
        else:
            gap_type = "internal"

        start_node_id = None if is_basal else str(ground_truth.iloc[start_gt_index]["node_id"])
        end_node_id = None if is_apical else str(ground_truth.iloc[end_gt_index]["node_id"])
        start_arc = 0.0 if is_basal else float(ground_truth.iloc[start_gt_index]["arc_m"])
        end_arc = full_length_m if is_apical else float(ground_truth.iloc[end_gt_index]["arc_m"])
        true_nodes = ground_truth.iloc[start_gt_index + 1:end_gt_index].reset_index(drop=True)

        in_gap = predictions.loc[
            predictions["arc_m"].gt(start_arc)
            & predictions["arc_m"].lt(end_arc)
        ].reset_index(drop=True)
        row: dict[str, Any] = {
            "cane_id": cane_id,
            "visibility_gap_id": f"cane_{cane_id}_visibility_gap_{gap_index:03d}",
            "gap_index": gap_index,
            "gap_type": gap_type,
            "gap_definition": "ground_truth_recon_visibility",
            "evaluation_status": "evaluable",
            "start_ground_truth_node_id": start_node_id,
            "end_ground_truth_node_id": end_node_id,
            "start_anchor_type": "skeleton_base" if is_basal else "annotated_detected_node",
            "end_anchor_type": "skeleton_apex" if is_apical else "annotated_detected_node",
            "true_missing_count": len(true_nodes),
            "start_ground_truth_arc_m": 0.0 if is_basal else float(ground_truth.iloc[start_gt_index]["arc_m"]),
            "end_ground_truth_arc_m": full_length_m if is_apical else float(ground_truth.iloc[end_gt_index]["arc_m"]),
            "evaluation_start_arc_m": start_arc,
            "evaluation_end_arc_m": end_arc,
            "gpla_peak_count": int(in_gap["type"].isin(GPLA_TYPES).sum()),
            "bulge_count": int(in_gap["type"].eq("bulge").sum()),
            "guessed_count": int(in_gap["type"].eq("guessed").sum()),
            "recovered_count": int(in_gap["type"].isin({"bulge", "guessed"}).sum()),
            "candidate_count": int(in_gap["type"].eq("candidate").sum()),
        }
        records.append(
            {
                "gap_id": row["visibility_gap_id"],
                "gap_row": row,
                "true_nodes": true_nodes,
                "gpla": in_gap.loc[in_gap["type"].isin(GPLA_TYPES)].reset_index(drop=True),
                "bulges": in_gap.loc[in_gap["type"].eq("bulge")].reset_index(drop=True),
                "guessed": in_gap.loc[in_gap["type"].eq("guessed")].reset_index(drop=True),
                "recovered": in_gap.loc[in_gap["type"].isin({"bulge", "guessed"})].reset_index(drop=True),
            }
        )
        rows.append(row)
    return rows, records


def score_within_gap_stage(
    stage: str,
    cane_id: int,
    recovery_tolerance_mm: float,
    records: list[dict[str, Any]],
    prediction_key: str,
    target_key: str = "true_nodes",
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Score a node type strictly inside each valid visibility-defined gap."""
    assignments: list[dict[str, Any]] = []
    errors_m: list[float] = []
    target_count = 0
    prediction_count = 0
    for record in records:
        targets = record[target_key]
        predictions = record[prediction_key]
        matches = ordered_match(targets, predictions, recovery_tolerance_mm / 1000)
        record[f"{stage}_matches"] = matches
        target_count += len(targets)
        prediction_count += len(predictions)
        errors_m.extend(error for _, _, error in matches)
        gap_assignments = assignment_rows(
            stage, cane_id, recovery_tolerance_mm, targets, predictions, matches,
            "ground_truth_visibility_gap + recovery_arc_tolerance",
        )
        for row in gap_assignments:
            row["visibility_gap_id"] = record["gap_id"]
        assignments.extend(gap_assignments)
        record["gap_row"][f"{stage}_TP"] = len(matches)
        record["gap_row"][f"{stage}_FP"] = len(predictions) - len(matches)
        record["gap_row"][f"{stage}_FN"] = len(targets) - len(matches)
    return (
        stage_summary_from_counts(
            stage, cane_id, recovery_tolerance_mm, target_count, prediction_count,
            errors_m, "ground_truth_visibility_gap + recovery_arc_tolerance", len(records), 0,
        ),
        assignments,
    )


def summarise_visibility_gap_counts(gaps: pd.DataFrame) -> pd.DataFrame:
    """Report per-gap count accuracy for GPLA, bulges, guesses, and recovery."""
    rows: list[dict[str, Any]] = []
    stage_columns = {
        "gpla_raw": "gpla_peak_count",
        "accepted_bulges": "bulge_count",
        "guessed": "guessed_count",
        "recovery_only": "recovered_count",
    }
    for recovery_tolerance_mm, tolerance_gaps in gaps.groupby("recovery_tolerance_mm", sort=True):
        evaluable = tolerance_gaps.loc[
            tolerance_gaps["evaluation_status"].eq("evaluable")
        ].copy()
        for stage, column in stage_columns.items():
            if evaluable.empty:
                rows.append(
                    {
                        "stage": stage,
                        "recovery_tolerance_mm": recovery_tolerance_mm,
                        "total_visibility_gaps": len(tolerance_gaps),
                        "evaluable_gaps": 0,
                        "excluded_gaps": len(tolerance_gaps),
                        "true_missing_nodes": np.nan,
                        "predicted_nodes": np.nan,
                        "exact_count_accuracy": np.nan,
                        "count_mae": np.nan,
                        "count_bias": np.nan,
                    }
                )
                continue
            truth = evaluable["true_missing_count"].astype(int)
            predicted = evaluable[column].astype(int)
            errors = predicted - truth
            rows.append(
                {
                    "stage": stage,
                    "recovery_tolerance_mm": recovery_tolerance_mm,
                    "total_visibility_gaps": len(tolerance_gaps),
                    "evaluable_gaps": len(evaluable),
                    "excluded_gaps": len(tolerance_gaps) - len(evaluable),
                    "true_missing_nodes": int(truth.sum()),
                    "predicted_nodes": int(predicted.sum()),
                    "exact_count_accuracy": (predicted == truth).mean(),
                    "count_mae": errors.abs().mean(),
                    "count_bias": errors.mean(),
                }
            )
    return pd.DataFrame(rows)


def evaluate_cane(
    cane_id: int,
    ground_truth: pd.DataFrame,
    predictions: pd.DataFrame,
    recovery_tolerance_mm: float,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    """Evaluate recovery stages in ground-truth-defined gaps at one fixed tolerance."""
    summaries: list[dict[str, Any]] = []
    assignments: list[dict[str, Any]] = []

    visibility_gaps, records = build_visibility_gap_records(
        cane_id, ground_truth, predictions
    )

    gpla_summary, gpla_assignments = score_within_gap_stage(
        "gpla_raw", cane_id, recovery_tolerance_mm, records, "gpla"
    )
    summaries.append(gpla_summary); assignments.extend(gpla_assignments)
    bulge_summary, bulge_assignments = score_within_gap_stage(
        "accepted_bulges", cane_id, recovery_tolerance_mm, records, "bulges"
    )
    summaries.append(bulge_summary); assignments.extend(bulge_assignments)
    for record in records:
        matched = {index for index, _, _ in record.get("accepted_bulges_matches", [])}
        record["remaining_after_bulges"] = record["true_nodes"].drop(
            index=list(matched)
        ).reset_index(drop=True)
    guessed_summary, guessed_assignments = score_within_gap_stage(
        "guessed", cane_id, recovery_tolerance_mm, records, "guessed",
        "remaining_after_bulges",
    )
    summaries.append(guessed_summary); assignments.extend(guessed_assignments)
    recovery_summary, recovery_assignments = score_within_gap_stage(
        "recovery_only", cane_id, recovery_tolerance_mm, records, "recovered"
    )
    summaries.append(recovery_summary); assignments.extend(recovery_assignments)

    for summary in summaries:
        summary["excluded_gaps"] = 0
    return summaries, assignments, visibility_gaps


def pooled_summaries(
    per_cane_metrics: pd.DataFrame,
    assignments: pd.DataFrame,
) -> pd.DataFrame:
    """Pool node counts and matched-location errors across FCs without cross-cane matching."""
    rows: list[dict[str, Any]] = []
    for (stage, tolerance_mm), group in per_cane_metrics.groupby(["stage", "tolerance_mm"], sort=True):
        matched = assignments.loc[
            (assignments["stage"] == stage)
            & (assignments["tolerance_mm"] == tolerance_mm)
            & (assignments["match_status"] == "matched"),
            "absolute_error_mm",
        ].dropna()
        target_nodes = int(group["target_nodes"].sum())
        predicted_nodes = int(group["predicted_nodes"].sum())
        true_positive = int(group["TP"].sum())
        false_positive = int(group["FP"].sum())
        false_negative = int(group["FN"].sum())
        precision = true_positive / predicted_nodes if predicted_nodes else np.nan
        recall = true_positive / target_nodes if target_nodes else np.nan
        f1 = (
            2 * precision * recall / (precision + recall)
            if np.isfinite(precision) and np.isfinite(recall) and precision + recall > 0
            else np.nan
        )
        rows.append(
            {
                "stage": stage,
                "cane_id": "all_canes_pooled",
                "tolerance_mm": tolerance_mm,
                "matching_basis": group["matching_basis"].iloc[0],
                "evaluable_gaps": group["evaluable_gaps"].sum(min_count=1),
                "excluded_gaps": group["excluded_gaps"].sum(min_count=1),
                "target_nodes": target_nodes,
                "predicted_nodes": predicted_nodes,
                "TP": true_positive,
                "FP": false_positive,
                "FN": false_negative,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "count_error": int(group["count_error"].sum()),
                "count_absolute_error": int(group["count_absolute_error"].sum()),
                "matched_nodes": true_positive,
                "mean_localisation_error_mm": matched.mean() if len(matched) else np.nan,
                "median_localisation_error_mm": matched.median() if len(matched) else np.nan,
                "rmse_localisation_error_mm": np.sqrt(np.mean(matched.to_numpy() ** 2)) if len(matched) else np.nan,
                "p95_localisation_error_mm": matched.quantile(0.95) if len(matched) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def diameter_metrics(match_assignments: pd.DataFrame) -> pd.DataFrame:
    """Assess diameter for positional matches among recovered nodes only."""
    matched = match_assignments.loc[
        (match_assignments["stage"] == "recovery_only")
        & np.isclose(match_assignments["tolerance_mm"], RECOVERY_TOLERANCE_MM)
        & (match_assignments["match_status"] == "matched")
    ].copy()
    matched = matched.dropna(subset=["ground_truth_diameter_mm", "predicted_diameter_mm"])
    if matched.empty:
        return pd.DataFrame(
            [{"cane_id": "all_canes_pooled", "matched_diameters": 0, "diameter_mae_mm": np.nan,
              "diameter_bias_mm": np.nan, "diameter_rmse_mm": np.nan}]
        )

    matched["diameter_error_mm"] = matched["predicted_diameter_mm"] - matched["ground_truth_diameter_mm"]
    rows: list[dict[str, Any]] = []
    for cane_id, group in list(matched.groupby("cane_id", sort=True)) + [("all_canes_pooled", matched)]:
        errors = group["diameter_error_mm"].to_numpy(dtype=float)
        rows.append(
            {
                "cane_id": cane_id,
                "matched_diameters": len(errors),
                "diameter_mae_mm": np.mean(np.abs(errors)),
                "diameter_bias_mm": np.mean(errors),
                "diameter_rmse_mm": np.sqrt(np.mean(errors**2)),
            }
        )
    return pd.DataFrame(rows)


def score_optional_pnm(
    pnm_path: Path | None,
    ground_truth: pd.DataFrame,
    visibility_gaps: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    """Score PNM counts against visibility-defined gaps matched by order."""
    if pnm_path is None:
        print(
            "PNM/MGC gap-level evaluation skipped: no file at "
            f"{PNM_GAP_XLSX_PATH} or {PNM_GAP_CSV_PATH}"
        )
        return None

    pnm = read_table(pnm_path)
    require_columns(pnm, PNM_REQUIRED, "PNM gap table")
    pnm = pnm.copy()
    pnm["cane_id"] = pd.to_numeric(pnm["cane_id"], errors="raise").astype(int)
    pnm["gap_id"] = pnm["gap_id"].astype(str)
    pnm["gap_index"] = pd.to_numeric(pnm["gap_index"], errors="raise").astype(int)
    for column in ["start_arc_m", "end_arc_m"]:
        pnm[column] = pd.to_numeric(pnm[column], errors="coerce")
    for column in ["mgc_prediction", "pnm_map_count", "pnm_lower_count", "pnm_upper_count"]:
        pnm[column] = pd.to_numeric(pnm[column], errors="coerce")

    if pnm.duplicated(["cane_id", "gap_id"]).any():
        raise ValueError("PNM gap_id values must be unique within each cane.")
    if pnm.duplicated(["cane_id", "gap_index"]).any():
        raise ValueError("PNM gap_index values must be unique within each cane.")
    if pnm[["start_arc_m", "end_arc_m", "mgc_prediction"]].isna().any().any():
        raise ValueError("PNM gap arc bounds and MGC predictions must be non-missing.")
    if (pnm["end_arc_m"] <= pnm["start_arc_m"]).any():
        raise ValueError("PNM gap end_arc_m must be greater than start_arc_m.")
    if not pnm["mgc_prediction"].dropna().isin([0, 1]).all():
        raise ValueError("mgc_prediction must contain only 0 or 1.")

    # Retain the former raw-arc count only to expose coordinate-frame
    # disagreements; it is not used as the PNM ground-truth label.
    arc_interval_counts: list[float] = []
    for row in pnm.itertuples(index=False):
        cane_gt = ground_truth.loc[ground_truth["cane_id"].eq(row.cane_id)]
        if cane_gt.empty:
            arc_interval_counts.append(np.nan)
            continue
        hidden_nodes = cane_gt.loc[~cane_gt["is_recon_detected"]]
        in_gap = hidden_nodes["arc_m"].gt(row.start_arc_m) & hidden_nodes["arc_m"].lt(row.end_arc_m)
        arc_interval_counts.append(float(in_gap.sum()))
    pnm["arc_interval_true_missing_count"] = arc_interval_counts

    visibility_required = {
        "cane_id",
        "gap_index",
        "visibility_gap_id",
        "evaluation_status",
        "true_missing_count",
    }
    require_columns(visibility_gaps, visibility_required, "visibility-gap table")
    visibility_truth = visibility_gaps[
        [
            "cane_id",
            "gap_index",
            "visibility_gap_id",
            "evaluation_status",
            "true_missing_count",
        ]
    ].copy()
    visibility_truth["cane_id"] = pd.to_numeric(
        visibility_truth["cane_id"], errors="raise"
    ).astype(int)
    visibility_truth["gap_index"] = pd.to_numeric(
        visibility_truth["gap_index"], errors="raise"
    ).astype(int)
    visibility_truth["true_missing_count"] = pd.to_numeric(
        visibility_truth["true_missing_count"], errors="coerce"
    )
    if visibility_truth.duplicated(["cane_id", "gap_index"]).any():
        raise ValueError(
            "Visibility gap_index values must be unique within each cane."
        )

    scored = pnm.merge(
        visibility_truth,
        on=["cane_id", "gap_index"],
        how="left",
        validate="one_to_one",
        indicator=True,
    )
    missing_truth = scored["_merge"].ne("both")
    if missing_truth.any():
        missing_keys = scored.loc[missing_truth, ["cane_id", "gap_index"]]
        raise ValueError(
            "PNM gaps have no matching visibility-defined gap: "
            f"{missing_keys.to_dict(orient='records')}"
        )
    scored = scored.drop(columns="_merge")
    scored["gap_evaluation_status"] = np.where(
        scored["evaluation_status"].eq("evaluable"),
        "evaluated_by_visibility_gap",
        "visibility_gap_not_evaluable",
    )
    scored["arc_interval_minus_visibility_count"] = (
        scored["arc_interval_true_missing_count"] - scored["true_missing_count"]
    )

    eligible = scored.loc[
        scored["gap_evaluation_status"].eq("evaluated_by_visibility_gap")
    ].copy()
    if eligible.empty:
        print("PNM/MGC gap-level evaluation skipped: no visibility-defined gap was evaluable.")
        return scored, pd.DataFrame()

    y_true = eligible["true_missing_count"].gt(0).astype(int)
    y_pred = eligible["mgc_prediction"].astype(int)
    true_positive = int(((y_true == 1) & (y_pred == 1)).sum())
    false_positive = int(((y_true == 0) & (y_pred == 1)).sum())
    false_negative = int(((y_true == 1) & (y_pred == 0)).sum())
    true_negative = int(((y_true == 0) & (y_pred == 0)).sum())

    pnm_positive = eligible.loc[
        eligible["mgc_prediction"].eq(1)
        & eligible[["pnm_map_count", "pnm_lower_count", "pnm_upper_count"]].notna().all(axis=1)
    ].copy()
    count_error = pnm_positive["pnm_map_count"] - pnm_positive["true_missing_count"]
    coverage = (
        (pnm_positive["true_missing_count"] >= pnm_positive["pnm_lower_count"])
        & (pnm_positive["true_missing_count"] <= pnm_positive["pnm_upper_count"])
    )
    pnm_metrics = pd.DataFrame(
        [
            {
                "evaluation_basis": "visibility_defined_gap_by_cane_id_and_gap_index",
                "eligible_gaps": len(eligible),
                "MGC_TP": true_positive,
                "MGC_FP": false_positive,
                "MGC_FN": false_negative,
                "MGC_TN": true_negative,
                "MGC_accuracy": (true_positive + true_negative) / len(eligible),
                "MGC_precision": true_positive / (true_positive + false_positive)
                if true_positive + false_positive else np.nan,
                "MGC_recall": true_positive / (true_positive + false_negative)
                if true_positive + false_negative else np.nan,
                "PNM_evaluated_gaps": len(pnm_positive),
                "PNM_exact_count_accuracy": (pnm_positive["pnm_map_count"] == pnm_positive["true_missing_count"]).mean()
                if len(pnm_positive) else np.nan,
                "PNM_count_MAE": count_error.abs().mean() if len(count_error) else np.nan,
                "PNM_count_bias": count_error.mean() if len(count_error) else np.nan,
                "PNM_credible_interval_coverage": coverage.mean() if len(coverage) else np.nan,
            }
        ]
    )
    return scored, pnm_metrics


def main() -> None:
    ground_truth = prepare_ground_truth(read_table(GROUND_TRUTH_PATH))
    predictions = prepare_predictions(read_table(locate_prediction_table()))

    shared_canes = sorted(set(ground_truth["cane_id"]).intersection(predictions["cane_id"]))
    if not shared_canes:
        raise ValueError("Ground-truth and prediction tables have no shared cane_id values.")

    missing_from_predictions = sorted(set(ground_truth["cane_id"]).difference(predictions["cane_id"]))
    missing_from_ground_truth = sorted(set(predictions["cane_id"]).difference(ground_truth["cane_id"]))
    if missing_from_predictions or missing_from_ground_truth:
        raise ValueError(
            "Ground-truth and prediction cane_id values must match. "
            f"Missing predictions: {missing_from_predictions}; "
            f"missing ground truth: {missing_from_ground_truth}."
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    detected_counts = count_only_metrics(
        ground_truth, predictions, "detected_count", shared_canes
    )
    overall_counts = count_only_metrics(
        ground_truth, predictions, "overall_node_count", shared_canes
    )
    all_summaries: list[dict[str, Any]] = []
    all_assignments: list[dict[str, Any]] = []
    all_visibility_gaps: list[dict[str, Any]] = []

    for cane_id in shared_canes:
        cane_gt = ground_truth.loc[ground_truth["cane_id"].eq(cane_id)].reset_index(drop=True)
        cane_predictions = predictions.loc[predictions["cane_id"].eq(cane_id)].reset_index(drop=True)
        summaries, assignments, visibility_gaps = evaluate_cane(
            cane_id, cane_gt, cane_predictions, RECOVERY_TOLERANCE_MM
        )
        all_summaries.extend(summaries)
        all_assignments.extend(assignments)
        for gap in visibility_gaps:
            gap["recovery_tolerance_mm"] = RECOVERY_TOLERANCE_MM
        all_visibility_gaps.extend(visibility_gaps)

    per_cane_metrics = pd.DataFrame(all_summaries)
    assignments = pd.DataFrame(all_assignments)
    pooled_metrics = pooled_summaries(per_cane_metrics, assignments)
    metrics = pd.concat([per_cane_metrics, pooled_metrics], ignore_index=True)
    visibility_gaps = pd.DataFrame(all_visibility_gaps)

    assignments.to_csv(OUTPUT_DIR / "node_match_assignments.csv", index=False)
    detected_counts.to_csv(OUTPUT_DIR / "detected_node_count_metrics.csv", index=False)
    overall_counts.to_csv(OUTPUT_DIR / "overall_node_count_metrics.csv", index=False)
    per_cane_metrics.to_csv(OUTPUT_DIR / "stagewise_metrics_by_cane.csv", index=False)
    pooled_metrics.to_csv(OUTPUT_DIR / "stagewise_metrics_pooled.csv", index=False)
    metrics.to_csv(OUTPUT_DIR / "stagewise_metrics_all.csv", index=False)
    visibility_gaps.to_csv(OUTPUT_DIR / "visibility_gap_evaluation.csv", index=False)
    summarise_visibility_gap_counts(visibility_gaps).to_csv(
        OUTPUT_DIR / "visibility_gap_count_metrics.csv", index=False
    )
    primary_assignments = assignments.loc[
        np.isclose(assignments["tolerance_mm"], RECOVERY_TOLERANCE_MM)
    ].copy()
    diameter_metrics(primary_assignments).to_csv(OUTPUT_DIR / "diameter_metrics.csv", index=False)

    pnm_gap_path = locate_pnm_gap_table()
    pnm_result = score_optional_pnm(pnm_gap_path, ground_truth, visibility_gaps)
    if pnm_result is not None:
        pnm_gaps, pnm_metrics = pnm_result
        pnm_gaps.to_csv(OUTPUT_DIR / "pnm_gap_evaluation.csv", index=False)
        pnm_metrics.to_csv(OUTPUT_DIR / "pnm_mgc_metrics.csv", index=False)

    print(
        f"Evaluated {len(shared_canes)} canes using count-only detected-node "
        f"summaries and {RECOVERY_TOLERANCE_MM:.1f} mm recovery tolerance."
    )
    print(f"Saved results to: {OUTPUT_DIR}")
    if pnm_gap_path is None:
        print("Add pnm_gap_results.xlsx or .csv to obtain MGC and PNM gap-level metrics.")
    if (OUTPUT_DIR / "detection_tolerance_sensitivity.csv").exists() or (
        OUTPUT_DIR / "tolerance_sensitivity_pooled.csv"
    ).exists():
        print(
            "Sensitivity CSVs from earlier runs were not regenerated and may be stale; "
            "the current evaluation does not use them."
        )


if __name__ == "__main__":
    main()
