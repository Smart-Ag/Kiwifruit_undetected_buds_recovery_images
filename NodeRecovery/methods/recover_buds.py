"""
Recover undetected buds by fusing the morphometric gap classifier (COUNT) with
the GPDA bulge detector (POSITION).
================================================================================
Standalone: give it the paths below (skeleton, flattened branch, detected buds,
fpv, model weights) -- no pipeline needed. Everything runs in ONE polyline
arc-length frame so counts and bulge positions share the same ruler.

Per observed gap (between consecutive detected buds or a fixed skeleton end):
  1. classifier flags the gap as missing-or-not; the probabilistic model gives
     the hidden-node count n_g = k - 1 (MAP) and a credible range [n_lo, n_hi],
     where k is the number of internodes spanning the gap.
  2. GPDA gives the bulges inside the gap (b), ranked by prominence.
  3. reconcile:
       n_lo <= b <= n_hi  -> accept the b bulges            (ORANGE, bulge-backed)
       b < n_lo           -> accept b bulges + fill (n-b)    (YELLOW, guessed)
                              guessed buds at POPULATION-MEAN spacing
       b > n_hi           -> strongest n_hi as buds (ORANGE) + extras (GREY candidate)
  4. detected buds stay RED.

Diameter (X-Y silhouette width, 15 mm apical) is measured for every bud.
Outputs: ordered CSV, a Times-New-Roman PNG, and a colour-coded .ply.
"""

import os
import csv
import json
import numpy as np
import open3d as o3d
import pickle
import pandas as pd
from glob import glob
from scipy.spatial import cKDTree
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from get_bud_data import (load_point_cloud_xyz, find_skeleton_extremes,
                          find_closest_skeleton_segment, polyline_arc,
                          point_tangent_at_arc, measure_diameter,
                          BUD_DIAM_OFFSET)
from detect_buds_bulge import find_bulges

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["mathtext.fontset"] = "stix"

# ==================== PATHS ====================
CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CASE_DIR = os.environ.get("NR_CASE_DIR", os.path.join(CODE_DIR, "case"))
# Must equal the numeric cane_id used for this FC in the ground-truth table.
EVALUATION_CANE_ID = int(os.environ.get("NR_CANE_ID", "1"))
CASE_LABEL = os.environ.get("NR_CASE_LABEL", os.path.basename(CASE_DIR))
OUT_DIR = os.environ.get("NR_RECOVERY_OUT_DIR", os.path.join(CODE_DIR, "runs", CASE_LABEL))
SKELETON_PLY_PATH = os.environ.get("NR_SKELETON_PLY", os.path.join(CASE_DIR, "in", "skeleton.ply"))
SMOOTH_PLY_PATH = os.environ.get("NR_SURFACE_PLY", os.path.join(CASE_DIR, "in", "smooth.ply"))
FPV_PLY_PATH = os.environ.get("NR_FPV_PLY", os.path.join(CASE_DIR, "in", "fpv_raw.ply"))
BUDS_DIR = os.environ.get("NR_BUDS_DIR", os.path.join(CASE_DIR, "buds"))
MODEL_PATH = os.environ.get("NR_MODEL_PATH", os.path.join(CODE_DIR, "models", "selected_model_mask_0.5.pkl"))
ORIGINAL_IMAGE_PATH = os.environ.get("NR_IMAGE_PATH", "")

OUT_CSV = os.path.join(OUT_DIR, f"{CASE_LABEL}_recovered_buds.csv")
OUT_IMG = os.path.join(OUT_DIR, f"{CASE_LABEL}_recovered_buds.png")
OUT_IMG_OVERLAY = os.path.join(OUT_DIR, f"{CASE_LABEL}_recovered_buds_on_original.png")
OUT_VIZ = os.path.join(OUT_DIR, f"{CASE_LABEL}_recovered_buds_viz.ply")
OUT_TABLE = os.path.join(OUT_DIR, f"{CASE_LABEL}_recovered_buds_table.png")

# Automatically generated inputs for evaluate_recovery_pipeline.py.  The
# case-specific files retain the current single-cane audit record; the combined
# files are updated by replacing rows for EVALUATION_CANE_ID on each rerun.
OUT_EVAL_NODES = os.path.join(OUT_DIR, f"{CASE_LABEL}_prediction_evaluation.csv")
OUT_EVAL_GAPS = os.path.join(OUT_DIR, f"{CASE_LABEL}_pnm_gap_results.csv")
OUT_EVAL_NODES_COMBINED = os.environ.get("NR_COMBINED_NODES_CSV", os.path.join(OUT_DIR, "recon_recovered.csv"))
OUT_EVAL_GAPS_COMBINED = os.environ.get("NR_COMBINED_GAPS_CSV", os.path.join(OUT_DIR, "pnm_gap_results.csv"))
# ===============================================

FEATURE_COLUMNS = ["gap_length_m", "mean_IL_m", "std_IL_m", "rel_position"]

# Intrinsics used to back-project this RGB-D acquisition into the input clouds.
CAMERA_INTRINSICS = json.loads(os.environ.get(
    "NR_CAMERA_INTRINSICS_JSON",
    '{"fx": 693.553467, "fy": 693.492859, "cx": 643.257935, "cy": 358.344116}',
))

POPULATION_MEAN_IL = float(os.environ.get("NR_PNM_MEAN_IL_M", "0.08406"))
POPULATION_STD_IL = float(os.environ.get("NR_PNM_STD_IL_M", "0.02730"))
KMAX = int(os.environ.get("NR_PNM_KMAX", "25"))
PNM_CREDIBLE_LEVEL = float(os.environ.get("NR_PNM_CREDIBLE_LEVEL", "0.90"))

PROMINENCE = float(os.environ.get("NR_PROMINENCE_M", "0.0022"))
MIN_SEP = float(os.environ.get("NR_MIN_SEP_M", "0.04"))
MATCH_TOL = float(os.environ.get("NR_DEDUP_TOLERANCE_M", "0.025"))
MARK_RADIUS = 0.003           # m, marker radius in the .ply

COLORS = {                    # RGB 0..1
    "detected":  [1.0, 0.0, 0.0],   # red
    "bulge":     [1.0, 0.55, 0.0],  # orange
    "guessed":   [1.0, 1.0, 0.0],   # yellow
    "candidate": [0.5, 0.5, 0.5],   # grey
}


def save_evaluation_table(
    data: pd.DataFrame,
    case_path: str,
    combined_path: str,
    sort_columns: list[str],
) -> None:
    """Save a case audit table and upsert its rows in the multi-cane table."""
    os.makedirs(os.path.dirname(case_path), exist_ok=True)
    data.to_csv(case_path, index=False)

    if os.path.exists(combined_path):
        combined = pd.read_csv(combined_path)
        if "cane_id" not in combined.columns:
            raise ValueError(
                f"Cannot update {combined_path}: existing table has no cane_id column."
            )
        existing_cane_id = pd.to_numeric(combined["cane_id"], errors="coerce")
        combined = combined.loc[existing_cane_id.ne(EVALUATION_CANE_ID)].copy()
        combined = pd.concat([combined, data], ignore_index=True, sort=False)
    else:
        combined = data.copy()

    combined = combined.sort_values(sort_columns).reset_index(drop=True)
    combined.to_csv(combined_path, index=False)
    print(f"Saved evaluation export: {case_path}")
    print(f"Updated multi-cane evaluation table: {combined_path}")


# ---------------- probabilistic count (gap = sum of k internodes) -------------
def count_probabilistic(
    gap,
    mu=POPULATION_MEAN_IL,
    sigma=POPULATION_STD_IL,
    kmax=KMAX,
    credible_level=PNM_CREDIBLE_LEVEL,
):
    """Return the PNM posterior summary for one gap.

    Candidate k is the number of internodes spanning the gap, with a discrete
    uniform prior over 1..kmax. Exported node counts use n_g = k - 1. The
    interval is an equal-tail credible interval with mass ``credible_level``.
    """
    if gap <= 0 or mu <= 0 or sigma <= 0 or kmax < 1:
        raise ValueError("PNM requires gap, mu, sigma, and kmax to be positive.")
    if not 0.0 < credible_level < 1.0:
        raise ValueError("credible_level must lie strictly between 0 and 1.")

    ks = np.arange(1, kmax + 1)
    var = ks * sigma ** 2
    log_like = -0.5 * np.log(2 * np.pi * var) - 0.5 * (gap - ks * mu) ** 2 / var
    log_prior = np.full(len(ks), -np.log(len(ks)), dtype=float)
    log_unnormalized = log_like + log_prior
    unnormalized = np.exp(log_unnormalized - np.max(log_unnormalized))
    post = unnormalized / unnormalized.sum()

    alpha = (1.0 - credible_level) / 2.0
    cum = np.cumsum(post)
    map_index = int(np.argmax(post))
    map_k = int(ks[map_index])
    lo_k = int(ks[min(np.searchsorted(cum, alpha), len(ks) - 1)])
    hi_k = int(ks[min(np.searchsorted(cum, 1.0 - alpha), len(ks) - 1)])

    node_map = max(0, map_k - 1)
    node_lo = max(0, lo_k - 1)
    node_hi = max(0, hi_k - 1)
    return {
        "k_map": map_k,
        "k_lo": lo_k,
        "k_hi": hi_k,
        "node_map": node_map,
        "node_lo": node_lo,
        "node_hi": node_hi,
        # Backward-compatible aliases used by the current reconciliation code.
        "map": node_map,
        "lo": node_lo,
        "hi": node_hi,
        "map_probability": float(post[map_index]),
        "credible_level": float(credible_level),
        "prior": "discrete_uniform",
        "k_min": 1,
        "k_max": int(kmax),
        "mu_m": float(mu),
        "sigma_m": float(sigma),
        "candidate_probabilities": "|".join(
            f"{int(k)}:{float(probability):.12g}"
            for k, probability in zip(ks, post)
        ),
    }


def project_to_image(points_xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Project camera-coordinate 3-D points into RGB pixel coordinates."""
    points_xyz = np.atleast_2d(np.asarray(points_xyz, dtype=float))
    pixels = np.full((len(points_xyz), 2), np.nan, dtype=float)
    valid = np.isfinite(points_xyz).all(axis=1) & (points_xyz[:, 2] > 0)
    if valid.any():
        points = points_xyz[valid]
        pixels[valid, 0] = (
            CAMERA_INTRINSICS["fx"] * points[:, 0] / points[:, 2]
            + CAMERA_INTRINSICS["cx"]
        )
        pixels[valid, 1] = (
            CAMERA_INTRINSICS["fy"] * points[:, 1] / points[:, 2]
            + CAMERA_INTRINSICS["cy"]
        )
    return pixels, valid


def save_original_image_overlay(surface, skeleton, buds) -> None:
    """Save the recovery visualisation over the registered colour image."""
    if not os.path.exists(ORIGINAL_IMAGE_PATH):
        raise FileNotFoundError(f"Original image not found: {ORIGINAL_IMAGE_PATH}")

    # The acquisition exports JPEG-encoded colour frames with a .png suffix.
    # Pillow identifies the format from the file signature, unlike plt.imread,
    # which selects a PNG decoder solely from the filename extension.
    with Image.open(ORIGINAL_IMAGE_PATH) as source_image:
        image = np.asarray(source_image.convert("RGB"))
    image_height, image_width = image.shape[:2]
    fig, ax = plt.subplots(figsize=(image_width / 120, image_height / 120), dpi=120)
    ax.imshow(image, origin="upper", zorder=0)

    surface_px, surface_valid = project_to_image(surface)
    ax.scatter(
        surface_px[surface_valid, 0],
        surface_px[surface_valid, 1],
        s=1.5,
        c="white",
        alpha=0.35,
        linewidths=0,
        zorder=1,
        label="surface point cloud",
    )
    skeleton_px, skeleton_valid = project_to_image(skeleton)
    ax.plot(
        skeleton_px[skeleton_valid, 0],
        skeleton_px[skeleton_valid, 1],
        "-",
        c="white",
        alpha=0.9,
        lw=1.2,
        zorder=2,
        label="skeleton",
    )

    for b in buds:
        if b["line"] is not None:
            line_px, line_valid = project_to_image(np.vstack(b["line"]))
            if line_valid.all():
                ax.plot(
                    line_px[:, 0], line_px[:, 1], "-", c="royalblue", lw=1.0, zorder=3
                )

    for bud_type in ["candidate", "guessed", "bulge", "detected"]:
        points = np.array([b["pos"] for b in buds if b["type"] == bud_type])
        if len(points):
            points_px, valid = project_to_image(points)
            ax.scatter(
                points_px[valid, 0],
                points_px[valid, 1],
                s=70,
                c=[COLORS[bud_type]],
                edgecolor="black",
                linewidths=0.5,
                zorder=5,
                label=bud_type,
            )

    ax.set_xlim(0, image_width)
    ax.set_ylim(image_height, 0)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.legend(loc="upper right", frameon=True, framealpha=0.85, fontsize=8)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    fig.savefig(OUT_IMG_OVERLAY, dpi=120)
    plt.close(fig)
    print(f"Saved {OUT_IMG_OVERLAY}")


def main():
    for description, path in (
        ("skeleton", SKELETON_PLY_PATH),
        ("smoothed surface", SMOOTH_PLY_PATH),
        ("basal-origin/FPV point cloud", FPV_PLY_PATH),
        ("trained MGC model", MODEL_PATH),
    ):
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Missing {description}: {path}")
    if not os.path.isdir(BUDS_DIR):
        raise FileNotFoundError(f"Missing detected-bud PLY directory: {BUDS_DIR}")
    os.makedirs(OUT_DIR, exist_ok=True)
    sk      = load_point_cloud_xyz(SKELETON_PLY_PATH)
    surface = load_point_cloud_xyz(SMOOTH_PLY_PATH)
    fpv     = load_point_cloud_xyz(FPV_PLY_PATH)
    cum, _  = polyline_arc(sk)
    total   = cum[-1]

    # basal -> apex direction; work in arc-from-basal coordinate
    ep1, ep2 = find_skeleton_extremes(sk)
    fpvc = fpv.mean(axis=0)
    basal_is_ep1 = np.linalg.norm(ep1 - fpvc) < np.linalg.norm(ep2 - fpvc)
    basal, apex = (ep1, ep2) if basal_is_ep1 else (ep2, ep1)
    tree = cKDTree(sk)
    basal_node, apex_node = int(tree.query(basal)[1]), int(tree.query(apex)[1])
    apex_dir = 1.0 if cum[apex_node] >= cum[basal_node] else -1.0
    a2b = lambda a: a if apex_dir > 0 else (total - a)        # raw arc -> arc-from-basal
    b2a = lambda ab: ab if apex_dir > 0 else (total - ab)     # arc-from-basal -> raw arc
    print(f"skeleton arc {total:.3f} m  basal->apex {'+arc' if apex_dir>0 else '-arc'}")

    def centerline_at(arc_basal):
        return point_tangent_at_arc(sk, cum, b2a(arc_basal))[0]

    # ---- detected buds (RED) ----
    detected = []
    for bp in sorted(glob(os.path.join(BUDS_DIR, "*.ply"))):
        xyz = load_point_cloud_xyz(bp)
        if xyz.shape[0] != 1:
            continue
        seg, proj, t = find_closest_skeleton_segment(xyz[0], sk)
        arc_basal = a2b(cum[seg] + t * (cum[seg + 1] - cum[seg]))
        detected.append({"type": "detected", "source": os.path.basename(bp),
                         "arc_basal": float(arc_basal), "pos": proj,
                         "prominence": np.nan})
    detected.sort(key=lambda b: b["arc_basal"])
    for detected_index, bud in enumerate(detected, start=1):
        bud["evaluation_id"] = (
            f"cane_{EVALUATION_CANE_ID}_detected_{detected_index:03d}"
        )
    print(f"detected buds: {len(detected)}")

    # ---- GPDA bulges (in arc-from-basal), dedup against detected ----
    raw_bulges, _ = find_bulges(surface, sk, prominence=PROMINENCE, min_sep=MIN_SEP)
    det_arcs = np.array([b["arc_basal"] for b in detected])
    bulges = []
    for r in raw_bulges:
        ab = a2b(r["arc"])
        if len(det_arcs) and np.min(np.abs(det_arcs - ab)) < MATCH_TOL:
            continue
        bulges.append({"arc_basal": float(ab), "pos": r["center"],
                       "prominence": r["prominence"]})
    bulges.sort(key=lambda b: b["arc_basal"])
    print(f"GPDA bulges (not on a detected bud): {len(bulges)}")

    # ---- observed gaps with fixed physical skeleton-end references ----
    # The virtual coordinates 0 and L_FC are reference points, not buds.  They
    # create basal and apical gaps and are deliberately excluded from `buds`,
    # so they can never be written as detected or recovered nodes.
    branch_len = float(total)
    reference_points = [
        {
            "reference_id": f"cane_{EVALUATION_CANE_ID}_skeleton_base",
            "reference_type": "skeleton_base",
            "arc_basal": 0.0,
        }
    ]
    reference_points.extend(
        {
            "reference_id": bud["evaluation_id"],
            "reference_type": "detected",
            "arc_basal": bud["arc_basal"],
        }
        for bud in detected
        if 0.0 < bud["arc_basal"] < branch_len
    )
    reference_points.append(
        {
            "reference_id": f"cane_{EVALUATION_CANE_ID}_skeleton_apex",
            "reference_type": "skeleton_apex",
            "arc_basal": branch_len,
        }
    )
    reference_arcs = np.array(
        [point["arc_basal"] for point in reference_points], dtype=float
    )
    if np.any(np.diff(reference_arcs) <= 0):
        raise ValueError(
            "Detected-bud reference arcs must be unique and strictly ordered within "
            "the skeleton interval (0, L_FC)."
        )
    ils = np.diff(reference_arcs)
    mean_IL = float(ils.mean()) if len(ils) else np.nan
    std_IL  = float(ils.std(ddof=1)) if len(ils) > 1 else np.nan

    def gap_type(lo_a, hi_a):
        if np.isclose(lo_a, 0.0) and np.isclose(hi_a, branch_len):
            return "basal_apical"
        if np.isclose(lo_a, 0.0):
            return "basal"
        if np.isclose(hi_a, branch_len):
            return "apical"
        return "internal"

    # ---- per-gap features + classifier flag ----
    feats = []
    for lo_a, hi_a in zip(reference_arcs[:-1], reference_arcs[1:]):
        gap = hi_a - lo_a
        rel = lo_a / branch_len if branch_len else np.nan
        feats.append([gap, mean_IL, std_IL, rel])
    X = pd.DataFrame(feats, columns=FEATURE_COLUMNS)

    with open(MODEL_PATH, "rb") as model_file:
        model = pickle.load(model_file)
    flags = model.predict(X) if len(X) else np.array([])
    if len(X) and hasattr(model, "predict_proba"):
        mgc_scores = model.predict_proba(X)[:, 1]
    elif len(X) and hasattr(model, "decision_function"):
        mgc_scores = model.decision_function(X)
    else:
        mgc_scores = np.full(len(X), np.nan)
    print(f"Loaded MGC model: {MODEL_PATH}")
    print(f"gaps: {len(X)}  flagged missing: {int(np.sum(flags))}")

    # ---- reconcile per gap ----
    recovered = []   # bulge / guessed / candidate buds
    gap_rows = []    # rows for the reconciliation table
    gap_evaluation_records = []

    def record_gap_evaluation(
        gap_index,
        left_reference,
        right_reference,
        current_gap_type,
        lo_a,
        hi_a,
        gpla_peak_count,
        recovered_start,
        pnm_counts,
        reconciliation_action,
    ):
        """Retain all classifier, PNM, GPLA, and reconciliation data for evaluation."""
        gap_nodes = recovered[recovered_start:]
        gap_evaluation_records.append(
            {
                "cane_id": EVALUATION_CANE_ID,
                "gap_id": f"cane_{EVALUATION_CANE_ID}_gap_{gap_index:03d}",
                "gap_index": gap_index,
                "gap_type": current_gap_type,
                "start_predicted_id": left_reference["reference_id"],
                "end_predicted_id": right_reference["reference_id"],
                "start_reference_type": left_reference["reference_type"],
                "end_reference_type": right_reference["reference_type"],
                "start_arc_m": lo_a,
                "end_arc_m": hi_a,
                "gap_length_m": hi_a - lo_a,
                "mean_IL_m": mean_IL,
                "std_IL_m": std_IL,
                "rel_position": lo_a / branch_len if branch_len else np.nan,
                "mgc_prediction": int(flags[gap_index - 1]),
                "mgc_score": float(mgc_scores[gap_index - 1]),
                "pnm_evaluated": pnm_counts is not None,
                "pnm_applied": bool(flags[gap_index - 1]),
                "pnm_prior": np.nan if pnm_counts is None else pnm_counts["prior"],
                "pnm_k_min": np.nan if pnm_counts is None else pnm_counts["k_min"],
                "pnm_k_max": np.nan if pnm_counts is None else pnm_counts["k_max"],
                "pnm_k_map": np.nan if pnm_counts is None else pnm_counts["k_map"],
                "pnm_k_lower": np.nan if pnm_counts is None else pnm_counts["k_lo"],
                "pnm_k_upper": np.nan if pnm_counts is None else pnm_counts["k_hi"],
                "pnm_node_count": np.nan if pnm_counts is None else pnm_counts["node_map"],
                "pnm_node_lower": np.nan if pnm_counts is None else pnm_counts["node_lo"],
                "pnm_node_upper": np.nan if pnm_counts is None else pnm_counts["node_hi"],
                # Retain the original column names for evaluate_recovery_pipeline.py.
                "pnm_map_count": np.nan if pnm_counts is None else pnm_counts["node_map"],
                "pnm_lower_count": np.nan if pnm_counts is None else pnm_counts["node_lo"],
                "pnm_upper_count": np.nan if pnm_counts is None else pnm_counts["node_hi"],
                "pnm_map_probability": (
                    np.nan if pnm_counts is None else pnm_counts["map_probability"]
                ),
                "pnm_credible_level": (
                    np.nan if pnm_counts is None else pnm_counts["credible_level"]
                ),
                "pnm_mu_m": np.nan if pnm_counts is None else pnm_counts["mu_m"],
                "pnm_sigma_m": np.nan if pnm_counts is None else pnm_counts["sigma_m"],
                "pnm_candidate_probabilities": (
                    "" if pnm_counts is None else pnm_counts["candidate_probabilities"]
                ),
                "gpla_peak_count": gpla_peak_count,
                "bulge_count": sum(node["type"] == "bulge" for node in gap_nodes),
                "guessed_count": sum(node["type"] == "guessed" for node in gap_nodes),
                "candidate_count": sum(node["type"] == "candidate" for node in gap_nodes),
                "reconciliation_action": reconciliation_action,
                "_gap_nodes": gap_nodes,
            }
        )

    print("\nper-gap reconciliation:")
    for i, (lo_a, hi_a) in enumerate(zip(reference_arcs[:-1], reference_arcs[1:])):
        left_reference = reference_points[i]
        right_reference = reference_points[i + 1]
        recovered_start = len(recovered)
        gap = hi_a - lo_a
        current_gap_type = gap_type(lo_a, hi_a)
        inside = [b for b in bulges if lo_a < b["arc_basal"] < hi_a]
        inside.sort(key=lambda b: -b["prominence"])
        b = len(inside)
        # Evaluate the PNM for every gap for audit purposes. The estimate is
        # used by the recovery algorithm only when the MGC flags the gap.
        c = count_probabilistic(gap)
        if not flags[i]:
            for x in inside:                       # bulges in an unflagged gap -> candidates
                recovered.append({**x, "type": "candidate", "source": "bulge"})
            result = (
                f"{current_gap_type}; not flagged ({b} candidate)"
                if b
                else f"{current_gap_type}; not flagged"
            )
            gap_rows.append({"cells": [f"{lo_a:.3f}-{hi_a:.3f}", f"{gap:.3f}", "no",
                                       str(b), "-", result], "cat": "not_flagged"})
            if b:
                print(
                    f"  {current_gap_type} gap[{lo_a:.3f},{hi_a:.3f}] "
                    f"not flagged: {b} bulge(s) -> candidate"
                )
            record_gap_evaluation(
                i + 1, left_reference, right_reference, current_gap_type,
                lo_a, hi_a, b, recovered_start, c, "not_flagged"
            )
            continue

        n_map, n_lo, n_hi = c["node_map"], c["node_lo"], c["node_hi"]
        action, cat = "", ""
        if n_lo <= b <= n_hi:                      # agree
            for x in inside:
                recovered.append({**x, "type": "bulge", "source": "bulge"})
            action, cat = f"agree -> {b} bulge", "agree"
        elif b < n_lo:                             # too few bulges -> fill
            for x in inside:
                recovered.append({**x, "type": "bulge", "source": "bulge"})
            n_fill = n_map - b
            taken = [x["arc_basal"] for x in inside]
            slots = lo_a + POPULATION_MEAN_IL * np.arange(1, int(gap / POPULATION_MEAN_IL) + 1)
            placed = 0
            for s in slots:
                if placed >= n_fill:
                    break
                if taken and np.min(np.abs(np.array(taken) - s)) < MATCH_TOL:
                    continue
                recovered.append({"arc_basal": float(s), "pos": centerline_at(s),
                                  "prominence": np.nan, "type": "guessed",
                                  "source": "popmean_fill"})
                taken.append(s); placed += 1
            action, cat = f"fill -> {b} bulge + {placed} guessed", "fill"
        else:                                      # b > n_hi -> demote extras
            for j, x in enumerate(inside):
                recovered.append({**x, "type": "bulge" if j < n_hi else "candidate",
                                  "source": "bulge"})
            action, cat = f"too many -> {n_hi} bulge + {b - n_hi} candidate", "too_many"
        gap_rows.append({"cells": [f"{lo_a:.3f}-{hi_a:.3f}", f"{gap:.3f}", "yes",
                                   str(b), f"[{n_lo},{n_map},{n_hi}]",
                                   f"{current_gap_type}; {action}"], "cat": cat})
        print(
            f"  {current_gap_type} gap[{lo_a:.3f},{hi_a:.3f}] {gap:.3f}m "
            f"b={b} n_g=[{n_lo},{n_map},{n_hi}]: {action}"
        )
        record_gap_evaluation(
            i + 1, left_reference, right_reference, current_gap_type,
            lo_a, hi_a, b, recovered_start, c, cat
        )

    # ---- final bud list ----
    for recovered_index, bud in enumerate(recovered, start=1):
        bud["evaluation_id"] = f"cane_{EVALUATION_CANE_ID}_recovered_{recovered_index:03d}"
    buds = detected + recovered
    accepted = [b for b in buds if b["type"] != "candidate"]
    accepted.sort(key=lambda b: b["arc_basal"])
    prev = None
    for b in accepted:
        b["internode"] = np.nan if prev is None else b["arc_basal"] - prev
        prev = b["arc_basal"]
    for b in buds:
        b.setdefault("internode", np.nan)

    # ---- diameter for every bud ----
    for b in buds:
        raw_arc = b2a(b["arc_basal"])
        mpt, mtan = point_tangent_at_arc(sk, cum, raw_arc + apex_dir * BUD_DIAM_OFFSET)
        diam, p_lo, p_hi, n = measure_diameter(mpt, mtan, surface)
        b["diameter"] = diam
        b["line"] = None if p_lo is None else (p_lo, p_hi)

    # ---- evaluator-ready node and per-gap tables ----
    # This preserves candidates for the GPLA stage as well as accepted bulge
    # and guessed nodes for the later recovery stages.
    evaluation_node_rows = []
    for bud in sorted(buds, key=lambda item: item["arc_basal"]):
        evaluation_node_rows.append(
            {
                "cane_id": EVALUATION_CANE_ID,
                "predicted_id": bud["evaluation_id"],
                "arc_m": bud["arc_basal"],
                "diameter_mm": bud["diameter"] * 1000 if np.isfinite(bud["diameter"]) else np.nan,
                "type": bud["type"],
                "source": bud["source"],
                "internode_length_m": bud["internode"],
                "bulge_prominence_mm": (
                    bud["prominence"] * 1000 if np.isfinite(bud["prominence"]) else np.nan
                ),
                "x": bud["pos"][0],
                "y": bud["pos"][1],
                "z": bud["pos"][2],
            }
        )
    evaluation_nodes = pd.DataFrame(evaluation_node_rows)

    evaluation_gap_rows = []
    for record in gap_evaluation_records:
        row = {key: value for key, value in record.items() if key != "_gap_nodes"}
        gap_nodes = record["_gap_nodes"]
        row["gpla_peak_ids"] = "|".join(node["evaluation_id"] for node in gap_nodes)
        row["bulge_node_ids"] = "|".join(
            node["evaluation_id"] for node in gap_nodes if node["type"] == "bulge"
        )
        row["guessed_node_ids"] = "|".join(
            node["evaluation_id"] for node in gap_nodes if node["type"] == "guessed"
        )
        row["candidate_node_ids"] = "|".join(
            node["evaluation_id"] for node in gap_nodes if node["type"] == "candidate"
        )
        row["accepted_recovered_count"] = row["bulge_count"] + row["guessed_count"]
        row["accepted_recovered_ids"] = "|".join(
            node["evaluation_id"]
            for node in gap_nodes
            if node["type"] in {"bulge", "guessed"}
        )
        evaluation_gap_rows.append(row)
    evaluation_gaps = pd.DataFrame(evaluation_gap_rows)

    save_evaluation_table(
        evaluation_nodes,
        OUT_EVAL_NODES,
        OUT_EVAL_NODES_COMBINED,
        ["cane_id", "arc_m", "predicted_id"],
    )
    save_evaluation_table(
        evaluation_gaps,
        OUT_EVAL_GAPS,
        OUT_EVAL_GAPS_COMBINED,
        ["cane_id", "gap_index"],
    )

    # ---- CSV (ordered basal->apex; candidates listed after, no internode) ----
    buds_sorted = sorted(buds, key=lambda b: (b["type"] == "candidate", b["arc_basal"]))
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["order_id", "type", "source", "arc_from_basal_m",
                    "internode_length_m", "diameter_mm", "bulge_prominence_mm",
                    "x", "y", "z"])
        for i, b in enumerate(buds_sorted):
            w.writerow([
                i, b["type"], b["source"], f"{b['arc_basal']:.6f}",
                "" if np.isnan(b["internode"]) else f"{b['internode']:.6f}",
                "" if np.isnan(b["diameter"]) else f"{b['diameter']*1000:.2f}",
                "" if np.isnan(b["prominence"]) else f"{b['prominence']*1000:.2f}",
                f"{b['pos'][0]:.6f}", f"{b['pos'][1]:.6f}", f"{b['pos'][2]:.6f}",
            ])
    counts = {t: sum(b["type"] == t for b in buds) for t in COLORS}
    print(f"\nSaved {OUT_CSV}")
    print(f"  detected={counts['detected']} bulge={counts['bulge']} "
          f"guessed={counts['guessed']} candidate={counts['candidate']}  "
          f"(total recovered buds = {counts['bulge']+counts['guessed']})")

    # ---- figure (Times New Roman) ----
    fig, ax = plt.subplots(figsize=(18, 7))
    ax.scatter(surface[:, 0], surface[:, 1], s=2, c="0.8", zorder=1)
    ax.plot(sk[:, 0], sk[:, 1], "-", c="0.45", lw=1, zorder=2)
    for b in buds:
        if b["line"] is not None:
            p0, p1 = b["line"]
            ax.plot([p0[0], p1[0]], [p0[1], p1[1]], "-", c="royalblue", lw=1.2, zorder=3)
    for t in ["candidate", "guessed", "bulge", "detected"]:   # draw red on top
        pts = np.array([b["pos"] for b in buds if b["type"] == t])
        if len(pts):
            ax.scatter(pts[:, 0], pts[:, 1], s=80, c=[COLORS[t]], edgecolor="k",
                       linewidths=0.4, zorder=5, label=t)
    ax.set_aspect("equal"); ax.legend(loc="best")
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    ax.set_title("Recovered buds: red=detected, orange=bulge, yellow=guessed, grey=candidate")
    fig.tight_layout()
    os.makedirs(os.path.dirname(OUT_IMG), exist_ok=True)
    fig.savefig(OUT_IMG, dpi=120)
    plt.close(fig)
    print(f"Saved {OUT_IMG}")

    # ---- original RGB image with registered recovery overlay ----
    if ORIGINAL_IMAGE_PATH:
        save_original_image_overlay(surface, sk, buds)

    # ---- per-gap reconciliation table (image) ----
    col_labels = ["Gap (m)", "Len (m)", "Flagged", "Bulges (b)",
                  "PNM n_g [lo,MAP,hi]", "Result"]
    cell_text = [r["cells"] for r in gap_rows]
    cat_color = {"agree": "#d6f5d6", "fill": "#fff2cc",
                 "too_many": "#e6e6e6", "not_flagged": "#f6d6d6"}
    figt, axt = plt.subplots(figsize=(13, 1.1 + 0.45 * max(1, len(cell_text))))
    axt.axis("off")
    tbl = axt.table(cellText=cell_text or [["-"] * 6], colLabels=col_labels,
                    loc="center", cellLoc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(11); tbl.scale(1, 1.6)
    for j in range(len(col_labels)):
        hc = tbl[0, j]; hc.set_facecolor("#4472c4")
        hc.set_text_props(color="white", weight="bold")
    for i, r in enumerate(gap_rows):
        tbl[i + 1, 5].set_facecolor(cat_color.get(r["cat"], "white"))
    n_rec = counts["bulge"] + counts["guessed"]
    axt.set_title(f"Per-gap reconciliation  (detected={counts['detected']}, "
                  f"recovered={n_rec}, candidate={counts['candidate']})",
                  fontweight="bold", pad=12)
    figt.tight_layout()
    figt.savefig(OUT_TABLE, dpi=150, bbox_inches="tight")
    print(f"Saved {OUT_TABLE}")

    # ---- colour-coded .ply ----
    surf_pcd = o3d.io.read_point_cloud(SMOOTH_PLY_PATH)
    surf_col = (np.asarray(surf_pcd.colors) if surf_pcd.has_colors()
                else np.tile([0.72, 0.72, 0.72], (len(surface), 1)))
    pts_all, col_all = [surface, sk], [surf_col, np.tile([0.15]*3, (len(sk), 1))]
    for b in buds:
        m = o3d.geometry.TriangleMesh.create_sphere(radius=MARK_RADIUS)
        m.translate(b["pos"])
        p = np.asarray(m.sample_points_uniformly(number_of_points=250).points)
        pts_all.append(p); col_all.append(np.tile(COLORS[b["type"]], (len(p), 1)))
    viz = o3d.geometry.PointCloud()
    viz.points = o3d.utility.Vector3dVector(np.vstack(pts_all))
    viz.colors = o3d.utility.Vector3dVector(np.vstack(col_all))
    os.makedirs(os.path.dirname(OUT_VIZ), exist_ok=True)
    o3d.io.write_point_cloud(OUT_VIZ, viz)
    print(f"Saved {OUT_VIZ}  ({len(viz.points)} points)")


if __name__ == "__main__":
    main()
