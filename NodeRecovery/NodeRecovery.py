"""Run field-data training, RGB-D preparation, node recovery, and evaluation.

The cane/FPV masks and visible-bud locations are explicit handoff inputs. They
can be supplied manually; visible-bud centres can also come from YOLO weights.
Nothing is downloaded, and no private field data or trained weights are bundled.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
METHODS = ROOT / "methods"
PROBABILITIES = {0.3, 0.4, 0.5, 0.6}


class ConfigurationError(ValueError):
    """The supplied configuration cannot describe a valid run."""


def source_path(value: str | None, config_dir: Path, description: str) -> Path:
    if not value:
        raise ConfigurationError(f"Missing {description} in the configuration.")
    path = Path(value).expanduser()
    return (path if path.is_absolute() else config_dir / path).resolve()


def existing_file(value: str | None, config_dir: Path, description: str) -> Path:
    path = source_path(value, config_dir, description)
    if not path.is_file():
        raise ConfigurationError(f"{description} does not exist: {path}")
    return path


def existing_dir(value: str | None, config_dir: Path, description: str) -> Path:
    path = source_path(value, config_dir, description)
    if not path.is_dir():
        raise ConfigurationError(f"{description} does not exist: {path}")
    return path


def load_config(path: Path) -> dict[str, Any]:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise ConfigurationError(f"Configuration file does not exist: {path}")
    with path.open(encoding="utf-8") as handle:
        config = json.load(handle)
    if not isinstance(config, dict):
        raise ConfigurationError("The configuration must be a JSON object.")
    config["_config_dir"] = path.parent
    output = Path(config.get("output_dir", "runs/default")).expanduser()
    config["_output_dir"] = (output if output.is_absolute() else path.parent / output).resolve()
    cases = config.get("cases", [])
    if not isinstance(cases, list):
        raise ConfigurationError("'cases' must be a list.")
    cane_ids = []
    labels = []
    for case in cases:
        if not isinstance(case, dict) or "cane_id" not in case:
            raise ConfigurationError("Every case must contain an integer cane_id.")
        try:
            cane_id = int(case["cane_id"])
        except (TypeError, ValueError) as error:
            raise ConfigurationError("Every cane_id must be an integer.") from error
        if cane_id < 1:
            raise ConfigurationError("Every cane_id must be positive.")
        case["cane_id"] = cane_id
        cane_ids.append(cane_id)
        labels.append(case_label(case))
    if len(cane_ids) != len(set(cane_ids)) or len(labels) != len(set(labels)):
        raise ConfigurationError("Case labels and cane_id values must be unique within a run.")
    return config


def case_label(case: dict[str, Any]) -> str:
    label = str(case.get("name", f"cane_{case['cane_id']}"))
    if not re.fullmatch(r"[A-Za-z0-9_-]+", label):
        raise ConfigurationError("A case name may contain only letters, digits, '-' and '_'.")
    return label


def selected_cases(config: dict[str, Any], requested: str | None) -> list[dict[str, Any]]:
    cases = config.get("cases", [])
    if not cases:
        raise ConfigurationError("At least one case is required for reconstruction or recovery.")
    if requested is None:
        return cases
    selected = [c for c in cases if requested in {case_label(c), str(c["cane_id"])}]
    if len(selected) != 1:
        raise ConfigurationError(f"Case {requested!r} was not found uniquely.")
    return selected


def camera_for(config: dict[str, Any], case: dict[str, Any]) -> dict[str, float]:
    camera = case.get("camera", config.get("camera"))
    if not isinstance(camera, dict):
        raise ConfigurationError(f"Case {case_label(case)} requires camera intrinsics fx, fy, cx, cy.")
    try:
        values = {key: float(camera[key]) for key in ("fx", "fy", "cx", "cy")}
    except (KeyError, TypeError, ValueError) as error:
        raise ConfigurationError(f"Case {case_label(case)} has incomplete camera intrinsics.") from error
    if values["fx"] <= 0 or values["fy"] <= 0:
        raise ConfigurationError("Camera focal lengths must be positive.")
    return values


def case_is_prepared(case: dict[str, Any]) -> bool:
    return any(key in case for key in ("skeleton_ply", "surface_ply", "fpv_ply", "buds_dir"))


def validate_case_sources(config: dict[str, Any], case: dict[str, Any]) -> None:
    base = config["_config_dir"]
    label = case_label(case)
    if case_is_prepared(case):
        for key in ("skeleton_ply", "surface_ply", "fpv_ply"):
            existing_file(case.get(key), base, f"{label}.{key}")
        existing_dir(case.get("buds_dir"), base, f"{label}.buds_dir")
        if case.get("rgb"):
            existing_file(case["rgb"], base, f"{label}.rgb")
            camera_for(config, case)
        return
    for key in ("rgb", "depth", "fc_mask", "fpv_mask"):
        existing_file(case.get(key), base, f"{label}.{key}")
    if case.get("bud_pixels_csv"):
        existing_file(case["bud_pixels_csv"], base, f"{label}.bud_pixels_csv")
    elif case.get("yolo_weights"):
        existing_file(case["yolo_weights"], base, f"{label}.yolo_weights")
    else:
        raise ConfigurationError(
            f"{label} needs bud_pixels_csv or yolo_weights; the runner cannot infer visible buds without either."
        )
    camera_for(config, case)


def model_path(config: dict[str, Any], *, require_exists: bool) -> Path:
    if config.get("model_path"):
        path = source_path(config["model_path"], config["_config_dir"], "model_path")
    else:
        probability = float(config.get("mask_probability", -1))
        if probability not in PROBABILITIES:
            raise ConfigurationError("Choose mask_probability from 0.3, 0.4, 0.5, 0.6.")
        path = config["_output_dir"] / "training" / "weights" / "grouped_cv" / f"selected_model_mask_{probability:.1f}.pkl"
    if require_exists and not path.is_file():
        raise ConfigurationError(f"Trained model does not exist: {path}. Run 'train' or provide model_path.")
    return path


def run_script(name: str, env: dict[str, str]) -> None:
    script = METHODS / name
    print(f"\n==> {script.name}", flush=True)
    subprocess.run([sys.executable, str(script)], check=True, cwd=ROOT, env=env)


def run_training(config: dict[str, Any], *, feature_importance: bool = False) -> Path:
    field_data = existing_file(config.get("field_data"), config["_config_dir"], "field_data")
    probability = float(config.get("mask_probability", -1))
    if probability not in PROBABILITIES:
        raise ConfigurationError("Training requires mask_probability = 0.3, 0.4, 0.5, or 0.6.")
    work_dir = config["_output_dir"] / "training"
    work_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update({
        "NR_FIELD_DATA": str(field_data),
        "NR_WORK_DIR": str(work_dir),
        "NR_FIELD_SHEET": str(config.get("field_sheet", 0)),
        "NR_TRAIN_RATIO": str(config.get("train_ratio", 0.8)),
        "NR_RANDOM_SEED": str(config.get("random_seed", 42)),
    })
    for name in ("split_field_data.py", "get_ML_trainData.py", "get_ML_training5fold.py"):
        run_script(name, env)
    if feature_importance:
        run_script("evaluate_feature_importance.py", env)
    path = work_dir / "weights" / "grouped_cv" / f"selected_model_mask_{probability:.1f}.pkl"
    if not path.is_file():
        raise RuntimeError(f"Training completed without the expected model: {path}")
    print(f"\nSelected configured masking probability: p={probability:.1f}; model: {path}")
    return path


def bud_pixels(config: dict[str, Any], case: dict[str, Any], rgb: Any, fc_mask: Any) -> list[tuple[int, int]]:
    base = config["_config_dir"]
    if case.get("bud_pixels_csv"):
        path = existing_file(case["bud_pixels_csv"], base, "bud_pixels_csv")
        with path.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames or not {"u", "v"}.issubset(reader.fieldnames):
                raise ConfigurationError(f"{path} must have columns u,v (pixel coordinates).")
            points = [(round(float(row["u"])), round(float(row["v"]))) for row in reader]
    else:
        # The model weights are supplied by the user; no YOLO weights are bundled.
        os.environ.setdefault("YOLO_CONFIG_DIR", str(config["_output_dir"] / "ultralytics"))
        from ultralytics import YOLO

        weights = existing_file(case["yolo_weights"], base, "yolo_weights")
        result = YOLO(str(weights))(rgb, conf=float(case.get("yolo_confidence", 0.4)), verbose=False)[0]
        bud_class = int(case.get("bud_class_id", 0))
        points = []
        if result.boxes is not None:
            for box, cls in zip(result.boxes.xyxy.cpu().numpy(), result.boxes.cls.cpu().numpy()):
                if int(cls) == bud_class:
                    points.append((round(float((box[0] + box[2]) / 2)), round(float((box[1] + box[3]) / 2))))
    import cv2

    x, y, width, height = cv2.boundingRect(fc_mask)
    if width == 0 or height == 0:
        raise ConfigurationError("The FC mask contains no foreground pixels.")
    associated = [(u, v) for u, v in points if x <= u < x + width and y <= v < y + height]
    if not associated:
        raise ConfigurationError(f"No visible-bud pixel centres fall within {case_label(case)}'s FC bounding box.")
    return associated


def reconstruct_case(config: dict[str, Any], case: dict[str, Any]) -> dict[str, Path]:
    base = config["_config_dir"]
    if case_is_prepared(case):
        return {
            "skeleton": existing_file(case["skeleton_ply"], base, "skeleton_ply"),
            "surface": existing_file(case["surface_ply"], base, "surface_ply"),
            "fpv": existing_file(case["fpv_ply"], base, "fpv_ply"),
            "buds": existing_dir(case["buds_dir"], base, "buds_dir"),
            "rgb": existing_file(case["rgb"], base, "rgb") if case.get("rgb") else None,
        }
    validate_case_sources(config, case)
    import cv2
    import numpy as np

    sys.path.insert(0, str(METHODS))
    from denoise_depth import denoise
    from mask2pcd_projection import project_mask_to_pcd
    from maskMend import mend_binary_mask
    from pcd_preprocessing import process_point_cloud
    import pixel2pcd_projection as pixel

    label = case_label(case)
    case_dir = config["_output_dir"] / "cases" / label
    input_dir = case_dir / "in"
    buds_dir = case_dir / "buds"
    input_dir.mkdir(parents=True, exist_ok=True)
    buds_dir.mkdir(parents=True, exist_ok=True)
    rgb_path = existing_file(case["rgb"], base, f"{label}.rgb")
    rgb = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
    if rgb is None:
        raise ConfigurationError(f"Cannot decode RGB image: {rgb_path}")
    depth_path = existing_file(case["depth"], base, f"{label}.depth")
    if depth_path.suffix.lower() == ".npy":
        depth_raw = np.load(depth_path)
    else:
        depth_raw = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
    if depth_raw is None or depth_raw.ndim != 2:
        raise ConfigurationError(f"Depth must be a single-channel image or 2D .npy array: {depth_path}")
    if depth_raw.shape != rgb.shape[:2]:
        raise ConfigurationError("RGB and depth dimensions must agree; provide registered images.")
    depth = depth_raw.astype(np.float32)
    if depth_raw.dtype == np.uint16:
        depth *= float(config.get("depth_scale_m", 0.001))
    elif not np.issubdtype(depth_raw.dtype, np.floating):
        raise ConfigurationError("Depth must be uint16 with depth_scale_m or floating-point metres.")
    def read_mask(key: str) -> Any:
        path = existing_file(case[key], base, f"{label}.{key}")
        mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if mask is None or mask.shape != rgb.shape[:2]:
            raise ConfigurationError(f"{key} must be a registered single-channel mask: {path}")
        return np.where(mask > 0, 255, 0).astype(np.uint8)

    fc_mask = read_mask("fc_mask")
    fpv_mask = read_mask("fpv_mask")
    if case.get("mend_fc_mask", True):
        fc_mask = mend_binary_mask(fc_mask, output_path=str(input_dir / "fc_mask_mended.png"))
    cv2.imwrite(str(input_dir / "fc_mask_used.png"), fc_mask)
    camera = camera_for(config, case)
    z_range = tuple(float(x) for x in config.get("z_range_m", [0.45, 1.5]))
    if len(z_range) != 2 or z_range[0] >= z_range[1]:
        raise ConfigurationError("z_range_m must contain an increasing [minimum, maximum].")
    raw = input_dir / "cane_raw.ply"
    raw_points, _ = project_mask_to_pcd(rgb, depth, fc_mask, str(raw), camera, z_range)
    if raw_points is None:
        raise ConfigurationError(f"FC projection produced no point-cloud points for {label}.")
    fpv = input_dir / "fpv_raw.ply"
    fpv_points, _ = project_mask_to_pcd(rgb, depth, fpv_mask, str(fpv), camera, z_range)
    if fpv_points is None:
        raise ConfigurationError(f"FPV projection produced no point-cloud points for {label}.")
    voxel = input_dir / "cane_voxel.ply"
    process_point_cloud(str(raw), str(voxel), voxel_size=float(config.get("voxel_size_m", 0.003)))
    clean = voxel
    if case.get("radial_clean", False):
        from clean_branch_radial import clean as radial_clean

        clean = input_dir / "cane_clean.ply"
        radial_clean(str(voxel), out=str(clean))
    surface = input_dir / "cane_smooth.ply"
    denoise(
        str(clean), out=str(surface),
        sig_s=float(config.get("denoise_sig_s_m", 0.018)),
        sig_v=float(config.get("denoise_sig_v_m", 0.008)),
        sig_r=float(config.get("denoise_sig_r_m", 0.012)),
        iters=int(config.get("denoise_iterations", 3)),
        h=float(config.get("skeleton_support_m", 0.12)),
        n_samples=int(config.get("skeleton_samples", 180)),
        skel_smooth=int(config.get("skeleton_smooth_window", 5)),
    )
    skeleton = input_dir / "cane_skeleton.ply"
    if not skeleton.is_file():
        raise RuntimeError(f"Expected skeleton was not created: {skeleton}")
    pixel.Z_MIN, pixel.Z_MAX = z_range
    centres = bud_pixels(config, case, rgb, fc_mask)
    n_saved = 0
    for u, v in centres:
        stable = pixel.find_stable_depth(u, v, depth)
        if stable is None:
            print(f"Warning: skipped visible bud at pixel ({u},{v}); no stable depth.")
            continue
        uu, vv, z = stable
        xyz = pixel.backproject(uu, vv, z, **camera)
        pixel.write_ply_xyzrgb(str(buds_dir / f"bud_{n_saved:03d}.ply"), [xyz], (255, 0, 0))
        n_saved += 1
    if n_saved == 0:
        raise ConfigurationError(f"No detected buds could be back-projected for {label}.")
    print(f"Prepared {label}: {len(raw_points)} raw points, {n_saved} detected buds.")
    return {"skeleton": skeleton, "surface": surface, "fpv": fpv, "buds": buds_dir, "rgb": rgb_path}


def recover_cases(config: dict[str, Any], cases: list[dict[str, Any]], model: Path) -> None:
    for case in cases:
        label = case_label(case)
        prepared = reconstruct_case(config, case)
        result_dir = config["_output_dir"] / "cases" / label / "results"
        result_dir.mkdir(parents=True, exist_ok=True)
        env = os.environ.copy()
        env.update({
            "NR_CASE_DIR": str(result_dir.parent),
            "NR_CASE_LABEL": label,
            "NR_CANE_ID": str(case["cane_id"]),
            "NR_SKELETON_PLY": str(prepared["skeleton"]),
            "NR_SURFACE_PLY": str(prepared["surface"]),
            "NR_FPV_PLY": str(prepared["fpv"]),
            "NR_BUDS_DIR": str(prepared["buds"]),
            "NR_MODEL_PATH": str(model),
            "NR_IMAGE_PATH": str(prepared["rgb"] or ""),
            "NR_RECOVERY_OUT_DIR": str(result_dir),
            "NR_COMBINED_NODES_CSV": str(config["_output_dir"] / "recon_recovered.csv"),
            "NR_COMBINED_GAPS_CSV": str(config["_output_dir"] / "pnm_gap_results.csv"),
            "NR_PNM_MEAN_IL_M": str(config.get("pnm_mean_il_m", 0.08406)),
            "NR_PNM_STD_IL_M": str(config.get("pnm_std_il_m", 0.02730)),
            "NR_PNM_KMAX": str(config.get("pnm_kmax", 25)),
            "NR_PNM_CREDIBLE_LEVEL": str(config.get("pnm_credible_level", 0.9)),
            "NR_PROMINENCE_M": str(config.get("prominence_m", 0.0022)),
            "NR_MIN_SEP_M": str(config.get("minimum_peak_separation_m", 0.04)),
            "NR_DEDUP_TOLERANCE_M": str(config.get("dedup_tolerance_m", 0.025)),
        })
        if prepared["rgb"]:
            env["NR_CAMERA_INTRINSICS_JSON"] = json.dumps(camera_for(config, case))
        run_script("recover_buds.py", env)
    # Rebuild pooled inputs from exactly the selected cases, avoiding stale rows.
    import pandas as pd

    for suffix, target in (
        ("prediction_evaluation.csv", "recon_recovered.csv"),
        ("pnm_gap_results.csv", "pnm_gap_results.csv"),
    ):
        paths = [config["_output_dir"] / "cases" / case_label(case) / "results" / f"{case_label(case)}_{suffix}" for case in cases]
        pooled = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
        pooled.to_csv(config["_output_dir"] / target, index=False)
    print(f"\nCombined recovery tables saved to {config['_output_dir']}")


def evaluate(config: dict[str, Any]) -> None:
    truth = existing_file(config.get("ground_truth_path"), config["_config_dir"], "ground_truth_path")
    output = config["_output_dir"]
    nodes = existing_file(str(output / "recon_recovered.csv"), config["_config_dir"], "combined recovery table")
    gaps = existing_file(str(output / "pnm_gap_results.csv"), config["_config_dir"], "combined PNM gap table")
    env = os.environ.copy()
    env.update({
        "NR_RUN_DIR": str(output),
        "NR_GROUND_TRUTH_PATH": str(truth),
        "NR_PREDICTION_CSV_PATH": str(nodes),
        "NR_PNM_GAP_CSV_PATH": str(gaps),
        "NR_EVAL_OUTPUT_DIR": str(output / "evaluation"),
        "NR_RECOVERY_TOLERANCE_MM": str(config.get("recovery_tolerance_mm", 30.0)),
    })
    run_script("evaluate_recovery_pipeline.py", env)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("check", "train", "reconstruct", "recover", "evaluate", "run"))
    parser.add_argument("--config", required=True, type=Path, help="JSON run configuration")
    parser.add_argument("--case", help="Case name or cane_id for reconstruct/recover")
    parser.add_argument("--feature-importance", action="store_true", help="Also run ablation and permutation analysis during training")
    args = parser.parse_args()
    try:
        config = load_config(args.config)
        if args.case and args.command not in {"reconstruct", "recover"}:
            raise ConfigurationError("--case is supported only with reconstruct or recover.")
        if args.command == "check":
            for case in config.get("cases", []):
                validate_case_sources(config, case)
            if config.get("field_data"):
                existing_file(config["field_data"], config["_config_dir"], "field_data")
            if config.get("model_path"):
                model_path(config, require_exists=True)
            print(f"Inputs OK: {len(config['cases'])} case(s). Output: {config['_output_dir']}")
        elif args.command == "train":
            run_training(config, feature_importance=args.feature_importance)
        elif args.command == "reconstruct":
            for case in selected_cases(config, args.case):
                paths = reconstruct_case(config, case)
                print(f"{case_label(case)}: {paths}")
        elif args.command == "recover":
            cases = selected_cases(config, args.case)
            recover_cases(config, cases, model_path(config, require_exists=True))
        elif args.command == "evaluate":
            evaluate(config)
        else:
            cases = selected_cases(config, None)
            for case in cases:
                validate_case_sources(config, case)
            if config.get("model_path"):
                model = model_path(config, require_exists=True)
            else:
                model = model_path(config, require_exists=False)
                if not model.is_file():
                    model = run_training(config, feature_importance=args.feature_importance)
            recover_cases(config, cases, model)
            if config.get("ground_truth_path"):
                evaluate(config)
            else:
                print("Ground truth not supplied; recovery completed without evaluation.")
    except (ConfigurationError, ValueError, FileNotFoundError) as error:
        parser.exit(2, f"Error: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
