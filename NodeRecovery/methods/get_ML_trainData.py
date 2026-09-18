import os
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------- USER INPUT ----------------
DATA_DIR = Path(os.environ.get("NR_WORK_DIR", Path(__file__).resolve().parent / "work"))
PARTITION_FILES = {
    "train": DATA_DIR / "train_field_data.csv",
    "test": DATA_DIR / "test_field_data.csv",
}
MISSING_PROBABILITIES = (0.3, 0.4, 0.5, 0.6)
RANDOM_SEED = int(os.environ.get("NR_RANDOM_SEED", "42"))
OUT_DIR = DATA_DIR / "res_out" / "ml_masks"
# --------------------------------------------

REQUIRED_COLUMNS = {
    "Branch_id",
    "Bud_arc_length_cm",
    "Bud_IL_cm",
    "Total_length_cm",
}


def validate_partition(data: pd.DataFrame, partition_name: str) -> None:
    """Validate fields needed to create physically normalised field-data gaps."""
    missing_columns = REQUIRED_COLUMNS.difference(data.columns)
    if missing_columns:
        raise ValueError(
            f"{partition_name}: missing required column(s): "
            + ", ".join(sorted(missing_columns))
        )

    if data["Branch_id"].isna().any():
        raise ValueError(f"{partition_name}: Branch_id contains missing values.")

    for branch_id, branch in data.groupby("Branch_id", sort=False):
        total_lengths = branch["Total_length_cm"].dropna()
        if len(total_lengths) != 1:
            raise ValueError(
                f"{partition_name}, Branch_id {branch_id}: expected exactly one "
                "non-missing Total_length_cm value."
            )

        total_length_cm = float(total_lengths.iloc[0])
        if total_length_cm <= 0:
            raise ValueError(
                f"{partition_name}, Branch_id {branch_id}: Total_length_cm must be positive."
            )

        node_arcs_cm = pd.to_numeric(branch["Bud_arc_length_cm"], errors="coerce")
        if node_arcs_cm.isna().any():
            raise ValueError(
                f"{partition_name}, Branch_id {branch_id}: Bud_arc_length_cm "
                "contains missing or non-numeric values."
            )

        sorted_arcs_cm = np.sort(node_arcs_cm.to_numpy(dtype=float))
        if np.any(np.diff(sorted_arcs_cm) <= 0):
            raise ValueError(
                f"{partition_name}, Branch_id {branch_id}: Bud_arc_length_cm "
                "must be strictly increasing with no duplicate node positions."
            )
        if sorted_arcs_cm[0] < 0 or sorted_arcs_cm[-1] > total_length_cm:
            raise ValueError(
                f"{partition_name}, Branch_id {branch_id}: measured buds must lie "
                "within the measured cane interval [0, Total_length_cm]."
            )

        max_node_arc_cm = sorted_arcs_cm[-1]
        if max_node_arc_cm > total_length_cm + 1e-9:
            raise ValueError(
                f"{partition_name}, Branch_id {branch_id}: maximum node arc "
                f"({max_node_arc_cm:.3f} cm) exceeds Total_length_cm "
                f"({total_length_cm:.3f} cm). Correct the source length before "
                "using physical relative position."
            )


def generate_gap_samples(
    data: pd.DataFrame,
    partition_name: str,
    missing_probability: float,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Mask a partition using fixed measured cane-end references.

    The virtual references at 0 and L_FC are not buds. They create basal and
    apical gaps even when the first or last measured bud is masked. The field
    measurements use the physical cane length, not a reconstructed skeleton.
    """
    samples = []

    for branch_id, branch in data.groupby("Branch_id", sort=False):
        branch = branch.sort_values("Bud_arc_length_cm").copy()
        branch["Bud_arc_length_m"] = branch["Bud_arc_length_cm"] / 100.0
        branch["Bud_IL_m"] = branch["Bud_IL_cm"] / 100.0

        total_length_m = branch["Total_length_cm"].dropna().iloc[0] / 100.0

        node_arcs_m = branch["Bud_arc_length_m"].to_numpy(dtype=float)

        # The two physical cane ends are always known. Every measured
        # bud strictly within this interval, including the first and last
        # measured buds, is independently maskable.  A rare bud recorded
        # exactly at 0 or L_FC is co-located with a fixed reference and hence
        # is not a separately maskable position.
        is_interior = (node_arcs_m > 0.0) & (node_arcs_m < total_length_m)
        is_observed = is_interior & (rng.random(len(branch)) > missing_probability)
        observed_interior_arcs_m = node_arcs_m[is_observed]
        reference_arcs_m = np.concatenate(
            ([0.0], np.sort(observed_interior_arcs_m), [total_length_m])
        )

        # These statistics use only the reference sequence available after
        # masking: virtual measured cane ends plus retained buds. Thus their
        # calculation parallels the information available at reconstruction.
        observed_gaps = np.diff(reference_arcs_m)
        mean_il_m = float(np.mean(observed_gaps))
        std_il_m = float(np.std(observed_gaps, ddof=1)) if len(observed_gaps) > 1 else np.nan

        for start_arc_m, end_arc_m in zip(reference_arcs_m[:-1], reference_arcs_m[1:]):
            gap_length_m = end_arc_m - start_arc_m

            hidden_count = int(
                np.sum((node_arcs_m > start_arc_m) & (node_arcs_m < end_arc_m))
            )
            is_basal_boundary = bool(np.isclose(start_arc_m, 0.0))
            is_apical_boundary = bool(np.isclose(end_arc_m, total_length_m))
            if is_basal_boundary and is_apical_boundary:
                gap_type = "basal_apical"
            elif is_basal_boundary:
                gap_type = "basal"
            elif is_apical_boundary:
                gap_type = "apical"
            else:
                gap_type = "internal"

            # rho_i = s_i / L_FC, where s_i is the lower reference coordinate
            # and L_FC is the measured total cane length.
            relative_position = start_arc_m / total_length_m

            samples.append(
                {
                    "Branch_id": branch_id,
                    "partition": partition_name,
                    "mask_probability": missing_probability,
                    "gap_length_m": gap_length_m,
                    "mean_IL_m": mean_il_m,
                    "std_IL_m": std_il_m,
                    "rel_position": relative_position,
                    "total_length_m": total_length_m,
                    "start_arc_m": start_arc_m,
                    "end_arc_m": end_arc_m,
                    "hidden_node_count": hidden_count,
                    "gap_type": gap_type,
                    "is_basal_boundary": is_basal_boundary,
                    "is_apical_boundary": is_apical_boundary,
                    "label_missing": int(hidden_count > 0),
                }
            )

    result = pd.DataFrame(samples)
    if not result["rel_position"].between(0, 1).all():
        raise RuntimeError(
            f"{partition_name}, p={missing_probability}: relative positions must lie in [0, 1]."
        )
    if not result["label_missing"].eq(result["hidden_node_count"].gt(0).astype(int)).all():
        raise RuntimeError(
            f"{partition_name}, p={missing_probability}: gap labels do not match hidden-node counts."
        )
    return result


partitions = {
    name: pd.read_csv(path)
    for name, path in PARTITION_FILES.items()
}
for name, data in partitions.items():
    validate_partition(data, name)

train_branches = set(partitions["train"]["Branch_id"])
test_branches = set(partitions["test"]["Branch_id"])
overlap = train_branches.intersection(test_branches)
if overlap:
    raise ValueError(
        "Train and test partitions share Branch_id value(s): "
        + ", ".join(map(str, sorted(overlap)))
    )

OUT_DIR.mkdir(parents=True, exist_ok=True)
composition_rows = []

for partition_index, (partition_name, data) in enumerate(partitions.items()):
    for probability_index, missing_probability in enumerate(MISSING_PROBABILITIES):
        # A separate deterministic seed makes masking independent across each
        # partition-probability combination, rather than nesting masks by p.
        seed_sequence = np.random.SeedSequence(
            [RANDOM_SEED, partition_index, probability_index]
        )
        rng = np.random.default_rng(seed_sequence)

        samples = generate_gap_samples(
            data=data,
            partition_name=partition_name,
            missing_probability=missing_probability,
            rng=rng,
        )

        output_path = OUT_DIR / f"{partition_name}_mask_{missing_probability:.1f}.csv"
        samples.to_csv(output_path, index=False)
        composition_rows.extend(
            samples.groupby(["partition", "mask_probability", "gap_type", "label_missing"], as_index=False)
            .agg(samples=("Branch_id", "size"), branches=("Branch_id", "nunique"))
            .to_dict("records")
        )
        print(
            f"{partition_name:5s} | p={missing_probability:.1f} | "
            f"branches={samples['Branch_id'].nunique():3d} | "
            f"samples={len(samples):4d} | "
            f"positive={samples['label_missing'].sum():4d} | {output_path.name}"
        )

pd.DataFrame(composition_rows).to_csv(
    OUT_DIR / "mask_gap_composition.csv", index=False
)
