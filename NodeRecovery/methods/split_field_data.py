import os
from pathlib import Path

import numpy as np
import pandas as pd

# The runner supplies the paths. Defaults keep this module independently usable.
WORK_DIR = Path(os.environ.get("NR_WORK_DIR", Path(__file__).resolve().parent / "work"))
INPUT_PATH = Path(os.environ.get("NR_FIELD_DATA", WORK_DIR / "all_raw_data_ln.xlsx"))
SHEET_NAME = os.environ.get("NR_FIELD_SHEET", "0")
SHEET_NAME = int(SHEET_NAME) if SHEET_NAME.isdigit() else SHEET_NAME
TRAIN_RATIO = float(os.environ.get("NR_TRAIN_RATIO", "0.80"))
RANDOM_SEED = int(os.environ.get("NR_RANDOM_SEED", "42"))
TRAIN_OUT = WORK_DIR / "train_field_data.csv"
TEST_OUT = WORK_DIR / "test_field_data.csv"

REQUIRED_COLUMNS = {"Branch_id", "Bud_arc_length_cm", "Bud_IL_cm", "Total_length_cm"}


def validate_input(data: pd.DataFrame) -> None:
    """Check that each branch has a usable, single recorded total length."""
    missing_columns = REQUIRED_COLUMNS.difference(data.columns)
    if missing_columns:
        raise ValueError(
            "The input file is missing required column(s): "
            + ", ".join(sorted(missing_columns))
        )

    if data["Branch_id"].isna().any():
        raise ValueError("Branch_id contains missing values.")

    lengths_per_branch = data.groupby("Branch_id")["Total_length_cm"].count()
    invalid_branches = lengths_per_branch[lengths_per_branch != 1]
    if not invalid_branches.empty:
        raise ValueError(
            "Each branch must have exactly one non-missing Total_length_cm. "
            f"Problematic Branch_id values: {invalid_branches.index.tolist()}"
        )

    recorded_lengths = data.dropna(subset=["Total_length_cm"])["Total_length_cm"]
    if (recorded_lengths <= 0).any():
        raise ValueError("Total_length_cm must be positive for every branch.")

    invalid_ranges = []
    for branch_id, branch in data.groupby("Branch_id", sort=False):
        arcs = pd.to_numeric(branch["Bud_arc_length_cm"], errors="coerce")
        if arcs.isna().any() or (arcs < 0).any() or arcs.duplicated().any():
            raise ValueError(
                f"Branch_id {branch_id}: Bud_arc_length_cm must contain distinct, "
                "nonnegative numeric node positions."
            )
        length = float(branch["Total_length_cm"].dropna().iloc[0])
        excess = float(arcs.max() - length)
        if excess > 1e-9:
            invalid_ranges.append((branch_id, round(excess, 2)))
    if invalid_ranges:
        raise ValueError(
            "Measured bud arcs exceed Total_length_cm; correct the source measurements "
            "before treating 0 and total length as physical boundaries. "
            "Branch_id and excess (cm): " + str(invalid_ranges)
        )


df = pd.read_excel(INPUT_PATH, sheet_name=SHEET_NAME)
validate_input(df)

# Split by Branch_id, never by individual rows. Therefore all node records
# from one branch, including its total length, remain in exactly one dataset.
branch_ids = df["Branch_id"].drop_duplicates().to_numpy(copy=True)
if len(branch_ids) < 2:
    raise ValueError("At least two branches are required to create train and test sets.")

rng = np.random.default_rng(RANDOM_SEED)
rng.shuffle(branch_ids)

n_train = round(len(branch_ids) * TRAIN_RATIO)
n_train = min(max(n_train, 1), len(branch_ids) - 1)
train_branches = branch_ids[:n_train]
test_branches = branch_ids[n_train:]

train_df = df[df["Branch_id"].isin(train_branches)].reset_index(drop=True)
test_df = df[df["Branch_id"].isin(test_branches)].reset_index(drop=True)

WORK_DIR.mkdir(parents=True, exist_ok=True)
train_df.to_csv(TRAIN_OUT, index=False)
test_df.to_csv(TEST_OUT, index=False)

print(f"Input:  {INPUT_PATH}")
print(f"Total:  {len(branch_ids)} branches, {len(df)} rows")
print(f"Train:  {len(train_branches)} branches, {len(train_df)} rows -> {TRAIN_OUT}")
print(f"Test:   {len(test_branches)} branches, {len(test_df)} rows -> {TEST_OUT}")
print(f"Shared branches: {set(train_branches).intersection(test_branches)}")
