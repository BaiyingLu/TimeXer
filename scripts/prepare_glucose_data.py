"""
Step 1 — Prepare unified CSVs for OhioT1DM, BrisT1D, and HUPA-UCM.

Target format (glucose LAST for MS mode):
    date, subject, carbs, total_insulin, steps, glucose

subject is kept for gap detection (not used as a model feature).

Splits: 70 / 15 / 15 by time within each subject, then merged.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# ── Column mappings to unified schema ────────────────────────────────────────

UNIFIED_COLS = ["date", "subject", "carbs", "total_insulin", "steps", "glucose"]


def interpolate_short_gaps(df: pd.DataFrame, max_gap_min: int = 15) -> pd.DataFrame:
    """Linearly interpolate glucose for gaps <= max_gap_min per subject.

    For each subject:
    1. Reindex timestamps to a regular 5-min grid
    2. Linearly interpolate glucose where the gap is <= max_gap_min
       (limit=max_gap_min//5 - 1 consecutive NaNs, e.g. limit=2 for 15 min)
    3. Fill exogenous features (carbs, total_insulin, steps) with 0.0 on
       newly created grid rows (no event = no signal)
    4. Drop rows where glucose is still NaN (gaps > max_gap_min)

    Returns the concatenated result with updated row counts printed.
    """
    max_interp = max_gap_min // 5 - 1  # e.g. 15 min → limit=2 consecutive NaNs

    parts = []
    n_before_total = 0
    n_interpolated_total = 0
    n_dropped_total = 0

    for subj, grp in df.groupby("subject"):
        grp = grp.sort_values("date").set_index("date")
        grp = grp[~grp.index.duplicated(keep="first")]

        # Reindex to a regular 5-min grid spanning this subject's range
        full_idx = pd.date_range(grp.index.min(), grp.index.max(), freq="5min")
        grp = grp.reindex(full_idx)
        grp.index.name = "date"
        grp["subject"] = subj

        # Count NaN glucose before interpolation
        nan_before = grp["glucose"].isna().sum()
        n_before_total += len(grp)

        # Linear interpolation with limit
        grp["glucose"] = grp["glucose"].interpolate(method="linear", limit=max_interp)

        # Count how many were filled
        nan_after = grp["glucose"].isna().sum()
        n_interpolated_total += nan_before - nan_after
        n_dropped_total += nan_after

        # Fill exogenous features with 0 on new grid rows
        for col in ["carbs", "total_insulin", "steps"]:
            grp[col] = grp[col].fillna(0.0)

        # Drop rows where glucose is still NaN (large gaps)
        grp = grp.dropna(subset=["glucose"])

        parts.append(grp.reset_index())

    result = pd.concat(parts, ignore_index=True)
    print(f"  Interpolated {n_interpolated_total:,} glucose values (gaps <= {max_gap_min} min)")
    print(f"  Dropped {n_dropped_total:,} rows with glucose gaps > {max_gap_min} min")

    return result


def load_ohio(src_dir: Path) -> pd.DataFrame:
    """Load OhioT1DM from filter_train + filter_test directories."""
    train_dir = src_dir / "filter_train"
    test_dir = src_dir / "filter_test"

    frames = []
    for d in [train_dir, test_dir]:
        for f in sorted(d.glob("*.csv")):
            subject_id = f.stem.split("-")[0]  # e.g. "559"
            df = pd.read_csv(f)
            df["subject"] = subject_id
            frames.append(df)

    raw = pd.concat(frames, ignore_index=True)

    unified = pd.DataFrame()
    unified["date"] = pd.to_datetime(raw["timestamp"])
    unified["carbs"] = raw["carbs"].fillna(0.0)
    unified["total_insulin"] = raw["total_insulin"].fillna(0.0)
    unified["steps"] = raw["steps"].fillna(0.0)
    unified["glucose"] = raw["glucose"]  # already mg/dL
    unified["subject"] = raw["subject"]

    unified = interpolate_short_gaps(unified)

    return unified


def load_brist1d(src_dir: Path) -> pd.DataFrame:
    """Load BrisT1D from filter_train + filter_test directories.

    Uses the curated 15-subject set in this repo (excludes P01, P05, P06,
    P13, P21 which have 15-min CGM intervals).

    Glucose gaps <= 15 min (up to 3 consecutive missing points on a 5-min
    grid) are filled via linear interpolation. Larger gaps are left as NaN
    and those rows are dropped.
    """
    train_dir = src_dir / "filter_train"
    test_dir = src_dir / "filter_test"

    frames = []
    for d in [train_dir, test_dir]:
        for f in sorted(d.glob("P*.csv")):
            subject_id = f.stem  # e.g. "P02"
            df = pd.read_csv(f)
            df = df.replace("", np.nan)
            df["subject"] = subject_id
            frames.append(df)

    raw = pd.concat(frames, ignore_index=True)

    unified = pd.DataFrame()
    unified["date"] = pd.to_datetime(raw["timestamp"])
    unified["carbs"] = pd.to_numeric(raw["carbs"], errors="coerce").fillna(0.0)
    # insulin = total insulin (basal + bolus)
    unified["total_insulin"] = pd.to_numeric(raw["insulin"], errors="coerce").fillna(0.0)
    unified["steps"] = pd.to_numeric(raw["steps"], errors="coerce").fillna(0.0)
    # bg already in mg/dL in this repo's curated filter files
    unified["glucose"] = pd.to_numeric(raw["bg"], errors="coerce")
    unified["subject"] = raw["subject"]

    # Interpolate glucose for gaps <= 15 min per subject
    unified = interpolate_short_gaps(unified)

    return unified


def load_hupa(src_dir: Path) -> pd.DataFrame:
    """Load HUPA-UCM from filter_train + filter_test directories."""
    train_dir = src_dir / "filter_train"
    test_dir = src_dir / "filter_test"

    frames = []
    for d in [train_dir, test_dir]:
        for f in sorted(d.glob("*.csv")):
            subject_id = f.stem  # e.g. "HUPA0001P"
            df = pd.read_csv(f)
            df["subject"] = subject_id
            frames.append(df)

    raw = pd.concat(frames, ignore_index=True)

    unified = pd.DataFrame()
    unified["date"] = pd.to_datetime(raw["timestamp"])
    unified["carbs"] = raw["carbs"].fillna(0.0)
    # insulin = total insulin (basal_rate + bolus_volume_delivered)
    unified["total_insulin"] = raw["insulin"].fillna(0.0)
    unified["steps"] = raw["steps"].fillna(0.0)
    unified["glucose"] = raw["bg"]  # already mg/dL
    unified["subject"] = raw["subject"]

    unified = interpolate_short_gaps(unified)

    return unified


# ── Split logic ──────────────────────────────────────────────────────────────


def split_by_time(df: pd.DataFrame, train_frac: float = 0.70, val_frac: float = 0.15):
    """
    Split 70/15/15 by time within each subject, then merge across subjects.
    """
    train_parts, val_parts, test_parts = [], [], []

    for _, grp in df.groupby("subject"):
        grp = grp.sort_values("date").reset_index(drop=True)
        n = len(grp)
        t1 = int(n * train_frac)
        t2 = int(n * (train_frac + val_frac))
        train_parts.append(grp.iloc[:t1])
        val_parts.append(grp.iloc[t1:t2])
        test_parts.append(grp.iloc[t2:])

    train = pd.concat(train_parts, ignore_index=True)
    val = pd.concat(val_parts, ignore_index=True)
    test = pd.concat(test_parts, ignore_index=True)

    return train, val, test


def save_split(df: pd.DataFrame, path: Path):
    """Save unified CSV (drop subject column)."""
    df[UNIFIED_COLS].to_csv(path, index=False)
    print(f"  {path.name}: {len(df):,} rows")


# ── Main ─────────────────────────────────────────────────────────────────────


DATASETS = {
    "OhioT1DM": {
        "loader": load_ohio,
        "src_dir_key": "ohio_src",
        "prefix": "ohio",
    },
    "BrisT1D": {
        "loader": load_brist1d,
        "src_dir_key": "bris_src",
        "prefix": "bris",
    },
    "HUPA-UCM": {
        "loader": load_hupa,
        "src_dir_key": "hupa_src",
        "prefix": "hupa",
    },
}


def main():
    parser = argparse.ArgumentParser(description="Prepare unified glucose CSVs")
    parser.add_argument(
        "--ohio_src",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / ".."
        / "multimodal_ts_llm"
        / "ohiot1dm",
    )
    parser.add_argument(
        "--bris_src",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "dataset" / "BrisT1D",
    )
    parser.add_argument(
        "--hupa_src",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "dataset" / "HUPA-UCM",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "dataset",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["OhioT1DM", "BrisT1D", "HUPA-UCM"],
        choices=["OhioT1DM", "BrisT1D", "HUPA-UCM"],
    )
    args = parser.parse_args()

    train_frames, val_frames, test_frames = [], [], []

    for name in args.datasets:
        cfg = DATASETS[name]
        src_dir = getattr(args, cfg["src_dir_key"])
        out = args.out_dir / name
        out.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"Processing {name} from {src_dir}")
        print(f"{'='*60}")

        df = cfg["loader"](src_dir)

        # Report NaN glucose rows (these will be excluded by Dataset_BGlucose)
        n_nan = df["glucose"].isna().sum()
        print(f"  Total rows: {len(df):,}  |  NaN glucose: {n_nan:,}")

        train, val, test = split_by_time(df)

        # Prefix subject IDs in ALL splits so individual and combined datasets
        # share the same subject ID namespace (e.g. ohio_559, bris_P01).
        # This ensures the combined scalers.pkl can be reused at per-dataset
        # test time without key mismatches.
        prefix = cfg["prefix"]
        for split_df in [train, val, test]:
            split_df["subject"] = prefix + "_" + split_df["subject"].astype(str)

        save_split(train, out / f"{cfg['prefix']}_train.csv")
        save_split(val, out / f"{cfg['prefix']}_val.csv")
        save_split(test, out / f"{cfg['prefix']}_test.csv")

        # Collect all three splits for the combined dataset (subjects already prefixed).
        for split_df, bucket in [(train, train_frames), (val, val_frames), (test, test_frames)]:
            bucket.append(split_df[UNIFIED_COLS].copy())

    # ── Combined dataset (train + val only; test stays per-dataset) ───────────
    if len(args.datasets) > 1:
        combined_out = args.out_dir / "combined"
        combined_out.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*60}")
        print("Creating combined dataset")
        print(f"{'='*60}")

        combined_train = pd.concat(train_frames, ignore_index=True)
        combined_val   = pd.concat(val_frames,   ignore_index=True)
        combined_test  = pd.concat(test_frames,  ignore_index=True)

        combined_train.to_csv(combined_out / "combined_train.csv", index=False)
        combined_val.to_csv(  combined_out / "combined_val.csv",   index=False)
        combined_test.to_csv( combined_out / "combined_test.csv",  index=False)

        print(f"  combined_train.csv: {len(combined_train):,} rows  "
              f"({combined_train['subject'].nunique()} subjects)")
        print(f"  combined_val.csv:   {len(combined_val):,} rows  "
              f"({combined_val['subject'].nunique()} subjects)")
        print(f"  combined_test.csv:  {len(combined_test):,} rows  "
              f"({combined_test['subject'].nunique()} subjects)")

    print("\nDone.")


if __name__ == "__main__":
    main()
