#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    default_input = repo_root / "capture" / "handheld_nifei_init.csv"

    p = argparse.ArgumentParser(description="Flip accel_z and gyro_z signs in capture CSV (2-level header).")
    p.add_argument("--input", default=str(default_input), help="Input CSV path")
    p.add_argument("--output", default="", help="Output CSV path; empty means overwrite input")
    args = p.parse_args()

    in_csv = Path(args.input).resolve()
    if not in_csv.exists():
        raise FileNotFoundError(f"Input not found: {in_csv}")
    out_csv = Path(args.output).resolve() if str(args.output).strip() else in_csv

    df = pd.read_csv(in_csv, header=[0, 1])

    accel_z_col = ("accel", "z")
    gyro_z_col = ("gyro", "z")
    missing = [c for c in [accel_z_col, gyro_z_col] if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns: {missing}; available head: {list(df.columns)[:10]}")

    df[accel_z_col] = -df[accel_z_col].astype(float)
    df[gyro_z_col] = -df[gyro_z_col].astype(float)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"done: {in_csv} -> {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
