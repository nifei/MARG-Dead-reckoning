#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.use("Agg")

def find_workspace_root(start: Path) -> Path:
    for p in [start, *start.parents]:
        if (p / "data").exists() and (p / "PDR").exists():
            return p
    raise FileNotFoundError(f"cannot locate workspace root from: {start}")


def find_pdr_lib_root(start: Path) -> Path:
    for p in [start, *start.parents]:
        if (p / "PDR" / "config.py").exists() and (p / "PDR" / "__init__.py").exists():
            return p
    ws = find_workspace_root(start)
    cand = ws / "PDR" / "MARG-Dead-reckoning"
    if (cand / "PDR" / "config.py").exists():
        return cand
    raise FileNotFoundError(f"cannot locate PDR lib root from: {start}")


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = find_workspace_root(SCRIPT_DIR)
PDR_LIB_ROOT = find_pdr_lib_root(SCRIPT_DIR)
if str(PDR_LIB_ROOT) not in sys.path:
    sys.path.insert(0, str(PDR_LIB_ROOT))

from PDR.config import GnssConstraintConfig
from PDR.gnss_constraint import apply_gnss_velocity_constraints

EARTH_RADIUS_M = 6378137.0
GPS_EPOCH_UNIX = 315964800
GPS_UTC_LEAP_SECONDS = 18


@dataclass
class FixRow:
    provider: str
    lat: float
    lon: float
    alt: float
    speed: float
    acc: float
    t_ms: int


def gps_week_sow_to_unix_ms(week: float, sow: float) -> int:
    gps_sec = week * 604800.0 + sow
    unix_sec = gps_sec + GPS_EPOCH_UNIX - GPS_UTC_LEAP_SECONDS
    return int(round(unix_sec * 1000.0))


def read_part04_window_ms(sat_csv: Path) -> tuple[int, int]:
    weeks: list[float] = []
    sows: list[float] = []
    with sat_csv.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            weeks.append(float(row["week"]))
            sows.append(float(row["seconds of week [s]"]))
    if not weeks:
        raise ValueError(f"empty sat csv: {sat_csv}")
    start_ms = gps_week_sow_to_unix_ms(min(weeks), min(sows))
    end_ms = gps_week_sow_to_unix_ms(max(weeks), max(sows))
    return start_ms, end_ms


def parse_raw_fix_rows(raw_file: Path, start_ms: int, end_ms: int) -> list[FixRow]:
    out: list[FixRow] = []
    with raw_file.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if not s.startswith("Fix,"):
                continue
            cols = [x.strip() for x in s.split(",")]
            if len(cols) < 8:
                continue
            try:
                t = int(float(cols[7]))
                if t < start_ms or t > end_ms:
                    continue
                out.append(
                    FixRow(
                        provider=cols[1],
                        lat=float(cols[2]),
                        lon=float(cols[3]),
                        alt=float(cols[4]),
                        speed=float(cols[5]),
                        acc=float(cols[6]),
                        t_ms=t,
                    )
                )
            except ValueError:
                continue
    out.sort(key=lambda r: r.t_ms)
    # drop duplicate timestamps
    dedup: list[FixRow] = []
    for r in out:
        if not dedup or r.t_ms != dedup[-1].t_ms:
            dedup.append(r)
    return dedup


def fix_rows_to_df(rows: list[FixRow]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(
        {
            "provider": [r.provider for r in rows],
            "lat_deg": [r.lat for r in rows],
            "lon_deg": [r.lon for r in rows],
            "alt_m": [r.alt for r in rows],
            "speed_fix_mps": [r.speed for r in rows],
            "acc_fix_m": [r.acc for r in rows],
            "unix_ms": [r.t_ms for r in rows],
        }
    )
    df["unix_sec"] = df["unix_ms"] / 1000.0
    lat0 = np.deg2rad(float(df["lat_deg"].iloc[0]))
    lon0 = np.deg2rad(float(df["lon_deg"].iloc[0]))
    lat = np.deg2rad(df["lat_deg"].to_numpy(dtype=float))
    lon = np.deg2rad(df["lon_deg"].to_numpy(dtype=float))
    df["east_m"] = (lon - lon0) * np.cos(lat0) * EARTH_RADIUS_M
    df["north_m"] = (lat - lat0) * EARTH_RADIUS_M
    df["up_m"] = df["alt_m"] - float(df["alt_m"].iloc[0])
    de = np.diff(df["east_m"].to_numpy(dtype=float), prepend=np.nan)
    dn = np.diff(df["north_m"].to_numpy(dtype=float), prepend=np.nan)
    du = np.diff(df["up_m"].to_numpy(dtype=float), prepend=np.nan)
    dt = np.diff(df["unix_sec"].to_numpy(dtype=float), prepend=np.nan)
    speed_xyz = np.sqrt(de * de + dn * dn + du * du)
    speed_xyz = np.where(dt > 0, speed_xyz / dt, np.nan)
    df["speed_xyz_mps"] = pd.Series(speed_xyz).interpolate(limit_direction="both")
    return df


def compute_speed_metrics(gnss_speed: np.ndarray, est_speed: np.ndarray) -> dict[str, float]:
    m = np.isfinite(gnss_speed) & np.isfinite(est_speed)
    g = gnss_speed[m]
    e = est_speed[m]
    if len(e) == 0:
        return {"n": 0, "mae": float("nan"), "rmse": float("nan"), "corr": float("nan")}
    mae = float(np.mean(np.abs(e - g)))
    rmse = float(np.sqrt(np.mean((e - g) ** 2)))
    corr = float(np.corrcoef(e, g)[0, 1]) if len(e) > 2 else float("nan")
    return {"n": int(len(e)), "mae": mae, "rmse": rmse, "corr": corr}


def interp_to_times(src_t: np.ndarray, src_v: np.ndarray, target_t: np.ndarray) -> np.ndarray:
    keep = np.concatenate(([True], np.diff(src_t) > 0))
    t = src_t[keep]
    v = src_v[keep]
    if len(t) < 2:
        return np.full_like(target_t, np.nan, dtype=float)
    y = pd.Series(v).interpolate(limit_direction="both").to_numpy(dtype=float)
    return np.interp(target_t, t, y, left=np.nan, right=np.nan)


def wrap180(x: np.ndarray) -> np.ndarray:
    return (x + 180.0) % 360.0 - 180.0


def main() -> None:
    p = argparse.ArgumentParser(description="Build zupt-aided-mag-gnss trajectory from GNSS Fix constraints.")
    p.add_argument("--segment", default="seg_20260315_full_part04")
    p.add_argument("--satellite-csv", default=None, help="Override satellite_lla.csv path")
    p.add_argument("--raw-file", default=None, help="Override GNSS Raw_Log file path")
    p.add_argument("--in-dir", default=None, help="Override data/pdr_input/<segment> directory")
    p.add_argument("--out-dir", default=None, help="Override PDR/output/<segment>_three_impl directory")
    args = p.parse_args()

    root = REPO_ROOT
    seg = args.segment
    out_dir = Path(args.out_dir).resolve() if args.out_dir else (root / "PDR" / "output" / f"{seg}_three_impl")
    in_dir = Path(args.in_dir).resolve() if args.in_dir else (root / "data" / "pdr_input" / seg)
    out_dir.mkdir(parents=True, exist_ok=True)
    in_dir.mkdir(parents=True, exist_ok=True)

    sat_csv = Path(args.satellite_csv).resolve() if args.satellite_csv else (root / "data" / "bison_input" / seg / "satellite_lla.csv")
    raw_file = Path(args.raw_file).resolve() if args.raw_file else (root / "data" / "GNSS-IMU-Logger" / "Raw_Log" / "V2307A__RAW__20260315_merged.txt")
    zupt_mag_csv = out_dir / "zupt_aided_mag_pdr_traj_vel.csv"
    if not zupt_mag_csv.exists():
        raise FileNotFoundError(f"missing {zupt_mag_csv}; run run_pdr_three_impl_part04.py first")

    start_ms, end_ms = read_part04_window_ms(sat_csv)
    gnss_raw_csv = in_dir / "gnss_raw_fix_part04.csv"
    fix_rows = parse_raw_fix_rows(raw_file, start_ms, end_ms)
    gnss = fix_rows_to_df(fix_rows)
    if len(gnss) >= 10:
        gnss.to_csv(gnss_raw_csv, index=False)
        gnss_source = "raw_log"
    elif gnss_raw_csv.exists():
        gnss = pd.read_csv(gnss_raw_csv)
        gnss_source = "fallback_cached_csv"
    else:
        raise RuntimeError("too few GNSS Fix rows in part04 window")

    zupt_mag = pd.read_csv(zupt_mag_csv)
    t = zupt_mag["t_sec_abs"].to_numpy(dtype=float)
    vel = zupt_mag[["vel_x_mps", "vel_y_mps", "vel_z_mps"]].to_numpy(dtype=float)
    dt = np.diff(t, prepend=t[0])
    dt[dt <= 0] = 0.0
    label_cols = ["is_moving", "motion_context", "step_event", "zupt_candidate"]
    labels: dict[str, np.ndarray] = {}
    label_src_csv = out_dir / "zupt_aided_mag_internal_ned_traj_ned.csv"
    if label_src_csv.exists():
        src = pd.read_csv(label_src_csv)
        if "t_sec_abs" in src.columns:
            ts = src["t_sec_abs"].to_numpy(dtype=float)
            for col in label_cols:
                if col not in src.columns:
                    continue
                vals = src[col].to_numpy(dtype=float)
                labels[col] = (interp_to_times(ts, vals, t) > 0.5).astype(int)
    if "is_moving" not in labels:
        labels["is_moving"] = (np.linalg.norm(vel[:, :2], axis=1) > 0.1).astype(int)
    if "motion_context" not in labels:
        labels["motion_context"] = labels["is_moving"].copy()
    if "step_event" not in labels:
        up = np.diff(labels["motion_context"], prepend=labels["motion_context"][0])
        labels["step_event"] = (up > 0).astype(int)
    if "zupt_candidate" not in labels:
        labels["zupt_candidate"] = (labels["motion_context"] == 0).astype(int)

    out_df, interval_df, heading_metrics_raw = apply_gnss_velocity_constraints(
        t_sec_abs=t,
        vel_enu=vel,
        gnss_fix_df=gnss,
        labels=labels,
        cfg=GnssConstraintConfig(),
    )
    g_t = gnss["unix_sec"].to_numpy(dtype=float)
    g_e = gnss["east_m"].to_numpy(dtype=float)
    g_n = gnss["north_m"].to_numpy(dtype=float)
    interval_csv = out_dir / "zupt_aided_mag_gnss_interval_scales.csv"
    interval_df.to_csv(interval_csv, index=False)
    out_traj_csv = out_dir / "zupt_aided_mag_gnss_pdr_traj_vel.csv"
    for col in label_cols:
        out_df[col] = labels[col]
    out_df.to_csv(out_traj_csv, index=False)

    # Compare against GNSS + zupt_mag baseline on GNSS Fix timestamps.
    b_speed = interp_to_times(t, np.linalg.norm(vel, axis=1), g_t)
    n_speed = interp_to_times(t, out_df["speed_mps"].to_numpy(dtype=float), g_t)
    b_e = interp_to_times(t, zupt_mag["pos_x_m"].to_numpy(dtype=float), g_t)
    b_n = interp_to_times(t, zupt_mag["pos_y_m"].to_numpy(dtype=float), g_t)
    n_e = interp_to_times(t, out_df["pos_x_m"].to_numpy(dtype=float), g_t)
    n_n = interp_to_times(t, out_df["pos_y_m"].to_numpy(dtype=float), g_t)
    gnss_hd = np.degrees(np.arctan2(np.diff(g_e, prepend=np.nan), np.diff(g_n, prepend=np.nan)))
    est_hd = np.degrees(np.arctan2(np.diff(n_e, prepend=np.nan), np.diff(n_n, prepend=np.nan)))
    hd_res = np.abs(wrap180(est_hd - gnss_hd))
    g_speed = gnss["speed_xyz_mps"].to_numpy(dtype=float)

    aligned_csv = out_dir / "gnss_fix_time_aligned_zupt_mag_vs_gnss.csv"
    pd.DataFrame(
        {
            "unix_sec": g_t,
            "gnss_east_m": g_e,
            "gnss_north_m": g_n,
            "gnss_speed_mps": g_speed,
            "zupt_mag_east_m": b_e,
            "zupt_mag_north_m": b_n,
            "zupt_mag_speed_mps": b_speed,
            "zupt_mag_gnss_east_m": n_e,
            "zupt_mag_gnss_north_m": n_n,
            "zupt_mag_gnss_speed_mps": n_speed,
        }
    ).to_csv(aligned_csv, index=False)

    traj_png = out_dir / "traj_compare_zupt_mag_gnss_vs_gnss.png"
    plt.figure(figsize=(10, 9))
    plt.plot(g_e, g_n, label="GNSS Fix", linewidth=2.0)
    plt.plot(b_e, b_n, label="ZUPT-aided-mag")
    plt.plot(n_e, n_n, label="ZUPT-aided-mag-gnss")
    plt.xlabel("East [m]")
    plt.ylabel("North [m]")
    plt.title("Trajectory: ZUPT-mag-gnss vs GNSS (Part04)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.axis("equal")
    plt.tight_layout()
    plt.savefig(traj_png, dpi=180)
    plt.close()

    speed_png = out_dir / "speed_trend_compare_zupt_mag_gnss_vs_gnss.png"
    tx = g_t - g_t[0]
    plt.figure(figsize=(12, 6))
    plt.plot(tx, g_speed, label="GNSS Fix speed", linewidth=2.0)
    plt.plot(tx, b_speed, label="ZUPT-aided-mag speed")
    plt.plot(tx, n_speed, label="ZUPT-aided-mag-gnss speed")
    plt.xlabel("Time from start [s]")
    plt.ylabel("Speed [m/s]")
    plt.title("Speed Trend: ZUPT-mag-gnss vs GNSS (Part04)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(speed_png, dpi=180)
    plt.close()

    metrics = {
        "zupt_mag_vs_gnss": compute_speed_metrics(g_speed, b_speed),
        "zupt_mag_gnss_vs_gnss": compute_speed_metrics(g_speed, n_speed),
    }
    heading_metrics = {
        "count": int(np.isfinite(hd_res).sum()),
        "mean_abs_deg": float(np.nanmean(hd_res)),
        "p95_abs_deg": float(np.nanpercentile(hd_res[np.isfinite(hd_res)], 95)),
        "mean_abs_deg_raw": heading_metrics_raw["heading_mean_abs_deg"],
        "p95_abs_deg_raw": heading_metrics_raw["heading_p95_abs_deg"],
        "gap_interval_ratio": heading_metrics_raw["gap_interval_ratio"],
    }
    summary = {
        "segment": seg,
        "inputs": {
            "raw_log_fix_file": str(raw_file),
            "satellite_lla_file": str(sat_csv),
            "base_zupt_mag_csv": str(zupt_mag_csv),
            "gnss_raw_fix_csv": str(gnss_raw_csv),
            "gnss_source": gnss_source,
        },
        "time_window_ms": {"start": int(start_ms), "end": int(end_ms)},
        "gnss_fix_rows": int(len(gnss)),
        "output_files": {
            "zupt_mag_gnss_csv": str(out_traj_csv),
            "interval_scale_csv": str(interval_csv),
            "aligned_compare_csv": str(aligned_csv),
            "traj_compare_png": str(traj_png),
            "speed_compare_png": str(speed_png),
        },
        "label_source": str(label_src_csv) if label_src_csv.exists() else "fallback_from_trajectory",
        "label_ratio": {k: float(np.mean(v)) for k, v in labels.items()},
        "heading_metrics_vs_gnss_fix": heading_metrics,
        "speed_metrics_vs_gnss_fix": metrics,
    }
    summary_json = out_dir / "summary_zupt_aided_mag_gnss.json"
    summary_json.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
