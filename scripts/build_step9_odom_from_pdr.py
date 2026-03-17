#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

GPS_EPOCH_UNIX = 315964800.0
GPS_UTC_LEAP_SECONDS = 18.0


def enu_to_ecef_velocity(v_e: float, v_n: float, v_u: float, lat_deg: float, lon_deg: float) -> tuple[float, float, float]:
    lat = np.deg2rad(lat_deg)
    lon = np.deg2rad(lon_deg)
    s_lat, c_lat = np.sin(lat), np.cos(lat)
    s_lon, c_lon = np.sin(lon), np.cos(lon)
    r = np.array(
        [
            [-s_lon, -s_lat * c_lon, c_lat * c_lon],
            [c_lon, -s_lat * s_lon, c_lat * s_lon],
            [0.0, c_lat, s_lat],
        ],
        dtype=float,
    )
    out = r @ np.array([v_e, v_n, v_u], dtype=float)
    return float(out[0]), float(out[1]), float(out[2])


def nearest_indices(ref_t: np.ndarray, query_t: np.ndarray) -> np.ndarray:
    pos = np.searchsorted(ref_t, query_t, side="left")
    pos = np.clip(pos, 0, len(ref_t) - 1)
    prev = np.clip(pos - 1, 0, len(ref_t) - 1)
    choose_prev = np.abs(query_t - ref_t[prev]) <= np.abs(query_t - ref_t[pos])
    return np.where(choose_prev, prev, pos)


def build_step9(
    sat_csv: Path,
    pdr_traj_csv: Path,
    out_csv: Path,
    gap_dt_thr_s: float = 2.0,
    ramp_sec: float = 8.0,
) -> dict[str, object]:
    sat = pd.read_csv(sat_csv)
    pdr = pd.read_csv(pdr_traj_csv)

    sat_week = sat["week"].to_numpy(dtype=float)
    sat_sow = sat["seconds of week [s]"].to_numpy(dtype=float)
    sat_lat = sat["Latitude"].to_numpy(dtype=float)
    sat_lon = sat["Longitude"].to_numpy(dtype=float)
    sat_alt = sat["Altitude"].to_numpy(dtype=float)
    sat_unix_sec = sat_week * 604800.0 + sat_sow + GPS_EPOCH_UNIX - GPS_UTC_LEAP_SECONDS

    pdr_t = pdr["t_sec_abs"].to_numpy(dtype=float)
    ve = np.interp(sat_unix_sec, pdr_t, pdr["vel_x_mps"].to_numpy(dtype=float))
    vn = np.interp(sat_unix_sec, pdr_t, pdr["vel_y_mps"].to_numpy(dtype=float))
    vu = np.interp(sat_unix_sec, pdr_t, pdr["vel_z_mps"].to_numpy(dtype=float))
    sat_dt = np.diff(sat_unix_sec, prepend=sat_unix_sec[0])
    is_gap_recover_point = sat_dt > float(gap_dt_thr_s)

    if {"gnss_gap_flag", "gap_age_s", "heading_reliable_flag"}.issubset(set(pdr.columns)):
        idx = nearest_indices(pdr_t, sat_unix_sec)
        pdr_gap = pdr["gnss_gap_flag"].to_numpy(dtype=float)[idx]
        pdr_gap_age = pdr["gap_age_s"].to_numpy(dtype=float)[idx]
        pdr_heading_rel = pdr["heading_reliable_flag"].to_numpy(dtype=float)[idx]
    else:
        pdr_gap = np.zeros(len(sat), dtype=float)
        pdr_gap_age = np.zeros(len(sat), dtype=float)
        pdr_heading_rel = np.ones(len(sat), dtype=float)

    gnss_update_weight = np.ones(len(sat), dtype=float)
    ramp_age_s = np.full(len(sat), np.nan, dtype=float)
    current_ramp_age = np.nan
    for i in range(len(sat)):
        if pdr_gap[i] >= 0.5:
            gnss_update_weight[i] = 0.0
            current_ramp_age = np.nan
            continue
        if bool(is_gap_recover_point[i]):
            current_ramp_age = 0.0
        elif np.isfinite(current_ramp_age):
            current_ramp_age += max(float(sat_dt[i]), 0.0)
        if np.isfinite(current_ramp_age):
            ramp_age_s[i] = current_ramp_age
            gnss_update_weight[i] = float(np.clip(current_ramp_age / max(ramp_sec, 1e-6), 0.0, 1.0))
        if pdr_heading_rel[i] < 0.5:
            gnss_update_weight[i] = min(gnss_update_weight[i], 0.3)

    # Gap freeze strategy: during declared GNSS gap, do not trust odom prediction.
    freeze_mask = pdr_gap >= 0.5
    ve = np.where(freeze_mask, 0.0, ve)
    vn = np.where(freeze_mask, 0.0, vn)
    vu = np.where(freeze_mask, 0.0, vu)
    # Recovery ramp is also applied to odom prediction to avoid one-step cross-gap drift.
    ve = ve * gnss_update_weight
    vn = vn * gnss_update_weight
    vu = vu * gnss_update_weight

    ecef_vel = np.array(
        [enu_to_ecef_velocity(ve[i], vn[i], vu[i], sat_lat[i], sat_lon[i]) for i in range(len(sat))],
        dtype=float,
    )

    odom = pd.DataFrame(
        {
            "": np.arange(len(sat), dtype=int),
            "seconds of week [s]": sat_sow,
            "ECEF_vel_x": ecef_vel[:, 0],
            "ECEF_vel_y": ecef_vel[:, 1],
            "ECEF_vel_z": ecef_vel[:, 2],
            "GPS(0):Lat[degrees]": sat_lat,
            "GPS(0):Long[degrees]": sat_lon,
            "GPS(0):heightMSL[meters]": sat_alt,
            "Normalized barometer:Raw[meters]": np.zeros(len(sat), dtype=float),
            "FixSpeed[m/s]": np.linalg.norm(np.column_stack([ve, vn, vu]), axis=1),
            "FixAccuracy[m]": np.full(len(sat), 5.0, dtype=float),
            "GNSS_Update_Weight": gnss_update_weight,
            "gnss_gap_flag": pdr_gap,
            "gap_age_s": pdr_gap_age,
            "heading_reliable_flag": pdr_heading_rel,
            "sat_dt_s": sat_dt,
            "is_gap_recover_point": is_gap_recover_point.astype(int),
            "ramp_age_s": ramp_age_s,
        }
    )
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    odom.to_csv(out_csv, index=False)

    report = {
        "satellite_csv": str(sat_csv),
        "pdr_traj_csv": str(pdr_traj_csv),
        "step9_odom_csv": str(out_csv),
        "rows": int(len(odom)),
        "gap_dt_thr_s": float(gap_dt_thr_s),
        "ramp_sec": float(ramp_sec),
        "gap_recover_points": int(np.sum(is_gap_recover_point)),
        "weight_min": float(np.min(gnss_update_weight)),
        "weight_mean": float(np.mean(gnss_update_weight)),
        "sow_min": float(np.min(sat_sow)),
        "sow_max": float(np.max(sat_sow)),
    }
    return report


def main() -> None:
    p = argparse.ArgumentParser(description="Build step9 odom csv for gnss-sensor-fusion from PDR ENU velocity.")
    p.add_argument("--satellite-csv", required=True)
    p.add_argument("--pdr-traj-csv", required=True)
    p.add_argument("--output-csv", required=True)
    p.add_argument("--gap-dt-thr-s", type=float, default=2.0)
    p.add_argument("--ramp-sec", type=float, default=8.0)
    args = p.parse_args()

    report = build_step9(
        Path(args.satellite_csv).resolve(),
        Path(args.pdr_traj_csv).resolve(),
        Path(args.output_csv).resolve(),
        gap_dt_thr_s=float(args.gap_dt_thr_s),
        ramp_sec=float(args.ramp_sec),
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
