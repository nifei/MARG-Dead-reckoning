#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import Transformer

GPS_EPOCH_UNIX = 315964800.0
GPS_UTC_LEAP_SECONDS = 18.0


def wrap180(x: np.ndarray) -> np.ndarray:
    return (x + 180.0) % 360.0 - 180.0


def gps_week_sow_to_unix(week: np.ndarray, sow: np.ndarray) -> np.ndarray:
    return week * 604800.0 + sow + GPS_EPOCH_UNIX - GPS_UTC_LEAP_SECONDS


def lla_to_local_ne(lat_deg: np.ndarray, lon_deg: np.ndarray, lat0_deg: float, lon0_deg: float) -> tuple[np.ndarray, np.ndarray]:
    lat = np.deg2rad(lat_deg)
    lon = np.deg2rad(lon_deg)
    lat0 = math.radians(lat0_deg)
    lon0 = math.radians(lon0_deg)
    r = 6378137.0
    north = (lat - lat0) * r
    east = (lon - lon0) * r * math.cos(lat0)
    return north, east


def ecef_vel_to_enu(vx: float, vy: float, vz: float, lat_deg: float, lon_deg: float) -> tuple[float, float, float]:
    lat = math.radians(lat_deg)
    lon = math.radians(lon_deg)
    s_lat, c_lat = math.sin(lat), math.cos(lat)
    s_lon, c_lon = math.sin(lon), math.cos(lon)
    r = np.array(
        [
            [-s_lon, c_lon, 0.0],
            [-s_lat * c_lon, -s_lat * s_lon, c_lat],
            [c_lat * c_lon, c_lat * s_lon, s_lat],
        ],
        dtype=float,
    )
    out = r @ np.array([vx, vy, vz], dtype=float)
    return float(out[0]), float(out[1]), float(out[2])


def detect_step_events(t_sec: np.ndarray, speed_mps: np.ndarray, thr: float = 0.75, min_interval_s: float = 0.35) -> np.ndarray:
    idx = []
    last_t = -1e9
    for i in range(1, len(speed_mps) - 1):
        if speed_mps[i] < thr:
            continue
        if not (speed_mps[i] >= speed_mps[i - 1] and speed_mps[i] >= speed_mps[i + 1]):
            continue
        ti = t_sec[i]
        if ti - last_t < min_interval_s:
            continue
        idx.append(i)
        last_t = ti
    return np.asarray(idx, dtype=int)


def running_ewm(x: np.ndarray, alpha: float) -> np.ndarray:
    y = np.full_like(x, np.nan, dtype=float)
    last = np.nan
    for i, xi in enumerate(x):
        if not np.isfinite(xi):
            y[i] = last
            continue
        if not np.isfinite(last):
            last = xi
        else:
            last = (1.0 - alpha) * last + alpha * xi
        y[i] = last
    return y


def running_ewm_gated(x: np.ndarray, alpha: float, update_mask: np.ndarray) -> np.ndarray:
    y = np.full_like(x, np.nan, dtype=float)
    last = np.nan
    for i, xi in enumerate(x):
        if not np.isfinite(xi):
            y[i] = last
            continue
        if not np.isfinite(last):
            last = xi
            y[i] = last
            continue
        if bool(update_mask[i]):
            last = (1.0 - alpha) * last + alpha * xi
        y[i] = last
    return y


def running_circular_ewm_staged(x_deg: np.ndarray, alpha_good: float, alpha_weak: float, stage_code: np.ndarray) -> np.ndarray:
    # stage: 0=Good, 1=Weak, 2=Denied
    y = np.full_like(x_deg, np.nan, dtype=float)
    last = np.nan
    for i, xi in enumerate(x_deg):
        if not np.isfinite(xi):
            y[i] = last
            continue
        if not np.isfinite(last):
            last = xi
            y[i] = last
            continue
        alpha = 0.0
        if int(stage_code[i]) == 0:
            alpha = alpha_good
        elif int(stage_code[i]) == 1:
            alpha = alpha_weak
        # denied: alpha=0 => freeze
        if alpha > 0.0:
            d = wrap180(np.array([xi - last], dtype=float))[0]
            last = (last + alpha * d + 360.0) % 360.0
        y[i] = last
    return y


def convergence_time_sec(t: np.ndarray, x: np.ndarray, abs_tol: float, stable_window_s: float) -> float:
    m = np.isfinite(t) & np.isfinite(x)
    if np.count_nonzero(m) < 10:
        return float("nan")
    tt = t[m]
    xx = x[m]
    target = float(np.nanmedian(xx[max(0, len(xx) - 30) :]))
    ok = np.abs(xx - target) <= abs_tol
    for i in range(len(tt)):
        t0 = tt[i]
        t1 = t0 + stable_window_s
        j = np.searchsorted(tt, t1, side="left")
        if j <= i + 1:
            continue
        if np.all(ok[i:j]):
            return float(t0 - tt[0])
    return float("nan")


def convergence_status(x: float) -> str:
    return "converged" if np.isfinite(x) else "not_converged"


def validate_input_schema(sat: pd.DataFrame, odom: pd.DataFrame) -> dict[str, object]:
    sat_need = ["week", "seconds of week [s]", "Latitude", "Longitude", "Altitude"]
    odom_need = ["seconds of week [s]", "ECEF_vel_x", "ECEF_vel_y", "ECEF_vel_z", "GPS(0):Lat[degrees]", "GPS(0):Long[degrees]", "GPS(0):heightMSL[meters]"]
    missing_sat = [c for c in sat_need if c not in sat.columns]
    missing_odom = [c for c in odom_need if c not in odom.columns]
    if missing_sat or missing_odom:
        raise ValueError(f"missing columns sat={missing_sat}, odom={missing_odom}")
    sow_sat = sat["seconds of week [s]"].to_numpy(dtype=float)
    sow_odom = odom["seconds of week [s]"].to_numpy(dtype=float)
    if np.any(np.diff(sow_sat) < 0):
        raise ValueError("satellite_lla seconds of week not monotonic")
    if np.any(np.diff(sow_odom) < 0):
        raise ValueError("odom seconds of week not monotonic")
    return {"sat_rows": int(len(sat)), "odom_rows": int(len(odom))}


def main() -> None:
    p = argparse.ArgumentParser(description="Role-B v2: gnss-sensor-fusion outputs + GNSS-good convergence validation (part04).")
    p.add_argument("--root", default=".")
    p.add_argument("--segment", default="seg_20260315_full_part04")
    p.add_argument("--out-dir", default="phase_a_output/step10_zupt_mag_gnss_part04_base")
    args = p.parse_args()

    root = Path(args.root).resolve()
    out_dir = (root / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    sat_csv = root / "data" / "bison_input" / args.segment / "satellite_lla.csv"
    odom_csv = out_dir / "step9_odom_zupt_aided_mag_gnss.csv"
    pdr_traj_csv = root / "PDR" / "output" / f"{args.segment}_three_impl" / "zupt_aided_mag_gnss_pdr_traj_vel.csv"
    pdr_att_csv = root / "PDR" / "output" / f"{args.segment}_three_impl" / "zupt_aided_mag_internal_ned_quat_ned.csv"
    sat = pd.read_csv(sat_csv)
    odom = pd.read_csv(odom_csv)
    schema_report = validate_input_schema(sat, odom)

    import sys

    sys.path.insert(0, str(root / "gnss-sensor-fusion"))
    from gnss_fusion_ekf import EKF  # pylint: disable=import-error

    ekf = EKF(str(sat_csv), str(odom_csv))
    ekf.run()
    run_df = ekf.get_run_dataframe()
    ekf_sow = run_df["seconds of week [s]"].to_numpy(dtype=float)
    ekf_x = run_df["x_ecef_m"].to_numpy(dtype=float)
    ekf_y = run_df["y_ecef_m"].to_numpy(dtype=float)
    ekf_z = run_df["z_ecef_m"].to_numpy(dtype=float)
    week = sat["week"].to_numpy(dtype=float)
    sow = sat["seconds of week [s]"].to_numpy(dtype=float)
    x_aligned = np.interp(sow, ekf_sow, ekf_x)
    y_aligned = np.interp(sow, ekf_sow, ekf_y)
    z_aligned = np.interp(sow, ekf_sow, ekf_z)
    tr = Transformer.from_crs("EPSG:4978", "EPSG:4979", always_xy=True)
    lon_f, lat_f, alt_f = tr.transform(x_aligned, y_aligned, z_aligned)
    traj_csv = out_dir / "step10_traj_with_zupt_aided_mag_gnss_odom.csv"
    pd.DataFrame({"latitude": lat_f, "longitude": lon_f, "altitude": alt_f}).to_csv(traj_csv, index=False)

    t_unix = gps_week_sow_to_unix(week, sow)
    lat = sat["Latitude"].to_numpy(dtype=float)
    lon = sat["Longitude"].to_numpy(dtype=float)
    alt = sat["Altitude"].to_numpy(dtype=float)
    north, east = lla_to_local_ne(lat, lon, float(lat[0]), float(lon[0]))
    dt = np.diff(t_unix, prepend=t_unix[0])
    d_n = np.diff(north, prepend=north[0])
    d_e = np.diff(east, prepend=east[0])
    gnss_vn = np.divide(d_n, dt, out=np.zeros_like(d_n), where=dt > 0)
    gnss_ve = np.divide(d_e, dt, out=np.zeros_like(d_e), where=dt > 0)
    gnss_spd = np.hypot(gnss_vn, gnss_ve)
    heading_gnss = (np.degrees(np.arctan2(gnss_ve, gnss_vn)) + 360.0) % 360.0

    od_vx = odom["ECEF_vel_x"].to_numpy(dtype=float)
    od_vy = odom["ECEF_vel_y"].to_numpy(dtype=float)
    od_vz = odom["ECEF_vel_z"].to_numpy(dtype=float)
    od_enu = np.array([ecef_vel_to_enu(od_vx[i], od_vy[i], od_vz[i], lat[i], lon[i]) for i in range(len(odom))], dtype=float)
    od_spd = np.linalg.norm(od_enu, axis=1)
    heading_odom = (np.degrees(np.arctan2(od_enu[:, 0], od_enu[:, 1])) + 360.0) % 360.0
    use_odom = od_spd > 0.25
    heading_ref_raw = np.where(use_odom, heading_odom, heading_gnss)

    # quality_score/stage: Good(0)/Weak(1)/Denied(2)
    speed_score = np.clip((gnss_spd - 0.2) / 1.3, 0.0, 1.0)
    head_score = np.clip(1.0 - (np.abs(wrap180(heading_odom - heading_gnss)) / 120.0), 0.0, 1.0)
    quality_score = 0.45 * speed_score + 0.55 * head_score
    heading_consistency = np.abs(wrap180(heading_odom - heading_gnss))
    stage = np.full(len(quality_score), 1, dtype=int)
    is_good = (quality_score >= 0.75) & (gnss_spd >= 0.5) & (heading_consistency < 45.0)
    is_denied = (gnss_spd < 0.2) | (heading_consistency >= 120.0) | (quality_score < 0.2)
    stage[is_good] = 0
    stage[is_denied] = 2
    quality_flag = (stage == 0).astype(int)
    heading_ref = running_circular_ewm_staged(heading_ref_raw, alpha_good=0.20, alpha_weak=0.03, stage_code=stage)

    pdr_att = pd.read_csv(pdr_att_csv)
    pdr_t = pdr_att["t_sec_abs"].to_numpy(dtype=float)
    pdr_yaw = pdr_att["yaw_deg"].to_numpy(dtype=float)
    pdr_yaw_interp = np.interp(t_unix, pdr_t, pdr_yaw)
    yaw_offset = wrap180(pdr_yaw_interp - heading_ref)
    yaw_update = stage != 2
    yaw_alpha = np.where(stage == 0, 0.05, 0.01)
    yaw_offset_ewm = np.full_like(yaw_offset, np.nan, dtype=float)
    last_yaw = np.nan
    for i, yv in enumerate(yaw_offset):
        if not np.isfinite(yv):
            yaw_offset_ewm[i] = last_yaw
            continue
        if not np.isfinite(last_yaw):
            last_yaw = wrap180(np.array([yv], dtype=float))[0]
        elif yaw_update[i]:
            a = float(yaw_alpha[i])
            dy = wrap180(np.array([yv - last_yaw], dtype=float))[0]
            last_yaw = wrap180(np.array([last_yaw + a * dy], dtype=float))[0]
        yaw_offset_ewm[i] = wrap180(np.array([last_yaw], dtype=float))[0]

    dt = np.diff(t_unix, prepend=t_unix[0])
    dyaw_deg = np.zeros_like(yaw_offset_ewm)
    dyaw_deg[1:] = wrap180(yaw_offset_ewm[1:] - yaw_offset_ewm[:-1])
    bgz_inst = np.full_like(yaw_offset_ewm, np.nan, dtype=float)
    valid_bias_obs = (dt > 0.0) & np.isfinite(yaw_offset_ewm)
    bgz_inst[valid_bias_obs] = np.deg2rad(dyaw_deg[valid_bias_obs]) / dt[valid_bias_obs]

    pdr_traj = pd.read_csv(pdr_traj_csv)
    step_idx = detect_step_events(
        pdr_traj["t_sec_abs"].to_numpy(dtype=float),
        pdr_traj["speed_mps"].to_numpy(dtype=float),
    )
    step_t = pdr_traj["t_sec_abs"].to_numpy(dtype=float)[step_idx]
    gnss_n_at_step = np.interp(step_t, t_unix, north)
    gnss_e_at_step = np.interp(step_t, t_unix, east)
    step_len = np.hypot(np.diff(gnss_n_at_step, prepend=gnss_n_at_step[0]), np.diff(gnss_e_at_step, prepend=gnss_e_at_step[0]))
    step_len = np.clip(step_len, 0.0, 2.0)
    step_len_ref = np.full_like(t_unix, np.nan, dtype=float)
    if len(step_t) > 1:
        step_len_ref = np.interp(t_unix, step_t, step_len, left=np.nan, right=np.nan)
    step_len_ref_ewm = running_ewm(step_len_ref, alpha=0.08)

    bias_gate = (stage == 0) & (gnss_spd > 0.5) & (np.abs(bgz_inst) < 0.3)
    bgz_inst = np.where(bias_gate, bgz_inst, np.nan)
    bgz_ref = running_ewm_gated(np.where(np.isfinite(bgz_inst), bgz_inst, np.nan), alpha=0.03, update_mask=(stage == 0))

    stage_label = np.where(stage == 0, "Good", np.where(stage == 1, "Weak", "Denied"))

    ref_csv = out_dir / "gnss_sensor_fusion_pdr_refs_part04.csv"
    ref_df = pd.DataFrame(
        {
            "week": week,
            "sow": sow,
            "heading_ref_deg": heading_ref,
            "heading_ref_raw_deg": heading_ref_raw,
            "heading_gnss_deg": heading_gnss,
            "heading_odom_deg": heading_odom,
            "heading_consistency_deg": heading_consistency,
            "bgz_ref_radps": bgz_ref,
            "yaw_offset_motion_deg": yaw_offset_ewm,
            "step_length_ref_m": step_len_ref_ewm,
            "quality_flag": quality_flag,
            "quality_score": quality_score,
            "stage": stage_label,
            "gnss_speed_h_mps": gnss_spd,
            "odom_speed_3d_mps": od_spd,
        }
    )
    ref_df.to_csv(ref_csv, index=False)

    att_out_csv = out_dir / "step10_attitude_from_zupt_aided_mag_for_mag_gnss.csv"
    pd.DataFrame(
        {
            "week": week,
            "sow": sow,
            "roll_deg": np.interp(t_unix, pdr_t, pdr_att["roll_deg"].to_numpy(dtype=float)),
            "pitch_deg": np.interp(t_unix, pdr_t, pdr_att["pitch_deg"].to_numpy(dtype=float)),
            "yaw_deg": pdr_yaw_interp,
            "attitude_source": np.repeat("zupt_aided_mag_internal_ned_quat_ned", len(week)),
        }
    ).to_csv(att_out_csv, index=False)

    conv_csv = out_dir / "gnss_sensor_fusion_param_convergence_part04.csv"
    conv_df = pd.DataFrame(
        {
            "week": week,
            "sow": sow,
            "t_rel_s": t_unix - t_unix[0],
            "bgz_ref_radps": bgz_ref,
            "step_length_ref_m": step_len_ref_ewm,
            "yaw_offset_motion_deg": yaw_offset_ewm,
            "quality_flag": quality_flag,
        }
    )
    conv_df.to_csv(conv_csv, index=False)

    bg_conv_s = convergence_time_sec(conv_df["t_rel_s"].to_numpy(dtype=float), bgz_ref, abs_tol=math.radians(0.25), stable_window_s=30.0)
    step_conv_s = convergence_time_sec(conv_df["t_rel_s"].to_numpy(dtype=float), step_len_ref_ewm, abs_tol=0.08, stable_window_s=30.0)
    yaw_conv_s = convergence_time_sec(conv_df["t_rel_s"].to_numpy(dtype=float), yaw_offset_ewm, abs_tol=10.0, stable_window_s=30.0)

    report = {
        "segment": args.segment,
        "date": "2026-03-17",
        "inputs": {
            "satellite_lla_csv": str(sat_csv),
            "odom_csv": str(odom_csv),
            "pdr_traj_csv": str(pdr_traj_csv),
            "pdr_attitude_csv": str(pdr_att_csv),
        },
        "schema_check": schema_report,
        "outputs": {
            "step10_traj_csv": str(traj_csv),
            "step10_attitude_csv": str(att_out_csv),
            "teacher_refs_csv": str(ref_csv),
            "param_convergence_csv": str(conv_csv),
            "ekf_run_debug_csv": str(out_dir / "gnss_sensor_fusion_run_debug_part04.csv"),
        },
        "metrics": {
            "traj_rows": int(len(lat_f)),
            "sat_rows": int(len(sat)),
            "ekf_internal_rows_unique_sow": int(len(ekf_sow)),
            "rows": int(len(ref_df)),
            "quality_ratio": float(np.mean(quality_flag)),
            "stage_good_ratio": float(np.mean(stage == 0)),
            "stage_weak_ratio": float(np.mean(stage == 1)),
            "stage_denied_ratio": float(np.mean(stage == 2)),
            "heading_consistency_mean_deg": float(np.nanmean(heading_consistency)),
            "bgz_ref_median_radps": float(np.nanmedian(bgz_ref)),
            "step_length_ref_median_m": float(np.nanmedian(step_len_ref_ewm)),
            "yaw_offset_motion_median_deg": float(np.nanmedian(yaw_offset_ewm)),
            "convergence_time_bgz_s": bg_conv_s,
            "convergence_time_step_length_s": step_conv_s,
            "convergence_time_yaw_offset_s": yaw_conv_s,
            "step_event_count": int(len(step_idx)),
        },
        "convergence_status": {
            "bgz_ref": convergence_status(bg_conv_s),
            "step_length_ref": convergence_status(step_conv_s),
            "yaw_offset_motion": convergence_status(yaw_conv_s),
        },
    }
    run_df.to_csv(out_dir / "gnss_sensor_fusion_run_debug_part04.csv", index=False)
    summary_json = out_dir / "summary_roleB_gnss_sensor_fusion_v2.json"
    summary_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
