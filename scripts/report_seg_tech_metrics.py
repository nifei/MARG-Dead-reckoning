#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.use("Agg")

EARTH_RADIUS_M = 6378137.0
GPS_EPOCH_UNIX = 315964800.0
GPS_UTC_LEAP_SECONDS = 18.0


def stats(x: np.ndarray) -> dict[str, float]:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return {"count": 0, "mean": float("nan"), "median": float("nan"), "p90": float("nan"), "p95": float("nan"), "max": float("nan")}
    return {
        "count": int(len(x)),
        "mean": float(np.mean(x)),
        "median": float(np.median(x)),
        "p90": float(np.percentile(x, 90)),
        "p95": float(np.percentile(x, 95)),
        "max": float(np.max(x)),
    }


def wrap180(x: np.ndarray) -> np.ndarray:
    return (x + 180.0) % 360.0 - 180.0


def interp(ts: np.ndarray, ys: np.ndarray, tq: np.ndarray) -> np.ndarray:
    keep = np.concatenate(([True], np.diff(ts) > 0))
    t = ts[keep]
    y = ys[keep]
    if len(t) < 2:
        return np.full_like(tq, np.nan, dtype=float)
    y = pd.Series(y).interpolate(limit_direction="both").to_numpy(dtype=float)
    return np.interp(tq, t, y, left=np.nan, right=np.nan)


def ll_to_local_ne(lat_deg: np.ndarray, lon_deg: np.ndarray, lat0_deg: float, lon0_deg: float) -> tuple[np.ndarray, np.ndarray]:
    lat = np.deg2rad(lat_deg)
    lon = np.deg2rad(lon_deg)
    lat0 = np.deg2rad(lat0_deg)
    lon0 = np.deg2rad(lon0_deg)
    north = (lat - lat0) * EARTH_RADIUS_M
    east = (lon - lon0) * np.cos(lat0) * EARTH_RADIUS_M
    return north, east


def path_length_2d(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.sum(np.hypot(np.diff(x), np.diff(y))))


def path_length_3d(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> float:
    dx = np.diff(x)
    dy = np.diff(y)
    dz = np.diff(z)
    return float(np.sum(np.sqrt(dx * dx + dy * dy + dz * dz)))


def speed_metrics(est: np.ndarray, gt: np.ndarray) -> dict[str, float]:
    m = np.isfinite(est) & np.isfinite(gt)
    e = est[m]
    g = gt[m]
    if len(e) == 0:
        return {"n": 0, "mae": float("nan"), "rmse": float("nan"), "corr": float("nan")}
    return {
        "n": int(len(e)),
        "mae": float(np.mean(np.abs(e - g))),
        "rmse": float(np.sqrt(np.mean((e - g) ** 2))),
        "corr": float(np.corrcoef(e, g)[0, 1]) if len(e) > 2 else float("nan"),
    }


def eval_windows(gt_t: np.ndarray, ge: np.ndarray, gn: np.ndarray, ee: np.ndarray, en: np.ndarray, durations: list[int]) -> dict[str, object]:
    out: dict[str, object] = {}
    for dur in durations:
        pe, he = [], []
        for i in range(len(gt_t)):
            t_end = gt_t[i] + float(dur)
            j = int(np.searchsorted(gt_t, t_end, side="left"))
            if j <= i or j >= len(gt_t):
                continue
            dge, dgn = ge[j] - ge[i], gn[j] - gn[i]
            dee, den = ee[j] - ee[i], en[j] - en[i]
            pe.append(np.hypot(dee - dge, den - dgn))
            hg = np.degrees(np.arctan2(dge, dgn))
            hee = np.degrees(np.arctan2(dee, den))
            he.append(abs(wrap180(np.array([hee - hg]))[0]))
        out[f"{dur}s"] = {"windows": int(len(pe)), "pos_err_m": stats(np.asarray(pe)), "heading_err_deg": stats(np.asarray(he))}
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Build technical metrics report for segment.")
    p.add_argument("--segment", default="seg_20260317_1354")
    p.add_argument("--root", default=".")
    p.add_argument("--out-dir", default=None)
    args = p.parse_args()

    root = Path(args.root).resolve()
    seg = args.segment
    out_dir = Path(args.out_dir).resolve() if args.out_dir else (root / "phase_a_output" / f"report_{seg}")
    out_dir.mkdir(parents=True, exist_ok=True)

    gnss_fix_csv = root / "data" / "pdr_input" / seg / "gnss_raw_fix_part04.csv"
    fusion_csv = root / "phase_a_output" / f"step10_{seg}_base" / "step10_traj_with_zupt_aided_mag_gnss_odom.csv"
    pdr_csv = root / "PDR" / "output" / f"{seg}_three_impl" / "zupt_aided_mag_gnss_pdr_traj_vel.csv"
    refs_csv = root / "phase_a_output" / f"step10_{seg}_base" / "gnss_sensor_fusion_pdr_refs_part04.csv"

    gnss = pd.read_csv(gnss_fix_csv).sort_values("unix_sec").reset_index(drop=True)
    fusion = pd.read_csv(fusion_csv)
    pdr = pd.read_csv(pdr_csv)
    refs = pd.read_csv(refs_csv)

    n = min(len(gnss), len(fusion), len(refs))
    gnss = gnss.iloc[:n].copy()
    fusion = fusion.iloc[:n].copy()
    refs = refs.iloc[:n].copy()

    lat0 = float(gnss["lat_deg"].iloc[0])
    lon0 = float(gnss["lon_deg"].iloc[0])
    g_n, g_e = ll_to_local_ne(gnss["lat_deg"].to_numpy(dtype=float), gnss["lon_deg"].to_numpy(dtype=float), lat0, lon0)
    g_u = gnss["up_m"].to_numpy(dtype=float)
    f_n, f_e = ll_to_local_ne(fusion["latitude"].to_numpy(dtype=float), fusion["longitude"].to_numpy(dtype=float), lat0, lon0)
    f_u = fusion["altitude"].to_numpy(dtype=float) - float(fusion["altitude"].iloc[0])

    gt = gnss["unix_sec"].to_numpy(dtype=float)
    p_t = pdr["t_sec_abs"].to_numpy(dtype=float)
    p_e = interp(p_t, pdr["pos_x_m"].to_numpy(dtype=float), gt)
    p_n = interp(p_t, pdr["pos_y_m"].to_numpy(dtype=float), gt)
    p_u = interp(p_t, pdr["pos_z_m"].to_numpy(dtype=float), gt)

    err_f_h = np.hypot(f_e - g_e, f_n - g_n)
    err_f_3d = np.sqrt((f_e - g_e) ** 2 + (f_n - g_n) ** 2 + (f_u - g_u) ** 2)
    err_p_h = np.hypot(p_e - g_e, p_n - g_n)

    g_len2 = path_length_2d(g_e, g_n)
    f_len2 = path_length_2d(f_e, f_n)
    mp = np.isfinite(p_e) & np.isfinite(p_n)
    p_len2 = path_length_2d(p_e[mp], p_n[mp]) if np.count_nonzero(mp) > 1 else float("nan")
    g_len3 = path_length_3d(g_e, g_n, g_u)
    f_len3 = path_length_3d(f_e, f_n, f_u)
    mp3 = np.isfinite(p_e) & np.isfinite(p_n) & np.isfinite(p_u)
    p_len3 = path_length_3d(p_e[mp3], p_n[mp3], p_u[mp3]) if np.count_nonzero(mp3) > 1 else float("nan")

    g_speed = gnss["speed_xyz_mps"].to_numpy(dtype=float)
    p_speed = interp(p_t, pdr["speed_mps"].to_numpy(dtype=float), gt)
    pdr_speed_metrics = {"zupt_mag_gnss_vs_gnss": speed_metrics(p_speed, g_speed)}

    windows = eval_windows(gt, g_e, g_n, p_e, p_n, [10, 20, 30])

    refs_t = refs["week"].to_numpy(dtype=float) * 604800.0 + refs["sow"].to_numpy(dtype=float) + GPS_EPOCH_UNIX - GPS_UTC_LEAP_SECONDS
    hd_ref = refs["heading_ref_deg"].to_numpy(dtype=float)
    hd_g = refs["heading_gnss_deg"].to_numpy(dtype=float)
    dt = np.diff(refs_t)
    d_h_err = np.abs(wrap180((hd_ref - hd_g)[1:] - (hd_ref - hd_g)[:-1]))
    drift = np.divide(d_h_err, dt, out=np.full_like(d_h_err, np.nan), where=dt > 0)
    heading_drift = {
        "count": int(np.isfinite(drift).sum()),
        "mean_abs_degps": float(np.nanmean(drift)),
        "p95_abs_degps": float(np.nanpercentile(drift[np.isfinite(drift)], 95)),
    }

    step_err = {}
    if "step_event" in pdr.columns and "step_length_ref_m" in refs.columns:
        m = pdr["step_event"].to_numpy(dtype=float) > 0.5
        idx = np.where(m)[0]
        if len(idx) > 1:
            se = pdr["pos_x_m"].to_numpy(dtype=float)
            sn = pdr["pos_y_m"].to_numpy(dtype=float)
            step_len = np.hypot(np.diff(se[idx]), np.diff(sn[idx]))
            st = pdr["t_sec_abs"].to_numpy(dtype=float)[idx][1:]
            ref_step = interp(refs_t, refs["step_length_ref_m"].to_numpy(dtype=float), st)
            abs_err = np.abs(step_len - ref_step)
            step_err = {
                "count": int(np.isfinite(abs_err).sum()),
                "mae_m": float(np.nanmean(abs_err)),
                "p95_m": float(np.nanpercentile(abs_err[np.isfinite(abs_err)], 95)),
            }
        else:
            step_err = {"count": 0, "mae_m": float("nan"), "p95_m": float("nan")}

    fig, ax = plt.subplots(figsize=(10, 9))
    ax.plot(g_e, g_n, label="GNSS", linewidth=2.0)
    ax.plot(f_e, f_n, label="Fusion", linewidth=1.5)
    ax.plot(p_e, p_n, label="PDR(zupt_mag_gnss)", linewidth=1.2)
    ax.set_xlabel("East [m]")
    ax.set_ylabel("North [m]")
    ax.set_title(f"Trajectory Compare {seg}")
    ax.grid(True, alpha=0.3)
    ax.axis("equal")
    ax.legend()
    out_png = out_dir / "traj_compare_gnss_fusion_pdr.png"
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)

    report = {
        "segment": seg,
        "inputs": {"raw_fix_csv": str(gnss_fix_csv), "fusion_traj_csv": str(fusion_csv), "pdr_traj_csv": str(pdr_csv)},
        "outputs": {"traj_compare_png": str(out_png)},
        "metrics_focus": {
            "fusion_vs_gnss_position_error_horizontal_m": stats(err_f_h),
            "fusion_vs_gnss_position_error_3d_m": stats(err_f_3d),
            "fusion_vs_gnss_path_len_2d_m": {"fusion": f_len2, "gnss": g_len2, "ratio": f_len2 / max(g_len2, 1e-9)},
            "fusion_vs_gnss_path_len_3d_m": {"fusion": f_len3, "gnss": g_len3, "ratio": f_len3 / max(g_len3, 1e-9)},
            "fusion_vs_gnss_relative_path_length_error_pct": float((f_len2 - g_len2) / max(g_len2, 1e-9) * 100.0),
            "pdr_zupt_mag_gnss_vs_gnss_horizontal_error_m": stats(err_p_h),
            "pdr_zupt_mag_gnss_relative_path_length_error_pct": float((p_len2 - g_len2) / max(g_len2, 1e-9) * 100.0),
            "pdr_speed_metrics_vs_gnss_fix": pdr_speed_metrics,
            "gnss_denied_windows": windows,
            "heading_drift_rate_degps": heading_drift,
            "step_length_error_vs_ref": step_err,
            "gnss_good_param_convergence": {
                "bgz_ref": "converged" if float(np.nanstd(refs["bgz_ref_radps"].to_numpy(dtype=float))) < 0.02 else "not_converged",
                "step_length_ref": "converged" if float(np.nanstd(refs["step_length_ref_m"].to_numpy(dtype=float))) < 0.15 else "not_converged",
                "yaw_offset_motion": "converged" if float(np.nanstd(refs["yaw_offset_motion_deg"].to_numpy(dtype=float))) < 20.0 else "not_converged",
            },
            "gnss_good_quality_ratio": float(np.mean(refs["quality_flag"].to_numpy(dtype=float) > 0.5)),
        },
    }
    out_json = out_dir / "summary_tech_metrics.json"
    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"summary_json": str(out_json), "plot_png": str(out_png)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
