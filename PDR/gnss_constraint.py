from __future__ import annotations

import numpy as np
import pandas as pd

from .config import GnssConstraintConfig


def wrap180(x: np.ndarray) -> np.ndarray:
    return (x + 180.0) % 360.0 - 180.0


def apply_gnss_velocity_constraints(
    t_sec_abs: np.ndarray,
    vel_enu: np.ndarray,
    gnss_fix_df: pd.DataFrame,
    labels: dict[str, np.ndarray] | None = None,
    cfg: GnssConstraintConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    c = cfg or GnssConstraintConfig()
    t = np.asarray(t_sec_abs, dtype=float)
    vel = np.asarray(vel_enu, dtype=float)
    g_t = gnss_fix_df["unix_sec"].to_numpy(dtype=float)
    g_e = gnss_fix_df["east_m"].to_numpy(dtype=float)
    g_n = gnss_fix_df["north_m"].to_numpy(dtype=float)
    g_acc = gnss_fix_df["acc_fix_m"].to_numpy(dtype=float)

    interval_scale = np.ones(len(g_t) - 1, dtype=float)
    interval_heading = np.full(len(g_t) - 1, np.nan, dtype=float)
    interval_alpha = np.zeros(len(g_t) - 1, dtype=float)
    interval_gap = np.zeros(len(g_t) - 1, dtype=int)
    rows: list[dict[str, float]] = []
    dt = np.diff(t, prepend=t[0])
    dt[dt <= 0] = 0.0

    for i in range(len(g_t) - 1):
        t0 = g_t[i]
        t1 = g_t[i + 1]
        m = (t >= t0) & (t < t1)
        if not np.any(m):
            rows.append({"idx": i, "t0": t0, "t1": t1, "gnss_dist_m": 0.0, "pdr_dist_m": 0.0, "scale": 1.0, "heading_alpha": 0.0, "gnss_gap_flag": 1, "gnss_good": 0})
            interval_gap[i] = 1
            continue
        gap = (t1 - t0) > c.max_interval_s
        interval_gap[i] = int(gap)
        gnss_dist = float(np.hypot(g_e[i + 1] - g_e[i], g_n[i + 1] - g_n[i]))
        pdr_dist = float(np.sum(np.linalg.norm(vel[m, :2], axis=1) * dt[m]))
        heading = float(np.arctan2(g_e[i + 1] - g_e[i], g_n[i + 1] - g_n[i])) if gnss_dist > 1e-6 else float("nan")
        good = int((g_acc[i] <= c.gnss_acc_good_thr_m) and (g_acc[i + 1] <= c.gnss_acc_good_thr_m))
        if (not gap) and good and pdr_dist > 1e-3:
            s = float(np.clip(gnss_dist / pdr_dist, c.min_scale, c.max_scale))
            acc_score = float(np.clip((c.gnss_acc_good_thr_m - max(g_acc[i], g_acc[i + 1])) / max(c.gnss_acc_good_thr_m - 5.0, 1e-9), 0.0, 1.0))
            a = c.heading_alpha_base + c.heading_alpha_gain * acc_score
        else:
            s = 1.0
            a = 0.0
        interval_scale[i] = s
        interval_heading[i] = heading
        interval_alpha[i] = a
        rows.append(
            {
                "idx": i,
                "t0": t0,
                "t1": t1,
                "gnss_dist_m": gnss_dist,
                "pdr_dist_m": pdr_dist,
                "gnss_heading_rad": heading,
                "scale": s,
                "heading_alpha": a,
                "gnss_gap_flag": int(gap),
                "gnss_good": good,
            }
        )

    g_idx = np.searchsorted(g_t, t, side="right") - 1
    g_idx = np.clip(g_idx, 0, len(interval_scale) - 1)
    scale = interval_scale[g_idx]
    alpha = interval_alpha[g_idx]
    vel_out = vel * scale[:, None]
    if labels is not None and "motion_context" in labels:
        alpha = np.where(np.asarray(labels["motion_context"], dtype=float) > 0.5, alpha, alpha * c.still_alpha_scale)

    for i in range(len(interval_heading)):
        hd = interval_heading[i]
        if not np.isfinite(hd):
            continue
        m = g_idx == i
        if not np.any(m):
            continue
        vh = np.linalg.norm(vel_out[m, :2], axis=1)
        valid = vh > c.heading_speed_thr_mps
        if not np.any(valid):
            continue
        u_ref = np.array([np.sin(hd), np.cos(hd)], dtype=float)
        vec = vel_out[m, :2].copy()
        u_est = np.zeros_like(vec)
        u_est[valid] = vec[valid] / vh[valid, None]
        a = float(alpha[m][0])
        u_mix = (1.0 - a) * u_est + a * u_ref[None, :]
        nrm = np.linalg.norm(u_mix, axis=1)
        safe = nrm > 1e-6
        u_mix[safe] = u_mix[safe] / nrm[safe, None]
        vec[valid] = u_mix[valid] * vh[valid, None]
        vel_out[m, 0] = vec[:, 0]
        vel_out[m, 1] = vec[:, 1]

    pos_out = np.zeros_like(vel_out)
    for i in range(1, len(t)):
        if dt[i] > 0:
            pos_out[i] = pos_out[i - 1] + vel_out[i] * dt[i]

    speed_out = np.linalg.norm(vel_out, axis=1)
    out_df = pd.DataFrame(
        {
            "t_sec_abs": t,
            "t_sec_rel": t - t[0],
            "scale_factor": scale,
            "vel_x_mps": vel_out[:, 0],
            "vel_y_mps": vel_out[:, 1],
            "vel_z_mps": vel_out[:, 2],
            "speed_mps": speed_out,
            "pos_x_m": pos_out[:, 0],
            "pos_y_m": pos_out[:, 1],
            "pos_z_m": pos_out[:, 2],
            "heading_fusion_alpha": alpha,
            "gnss_gap_flag": interval_gap[g_idx].astype(int),
        }
    )
    interval_df = pd.DataFrame(rows)

    gnss_hd = np.degrees(np.arctan2(np.diff(g_e, prepend=np.nan), np.diff(g_n, prepend=np.nan)))
    est_hd = np.degrees(np.arctan2(np.diff(np.interp(g_t, t, pos_out[:, 0], left=np.nan, right=np.nan), prepend=np.nan), np.diff(np.interp(g_t, t, pos_out[:, 1], left=np.nan, right=np.nan), prepend=np.nan)))
    hd_res = np.abs(wrap180(est_hd - gnss_hd))
    metrics = {
        "heading_mean_abs_deg": float(np.nanmean(hd_res)),
        "heading_p95_abs_deg": float(np.nanpercentile(hd_res[np.isfinite(hd_res)], 95)),
        "gap_interval_ratio": float(np.mean(interval_gap.astype(float))) if len(interval_gap) else 0.0,
    }
    return out_df, interval_df, metrics

