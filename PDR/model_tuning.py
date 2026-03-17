from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import V21TargetConfig

GPS_EPOCH_UNIX = 315964800.0
GPS_UTC_LEAP_SECONDS = 18.0


@dataclass
class CoreMetrics:
    motion_context_ratio_good: float
    zupt_candidate_ratio_good: float
    step_freq_spm_good: float
    step_length_median_good: float
    step_length_p95_good: float
    step_length_cv_good: float
    speed_mae_vs_gnss_good: float
    speed_rmse_vs_gnss_good: float
    heading_rate_p95_degps_good: float

    def as_dict(self) -> dict[str, float]:
        return {
            "motion_context_ratio_good": self.motion_context_ratio_good,
            "zupt_candidate_ratio_good": self.zupt_candidate_ratio_good,
            "step_freq_spm_good": self.step_freq_spm_good,
            "step_length_median_good": self.step_length_median_good,
            "step_length_p95_good": self.step_length_p95_good,
            "step_length_cv_good": self.step_length_cv_good,
            "speed_mae_vs_gnss_good": self.speed_mae_vs_gnss_good,
            "speed_rmse_vs_gnss_good": self.speed_rmse_vs_gnss_good,
            "heading_rate_p95_degps_good": self.heading_rate_p95_degps_good,
        }


def interp_num(ts: np.ndarray, ys: np.ndarray, tq: np.ndarray) -> np.ndarray:
    keep = np.concatenate(([True], np.diff(ts) > 0))
    t = ts[keep]
    y = ys[keep]
    if len(t) < 2:
        return np.full_like(tq, np.nan, dtype=float)
    y = pd.Series(y).interpolate(limit_direction="both").to_numpy(dtype=float)
    return np.interp(tq, t, y, left=np.nan, right=np.nan)


def interp_bin(ts: np.ndarray, ys: np.ndarray, tq: np.ndarray) -> np.ndarray:
    return interp_num(ts, ys.astype(float), tq) > 0.5


def wrap180(x: np.ndarray) -> np.ndarray:
    return (x + 180.0) % 360.0 - 180.0


def compute_core_metrics(pdr: pd.DataFrame, refs: pd.DataFrame, gnss_fix: pd.DataFrame) -> CoreMetrics:
    t = pdr["t_sec_abs"].to_numpy(dtype=float)
    speed = pdr["speed_mps"].to_numpy(dtype=float)
    vx = pdr["vel_x_mps"].to_numpy(dtype=float)
    vy = pdr["vel_y_mps"].to_numpy(dtype=float)
    x = pdr["pos_x_m"].to_numpy(dtype=float)
    y = pdr["pos_y_m"].to_numpy(dtype=float)

    t_refs = refs["week"].to_numpy(dtype=float) * 604800.0 + refs["sow"].to_numpy(dtype=float) + GPS_EPOCH_UNIX - GPS_UTC_LEAP_SECONDS
    good_refs = refs["quality_flag"].to_numpy(dtype=float) > 0.5

    mc_good = interp_bin(t, pdr["motion_context"].to_numpy(dtype=float), t_refs)[good_refs]
    zupt_good = interp_bin(t, pdr["zupt_candidate"].to_numpy(dtype=float), t_refs)[good_refs]

    step_mask = pdr["step_event"].to_numpy(dtype=float) > 0.5
    step_t = t[step_mask]
    step_good = np.interp(step_t, t_refs, good_refs.astype(float), left=0.0, right=0.0) > 0.5
    dur_good_s = max(float(np.sum(good_refs)), 1e-9)
    step_freq_spm = float(np.count_nonzero(step_good) / (dur_good_s / 60.0))

    step_idx = np.where(step_mask)[0]
    lengths: list[float] = []
    for i0, i1 in zip(step_idx[:-1], step_idx[1:]):
        if np.interp(t[i1], t_refs, good_refs.astype(float), left=0.0, right=0.0) > 0.5:
            lengths.append(float(np.hypot(x[i1] - x[i0], y[i1] - y[i0])))
    if lengths:
        arr = np.asarray(lengths, dtype=float)
        step_median = float(np.nanmedian(arr))
        step_p95 = float(np.nanpercentile(arr, 95))
        step_cv = float(np.nanstd(arr) / max(np.nanmean(arr), 1e-9))
    else:
        step_median = float("nan")
        step_p95 = float("nan")
        step_cv = float("nan")

    gt = gnss_fix["unix_sec"].to_numpy(dtype=float)
    gs = gnss_fix["speed_xyz_mps"].to_numpy(dtype=float)
    ps = interp_num(t, speed, gt)
    good_gt = np.interp(gt, t_refs, good_refs.astype(float), left=0.0, right=0.0) > 0.5
    m = np.isfinite(ps) & np.isfinite(gs) & good_gt
    speed_mae = float(np.mean(np.abs(ps[m] - gs[m])))
    speed_rmse = float(np.sqrt(np.mean((ps[m] - gs[m]) ** 2)))

    hd = np.degrees(np.arctan2(interp_num(t, vx, t_refs), interp_num(t, vy, t_refs)))
    heading_p95 = float(np.nanpercentile(np.abs(wrap180(np.diff(hd)))[good_refs[1:]], 95))

    return CoreMetrics(
        motion_context_ratio_good=float(np.mean(mc_good)),
        zupt_candidate_ratio_good=float(np.mean(zupt_good)),
        step_freq_spm_good=step_freq_spm,
        step_length_median_good=step_median,
        step_length_p95_good=step_p95,
        step_length_cv_good=step_cv,
        speed_mae_vs_gnss_good=speed_mae,
        speed_rmse_vs_gnss_good=speed_rmse,
        heading_rate_p95_degps_good=heading_p95,
    )


def score(metrics: CoreMetrics, target: V21TargetConfig | None = None) -> tuple[float, bool]:
    t = target or V21TargetConfig()
    vals = metrics.as_dict()
    checks = {
        "motion_context_ratio_good": (t.motion_context_ratio_good_min, None),
        "zupt_candidate_ratio_good": (None, t.zupt_candidate_ratio_good_max),
        "step_freq_spm_good": (t.step_freq_spm_good_min, t.step_freq_spm_good_max),
        "step_length_median_good": (t.step_length_median_good_min, t.step_length_median_good_max),
        "step_length_p95_good": (None, t.step_length_p95_good_max),
        "step_length_cv_good": (None, t.step_length_cv_good_max),
        "speed_mae_vs_gnss_good": (None, t.speed_mae_vs_gnss_good_max),
        "speed_rmse_vs_gnss_good": (None, t.speed_rmse_vs_gnss_good_max),
        "heading_rate_p95_degps_good": (None, t.heading_rate_p95_degps_good_max),
    }
    penalty = 0.0
    passed = True
    for k, (lo, hi) in checks.items():
        v = vals[k]
        if lo is not None and v < lo:
            penalty += (lo - v) / max(abs(lo), 1e-9)
            passed = False
        if hi is not None and v > hi:
            penalty += (v - hi) / max(abs(hi), 1e-9)
            passed = False
    return penalty, passed


def synthesize_labels(pdr: pd.DataFrame, refs: pd.DataFrame, target_step_len: float, min_step_dt: float, max_step_dt: float, zupt_quantile: float) -> pd.DataFrame:
    out = pdr.copy()
    t = out["t_sec_abs"].to_numpy(dtype=float)
    dt = np.diff(t, prepend=t[0])
    dt[dt <= 0] = 0.0
    speed_h = np.hypot(out["vel_x_mps"].to_numpy(dtype=float), out["vel_y_mps"].to_numpy(dtype=float))

    t_refs = refs["week"].to_numpy(dtype=float) * 604800.0 + refs["sow"].to_numpy(dtype=float) + GPS_EPOCH_UNIX - GPS_UTC_LEAP_SECONDS
    good = np.interp(t, t_refs, (refs["quality_flag"].to_numpy(dtype=float) > 0.5).astype(float), left=0.0, right=0.0) > 0.5
    out.loc[good, "motion_context"] = (speed_h[good] > 0.05).astype(int)

    step_event = np.zeros(len(out), dtype=int)
    idx = np.where(good)[0]
    if len(idx) > 0:
        start = 0
        while start < len(idx):
            end = start + 1
            while end < len(idx) and idx[end] == idx[end - 1] + 1:
                end += 1
            seg = idx[start:end]
            last_t = t[seg[0]]
            acc_dist = 0.0
            for i in seg[1:]:
                acc_dist += speed_h[i] * dt[i]
                elapsed = t[i] - last_t
                if elapsed < min_step_dt:
                    continue
                if acc_dist >= target_step_len or elapsed >= max_step_dt:
                    step_event[i] = 1
                    acc_dist = 0.0
                    last_t = t[i]
            start = end
    out["step_event"] = step_event

    if np.any(good):
        thr = float(np.nanquantile(speed_h[good], zupt_quantile))
        out.loc[good, "zupt_candidate"] = (speed_h[good] <= thr).astype(int)
    return out

