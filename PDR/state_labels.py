from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from .config import StateLabelConfig, ZuptIntegrationConfig


def merge_short_runs(mask: np.ndarray, min_len: int) -> np.ndarray:
    out = np.asarray(mask, dtype=bool).copy()
    if min_len <= 1 or len(out) == 0:
        return out
    start = 0
    current = bool(out[0])
    for idx in range(1, len(out) + 1):
        ended = idx == len(out) or bool(out[idx]) != current
        if not ended:
            continue
        if idx - start < min_len:
            left = bool(out[start - 1]) if start > 0 else None
            right = bool(out[idx]) if idx < len(out) else None
            replacement = left if left is not None else right
            if replacement is not None:
                out[start:idx] = replacement
        if idx < len(out):
            start = idx
            current = bool(out[idx])
    return out


def build_motion_context_from_signals(
    raw_is_moving: np.ndarray,
    speed_h: np.ndarray,
    acc_norm: np.ndarray,
    gyro_dps: np.ndarray,
    freq_hz: float,
    cfg: StateLabelConfig | None = None,
) -> np.ndarray:
    c = cfg or StateLabelConfig()
    sample_rate = max(1.0, float(freq_hz))
    acc_move_thr = float(np.nanpercentile(acc_norm, c.motion_acc_quantile * 100.0))
    gyro_move_thr = float(np.nanpercentile(gyro_dps, c.motion_gyro_quantile * 100.0))
    mc_seed = (
        np.asarray(raw_is_moving, dtype=bool)
        | (speed_h > c.motion_speed_thr_mps)
        | (acc_norm > acc_move_thr)
        | (gyro_dps > max(c.motion_min_gyro_dps, gyro_move_thr))
    )
    motion_context = merge_short_runs(mc_seed, max(1, int(round(c.motion_merge_s * sample_rate))))
    motion_context = ~merge_short_runs(~motion_context, max(1, int(round(c.motion_fill_s * sample_rate))))
    return motion_context.astype(bool)


def derive_motion_labels(
    raw_is_moving: np.ndarray,
    vel_ned: np.ndarray,
    acc_earth: np.ndarray,
    gyr_radps: np.ndarray,
    freq_hz: float,
    cfg: StateLabelConfig | None = None,
) -> dict[str, np.ndarray]:
    c = cfg or StateLabelConfig()
    sample_rate = max(1.0, float(freq_hz))
    speed_h = np.linalg.norm(vel_ned[:, :2], axis=1)
    acc_norm = np.linalg.norm(acc_earth, axis=1)
    gyro_dps = np.linalg.norm(gyr_radps, axis=1) * (180.0 / np.pi)
    motion_context = build_motion_context_from_signals(raw_is_moving, speed_h, acc_norm, gyro_dps, sample_rate, c)

    acc_quiet_thr = float(np.nanpercentile(acc_norm, c.zupt_acc_quantile * 100.0))
    zupt_candidate = ((acc_norm <= acc_quiet_thr) & (gyro_dps <= c.zupt_gyro_dps_thr)) | (speed_h <= c.zupt_speed_thr_mps)
    zupt_candidate = merge_short_runs(zupt_candidate, max(1, int(round(c.zupt_merge_s * sample_rate))))

    min_dist = max(1, int(round(c.step_peak_distance_s * sample_rate)))
    peaks, _ = find_peaks(speed_h, distance=min_dist, prominence=c.step_peak_prominence)
    step_event = np.zeros(len(speed_h), dtype=bool)
    if len(peaks) > 0:
        step_event[peaks] = True
    else:
        rises = np.where((~motion_context[:-1]) & motion_context[1:])[0] + 1
        step_event[rises] = True
    step_event &= motion_context

    return {
        "motion_context": motion_context.astype(bool),
        "step_event": step_event.astype(bool),
        "zupt_candidate": zupt_candidate.astype(bool),
    }


def integrate_velocity_with_constraints(
    acc_earth: np.ndarray,
    dt: np.ndarray,
    motion_context: np.ndarray,
    g_est: float,
    gyr_radps: np.ndarray,
    cfg: ZuptIntegrationConfig | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    c = cfg or ZuptIntegrationConfig()
    n = len(acc_earth)
    velocity = np.zeros((n, 3), dtype=float)
    for idx in range(1, n):
        gain = c.moving_gain if motion_context[idx] else c.still_gain
        velocity[idx] = velocity[idx - 1] + gain * acc_earth[idx] * dt[idx]

    acc_norm = np.linalg.norm(acc_earth, axis=1)
    gyro_norm_dps = np.linalg.norm(gyr_radps, axis=1) * (180.0 / np.pi)
    zupt_anchor = (~motion_context) & (acc_norm < (g_est + c.drift_acc_margin_mps2)) & (gyro_norm_dps < c.drift_gyro_dps_thr)
    if int(np.sum(zupt_anchor)) < 2:
        zupt_anchor = ~motion_context
    if int(np.sum(zupt_anchor)) < 2:
        zupt_anchor = np.zeros(n, dtype=bool)
        zupt_anchor[[0, n - 1]] = True

    drift = np.where(zupt_anchor[:, None], velocity, np.nan)
    drift = pd.DataFrame(drift).interpolate(limit_direction="both").fillna(0.0).to_numpy(dtype=float)
    velocity = velocity - drift
    return velocity, zupt_anchor.astype(bool)

