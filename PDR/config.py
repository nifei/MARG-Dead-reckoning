from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class StateLabelConfig:
    motion_speed_thr_mps: float = 0.16
    motion_acc_quantile: float = 0.55
    motion_gyro_quantile: float = 0.60
    motion_min_gyro_dps: float = 12.0
    motion_merge_s: float = 0.12
    motion_fill_s: float = 0.08
    zupt_acc_quantile: float = 0.35
    zupt_gyro_dps_thr: float = 20.0
    zupt_speed_thr_mps: float = 0.06
    zupt_merge_s: float = 0.04
    step_peak_distance_s: float = 0.28
    step_peak_prominence: float = 0.03


@dataclass(frozen=True)
class ZuptIntegrationConfig:
    moving_gain: float = 1.0
    still_gain: float = 0.15
    drift_acc_margin_mps2: float = 0.8
    drift_gyro_dps_thr: float = 25.0


@dataclass(frozen=True)
class GnssConstraintConfig:
    max_scale: float = 3.0
    min_scale: float = 0.3
    gnss_acc_good_thr_m: float = 25.0
    heading_speed_thr_mps: float = 0.12
    heading_alpha_base: float = 0.15
    heading_alpha_gain: float = 0.35
    still_alpha_scale: float = 0.35
    max_interval_s: float = 2.5


@dataclass(frozen=True)
class V21TargetConfig:
    motion_context_ratio_good_min: float = 0.98
    zupt_candidate_ratio_good_max: float = 0.05
    step_freq_spm_good_min: float = 55.0
    step_freq_spm_good_max: float = 120.0
    step_length_median_good_min: float = 0.65
    step_length_median_good_max: float = 0.85
    step_length_p95_good_max: float = 1.20
    step_length_cv_good_max: float = 0.35
    speed_mae_vs_gnss_good_max: float = 0.20
    speed_rmse_vs_gnss_good_max: float = 0.35
    heading_rate_p95_degps_good_max: float = 60.0

