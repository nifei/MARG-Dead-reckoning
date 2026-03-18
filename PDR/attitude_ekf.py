from __future__ import annotations

import numpy as np
from dataclasses import dataclass

G0 = 9.80665


@dataclass(frozen=True)
class AttitudeEkfConfig:
    sigma_gyro_radps: float = 0.02
    sigma_gyro_bias_rw_radps2: float = 0.001
    sigma_acc_dir: float = 0.03
    gate_gyro_dps_thr: float = 25.0
    gate_acc_abs_err_mps2: float = 1.2
    gate_acc_var_thr: float = 0.15
    gate_var_window_s: float = 0.25
    max_dtheta_deg_per_update: float = 2.0
    max_dbg_radps_per_update: float = 0.002
    max_gravity_residual_norm: float = 0.35


@dataclass(frozen=True)
class NavEkfConfig:
    sigma_gyro_radps: float = 0.02
    sigma_acc_mps2: float = 0.6
    sigma_gyro_bias_rw_radps2: float = 0.001
    sigma_acc_dir: float = 0.03
    gate_gyro_dps_thr: float = 25.0
    gate_acc_abs_err_mps2: float = 1.2
    gate_acc_var_thr: float = 0.15
    gate_var_window_s: float = 0.25
    max_dtheta_deg_per_update: float = 2.0
    max_dbg_radps_per_update: float = 0.002
    max_gravity_residual_norm: float = 0.35
    gnss_pos_sigma_min_m: float = 3.0
    gnss_vel_sigma_min_mps: float = 0.3
    gnss_time_match_tol_s: float = 0.03
    gnss_acc_good_thr_m: float = 25.0


def _normalize_quat(q: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(q)
    if n <= 0:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    return q / n


def _quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=float,
    )


def _quat_from_rotvec(rv: np.ndarray) -> np.ndarray:
    a = float(np.linalg.norm(rv))
    if a < 1e-12:
        return np.array([1.0, 0.5 * rv[0], 0.5 * rv[1], 0.5 * rv[2]], dtype=float)
    axis = rv / a
    h = 0.5 * a
    return np.array([np.cos(h), *(np.sin(h) * axis)], dtype=float)


def _quat_from_two_vectors(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a_n = a / max(np.linalg.norm(a), 1e-12)
    b_n = b / max(np.linalg.norm(b), 1e-12)
    c = np.cross(a_n, b_n)
    d = float(np.dot(a_n, b_n))
    if d < -0.999999:
        axis = np.array([1.0, 0.0, 0.0], dtype=float)
        if abs(a_n[0]) > 0.9:
            axis = np.array([0.0, 1.0, 0.0], dtype=float)
        axis = axis - a_n * np.dot(a_n, axis)
        axis = axis / max(np.linalg.norm(axis), 1e-12)
        return _normalize_quat(np.array([0.0, axis[0], axis[1], axis[2]], dtype=float))
    s = np.sqrt((1.0 + d) * 2.0)
    q = np.array([0.5 * s, c[0] / s, c[1] / s, c[2] / s], dtype=float)
    return _normalize_quat(q)


def _quat_to_rotmat(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    r00 = 2.0 * (w * w + x * x) - 1.0
    r01 = 2.0 * (x * y - w * z)
    r02 = 2.0 * (x * z + w * y)
    r10 = 2.0 * (x * y + w * z)
    r11 = 2.0 * (w * w + y * y) - 1.0
    r12 = 2.0 * (y * z - w * x)
    r20 = 2.0 * (x * z - w * y)
    r21 = 2.0 * (y * z + w * x)
    r22 = 2.0 * (w * w + z * z) - 1.0
    return np.array([[r00, r01, r02], [r10, r11, r12], [r20, r21, r22]], dtype=float)


def _quat_to_euler_deg(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    t0 = 2.0 * (w * x + y * z)
    t1 = 1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(t0, t1)

    t2 = 2.0 * (w * y - z * x)
    t2 = np.clip(t2, -1.0, 1.0)
    pitch = np.arcsin(t2)

    t3 = 2.0 * (w * z + x * y)
    t4 = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(t3, t4)
    return np.degrees(np.array([roll, pitch, yaw], dtype=float))


def _skew(v: np.ndarray) -> np.ndarray:
    return np.array(
        [[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]],
        dtype=float,
    )


def _rolling_var(x: np.ndarray, win: int) -> np.ndarray:
    if win <= 1 or len(x) == 0:
        return np.zeros_like(x, dtype=float)
    out = np.zeros_like(x, dtype=float)
    c1 = np.cumsum(np.r_[0.0, x])
    c2 = np.cumsum(np.r_[0.0, x * x])
    for i in range(len(x)):
        j0 = max(0, i - win + 1)
        n = i - j0 + 1
        s = c1[i + 1] - c1[j0]
        s2 = c2[i + 1] - c2[j0]
        m = s / max(n, 1)
        out[i] = max(0.0, s2 / max(n, 1) - m * m)
    return out


def estimate_attitude_gyro_bias(
    t_sec_abs: np.ndarray,
    gyro_radps: np.ndarray,
    accel_mps2: np.ndarray,
    cfg: AttitudeEkfConfig | None = None,
) -> dict[str, np.ndarray]:
    c = cfg or AttitudeEkfConfig()
    n = len(t_sec_abs)
    if n == 0:
        return {
            "q_wxyz": np.empty((0, 4), dtype=float),
            "rpy_deg": np.empty((0, 3), dtype=float),
            "gyro_bias_radps": np.empty((0, 3), dtype=float),
            "cov": np.empty((0, 6, 6), dtype=float),
            "gate_gravity": np.empty((0,), dtype=bool),
        }

    dt = np.diff(t_sec_abs, prepend=t_sec_abs[0])
    dt[dt <= 0] = np.median(dt[dt > 0]) if np.any(dt > 0) else 0.005

    q = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    if n > 10:
        acc0 = np.nanmedian(accel_mps2[: min(n, 300)], axis=0)
        q = _quat_from_two_vectors(acc0, np.array([0.0, 0.0, G0], dtype=float))
    bg = np.zeros(3, dtype=float)
    P = np.diag([0.05, 0.05, 0.05, 0.02, 0.02, 0.02]).astype(float)

    q_arr = np.zeros((n, 4), dtype=float)
    rpy = np.zeros((n, 3), dtype=float)
    bg_arr = np.zeros((n, 3), dtype=float)
    cov = np.zeros((n, 6, 6), dtype=float)

    gyro_dps = np.linalg.norm(gyro_radps, axis=1) * (180.0 / np.pi)
    acc_norm = np.linalg.norm(accel_mps2, axis=1)
    fs = 1.0 / max(np.median(dt[dt > 0]) if np.any(dt > 0) else 0.005, 1e-6)
    win = max(3, int(round(c.gate_var_window_s * fs)))
    acc_var = _rolling_var(acc_norm, win)
    gate = (
        (gyro_dps < c.gate_gyro_dps_thr)
        & (np.abs(acc_norm - G0) < c.gate_acc_abs_err_mps2)
        & (acc_var < c.gate_acc_var_thr)
    )

    I6 = np.eye(6, dtype=float)
    I3 = np.eye(3, dtype=float)
    g_w = np.array([0.0, 0.0, 1.0], dtype=float)

    for i in range(n):
        dti = float(max(dt[i], 1e-6))

        w = gyro_radps[i] - bg
        dq = _quat_from_rotvec(w * dti)
        q = _normalize_quat(_quat_mul(q, dq))

        F = np.eye(6, dtype=float)
        F[0:3, 0:3] = I3 - _skew(w) * dti
        F[0:3, 3:6] = -I3 * dti

        Q = np.zeros((6, 6), dtype=float)
        qg = (c.sigma_gyro_radps * dti) ** 2
        qbg = (c.sigma_gyro_bias_rw_radps2 * dti) ** 2
        Q[0:3, 0:3] = I3 * qg
        Q[3:6, 3:6] = I3 * qbg
        P = F @ P @ F.T + Q

        if gate[i]:
            Rb2w = _quat_to_rotmat(q)
            h = Rb2w.T @ g_w
            z = accel_mps2[i] / max(np.linalg.norm(accel_mps2[i]), 1e-9)
            r = z - h

            H = np.zeros((3, 6), dtype=float)
            H[:, 0:3] = _skew(h)
            Rm = I3 * (c.sigma_acc_dir ** 2)

            S = H @ P @ H.T + Rm
            K = P @ H.T @ np.linalg.inv(S)
            dx = K @ r

            if np.linalg.norm(r) <= c.max_gravity_residual_norm:
                dtheta = dx[0:3]
                dbg = dx[3:6]
                max_dtheta = np.deg2rad(c.max_dtheta_deg_per_update)
                nd = float(np.linalg.norm(dtheta))
                if nd > max_dtheta and nd > 0:
                    dtheta = dtheta * (max_dtheta / nd)
                dbg = np.clip(dbg, -c.max_dbg_radps_per_update, c.max_dbg_radps_per_update)
                q = _normalize_quat(_quat_mul(q, _quat_from_rotvec(dtheta)))
                bg = bg + dbg

                IKH = I6 - K @ H
                P = IKH @ P @ IKH.T + K @ Rm @ K.T

        q_arr[i] = q
        rpy[i] = _quat_to_euler_deg(q)
        bg_arr[i] = bg
        cov[i] = P

    return {
        "q_wxyz": q_arr,
        "rpy_deg": rpy,
        "gyro_bias_radps": bg_arr,
        "cov": cov,
        "gate_gravity": gate.astype(bool),
    }


def estimate_nav_state_with_gnss(
    t_sec_abs: np.ndarray,
    gyro_radps: np.ndarray,
    accel_mps2: np.ndarray,
    gnss_t_sec: np.ndarray,
    gnss_pos_enu_m: np.ndarray,
    gnss_vel_enu_mps: np.ndarray,
    gnss_acc_m: np.ndarray,
    cfg: NavEkfConfig | None = None,
) -> dict[str, np.ndarray]:
    c = cfg or NavEkfConfig()
    n = len(t_sec_abs)
    if n == 0:
        return {
            "q_wxyz": np.empty((0, 4), dtype=float),
            "rpy_deg": np.empty((0, 3), dtype=float),
            "gyro_bias_radps": np.empty((0, 3), dtype=float),
            "vel_enu_mps": np.empty((0, 3), dtype=float),
            "pos_enu_m": np.empty((0, 3), dtype=float),
            "cov": np.empty((0, 12, 12), dtype=float),
            "gate_gravity": np.empty((0,), dtype=bool),
            "gnss_update_flag": np.empty((0,), dtype=bool),
            "acc_world_linear_mps2": np.empty((0, 3), dtype=float),
        }

    dt = np.diff(t_sec_abs, prepend=t_sec_abs[0])
    dt[dt <= 0] = np.median(dt[dt > 0]) if np.any(dt > 0) else 0.005

    q = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    if n > 10:
        acc0 = np.nanmedian(accel_mps2[: min(n, 300)], axis=0)
        q = _quat_from_two_vectors(acc0, np.array([0.0, 0.0, G0], dtype=float))
    bg = np.zeros(3, dtype=float)
    v = np.zeros(3, dtype=float)
    p = np.zeros(3, dtype=float)
    P = np.diag([0.05, 0.05, 0.05, 2.0, 2.0, 2.0, 5.0, 5.0, 5.0, 0.02, 0.02, 0.02]).astype(float)

    q_arr = np.zeros((n, 4), dtype=float)
    rpy = np.zeros((n, 3), dtype=float)
    bg_arr = np.zeros((n, 3), dtype=float)
    v_arr = np.zeros((n, 3), dtype=float)
    p_arr = np.zeros((n, 3), dtype=float)
    cov = np.zeros((n, 12, 12), dtype=float)
    a_lin = np.zeros((n, 3), dtype=float)
    gnss_update = np.zeros(n, dtype=bool)

    gyro_dps = np.linalg.norm(gyro_radps, axis=1) * (180.0 / np.pi)
    acc_norm = np.linalg.norm(accel_mps2, axis=1)
    fs = 1.0 / max(np.median(dt[dt > 0]) if np.any(dt > 0) else 0.005, 1e-6)
    win = max(3, int(round(c.gate_var_window_s * fs)))
    acc_var = _rolling_var(acc_norm, win)
    gravity_gate = (
        (gyro_dps < c.gate_gyro_dps_thr)
        & (np.abs(acc_norm - G0) < c.gate_acc_abs_err_mps2)
        & (acc_var < c.gate_acc_var_thr)
    )

    I3 = np.eye(3, dtype=float)
    I12 = np.eye(12, dtype=float)
    g_w = np.array([0.0, 0.0, 9.80665], dtype=float)

    # Build sparse GNSS observations at IMU indices (avoid interpolating through large gaps).
    obs_pos = np.full((n, 3), np.nan, dtype=float)
    obs_vel = np.full((n, 3), np.nan, dtype=float)
    obs_acc = np.full(n, np.nan, dtype=float)
    for i in range(len(gnss_t_sec)):
        gt = float(gnss_t_sec[i])
        idx = int(np.searchsorted(t_sec_abs, gt))
        cand = []
        if idx < n:
            cand.append(idx)
        if idx > 0:
            cand.append(idx - 1)
        if not cand:
            continue
        j = min(cand, key=lambda k: abs(t_sec_abs[k] - gt))
        if abs(float(t_sec_abs[j]) - gt) > c.gnss_time_match_tol_s:
            continue
        obs_pos[j] = gnss_pos_enu_m[i]
        obs_vel[j] = gnss_vel_enu_mps[i]
        obs_acc[j] = gnss_acc_m[i]

    for i in range(n):
        dti = float(max(dt[i], 1e-6))
        w = gyro_radps[i] - bg
        dq = _quat_from_rotvec(w * dti)
        q = _normalize_quat(_quat_mul(q, dq))
        Rb2w = _quat_to_rotmat(q)
        a_world = Rb2w @ accel_mps2[i] - g_w
        a_lin[i] = a_world
        v = v + a_world * dti
        p = p + v * dti

        F = np.zeros((12, 12), dtype=float)
        F[0:3, 0:3] = -_skew(w)
        F[0:3, 9:12] = -I3
        F[3:6, 0:3] = -Rb2w @ _skew(accel_mps2[i])
        F[6:9, 3:6] = I3
        Fd = I12 + F * dti

        Q = np.zeros((12, 12), dtype=float)
        qg = (c.sigma_gyro_radps * dti) ** 2
        qa = (c.sigma_acc_mps2 * dti) ** 2
        qbg = (c.sigma_gyro_bias_rw_radps2 * dti) ** 2
        Q[0:3, 0:3] = I3 * qg
        Q[3:6, 3:6] = I3 * qa
        Q[6:9, 6:9] = I3 * (qa * dti * dti)
        Q[9:12, 9:12] = I3 * qbg
        P = Fd @ P @ Fd.T + Q

        # gravity direction observation update (tilt correction only)
        if gravity_gate[i]:
            h = Rb2w.T @ np.array([0.0, 0.0, 1.0], dtype=float)
            z = accel_mps2[i] / max(np.linalg.norm(accel_mps2[i]), 1e-9)
            r = z - h
            if np.linalg.norm(r) <= c.max_gravity_residual_norm:
                H = np.zeros((3, 12), dtype=float)
                H[:, 0:3] = _skew(h)
                Rm = I3 * (c.sigma_acc_dir ** 2)
                S = H @ P @ H.T + Rm
                K = P @ H.T @ np.linalg.inv(S)
                dx = K @ r
                dtheta = dx[0:3]
                dbg = dx[9:12]
                max_dtheta = np.deg2rad(c.max_dtheta_deg_per_update)
                nd = float(np.linalg.norm(dtheta))
                if nd > max_dtheta and nd > 0:
                    dtheta = dtheta * (max_dtheta / nd)
                dbg = np.clip(dbg, -c.max_dbg_radps_per_update, c.max_dbg_radps_per_update)
                q = _normalize_quat(_quat_mul(q, _quat_from_rotvec(dtheta)))
                bg = bg + dbg
                # inject v/p
                v = v + dx[3:6]
                p = p + dx[6:9]
                IKH = I12 - K @ H
                P = IKH @ P @ IKH.T + K @ Rm @ K.T

        # GNSS strong update when fix is good
        if np.isfinite(obs_acc[i]) and obs_acc[i] <= c.gnss_acc_good_thr_m:
            z = np.concatenate([obs_vel[i], obs_pos[i]])
            h = np.concatenate([v, p])
            r = z - h
            H = np.zeros((6, 12), dtype=float)
            H[0:3, 3:6] = I3
            H[3:6, 6:9] = I3
            pos_sigma = max(c.gnss_pos_sigma_min_m, float(obs_acc[i]))
            vel_sigma = max(c.gnss_vel_sigma_min_mps, 0.06 * float(obs_acc[i]))
            Rv = I3 * (vel_sigma ** 2)
            Rp = I3 * (pos_sigma ** 2)
            Rm = np.block([[Rv, np.zeros((3, 3))], [np.zeros((3, 3)), Rp]])
            S = H @ P @ H.T + Rm
            K = P @ H.T @ np.linalg.inv(S)
            dx = K @ r
            dtheta = dx[0:3]
            dbg = dx[9:12]
            q = _normalize_quat(_quat_mul(q, _quat_from_rotvec(dtheta)))
            v = v + dx[3:6]
            p = p + dx[6:9]
            bg = bg + np.clip(dbg, -c.max_dbg_radps_per_update, c.max_dbg_radps_per_update)
            IKH = I12 - K @ H
            P = IKH @ P @ IKH.T + K @ Rm @ K.T
            gnss_update[i] = True

        q_arr[i] = q
        rpy[i] = _quat_to_euler_deg(q)
        bg_arr[i] = bg
        v_arr[i] = v
        p_arr[i] = p
        cov[i] = P

    return {
        "q_wxyz": q_arr,
        "rpy_deg": rpy,
        "gyro_bias_radps": bg_arr,
        "vel_enu_mps": v_arr,
        "pos_enu_m": p_arr,
        "cov": cov,
        "gate_gravity": gravity_gate.astype(bool),
        "gnss_update_flag": gnss_update.astype(bool),
        "acc_world_linear_mps2": a_lin,
    }
