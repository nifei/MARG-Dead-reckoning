#!/usr/bin/env python3
from __future__ import annotations

import argparse
import bisect
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import imufusion
import numpy as np
import pandas as pd
from ahrs.filters import Madgwick

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

from PDR.config import ZuptIntegrationConfig
from PDR.state_labels import derive_motion_labels, integrate_velocity_with_constraints

GPS_EPOCH_UNIX = 315964800
GPS_UTC_LEAP_SECONDS = 18
G0 = 9.80665
EARTH_RADIUS_M = 6378137.0


@dataclass
class Vec3Row:
    t_ms: int
    x: float
    y: float
    z: float


def gps_week_sow_to_unix_ms(week: float, sow: float) -> int:
    gps_sec = week * 604800.0 + sow
    unix_sec = gps_sec + GPS_EPOCH_UNIX - GPS_UTC_LEAP_SECONDS
    return int(round(unix_sec * 1000.0))


def read_satellite_time_window_ms(segment_sat_csv: Path) -> tuple[int, int]:
    weeks: list[float] = []
    sows: list[float] = []
    with segment_sat_csv.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            weeks.append(float(row["week"]))
            sows.append(float(row["seconds of week [s]"]))
    if not weeks:
        raise ValueError(f"empty satellite_lla: {segment_sat_csv}")
    start_ms = gps_week_sow_to_unix_ms(min(weeks), min(sows))
    end_ms = gps_week_sow_to_unix_ms(max(weeks), max(sows))
    return start_ms, end_ms


def parse_uimu_window(
    uimu_file: Path, start_ms: int, end_ms: int
) -> tuple[list[Vec3Row], list[Vec3Row], list[Vec3Row]]:
    acc_rows: list[Vec3Row] = []
    gyr_rows: list[Vec3Row] = []
    mag_rows: list[Vec3Row] = []
    with uimu_file.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if (
                not s
                or s.startswith("*")
                or s.startswith("Sensor ")
                or s.startswith("UNIX ")
            ):
                continue
            cols = [x.strip() for x in s.split(",")]
            if len(cols) < 6:
                continue
            try:
                t = int(float(cols[0]))
                flag = int(cols[2])
                x = float(cols[3])
                y = float(cols[4])
                z = float(cols[5])
            except ValueError:
                continue
            if t < start_ms or t > end_ms:
                continue
            row = Vec3Row(t, x, y, z)
            if flag == 1:
                acc_rows.append(row)
            elif flag == 2:
                gyr_rows.append(row)
            elif flag == 3:
                mag_rows.append(row)
    acc_rows.sort(key=lambda r: r.t_ms)
    gyr_rows.sort(key=lambda r: r.t_ms)
    mag_rows.sort(key=lambda r: r.t_ms)
    return acc_rows, gyr_rows, mag_rows


def nearest_by_time(rows: list[Vec3Row], ts: list[int], t_ms: int) -> Vec3Row | None:
    if not rows:
        return None
    i = bisect.bisect_left(ts, t_ms)
    cands: list[Vec3Row] = []
    if i < len(rows):
        cands.append(rows[i])
    if i > 0:
        cands.append(rows[i - 1])
    if not cands:
        return None
    return min(cands, key=lambda r: abs(r.t_ms - t_ms))


def pair_acc_gyro_mag(
    acc_rows: list[Vec3Row],
    gyr_rows: list[Vec3Row],
    mag_rows: list[Vec3Row],
    max_ag_gap_ms: int = 5,
    max_am_gap_ms: int = 50,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not acc_rows or not gyr_rows or not mag_rows:
        return (
            np.empty((0,), dtype=float),
            np.empty((0, 3), dtype=float),
            np.empty((0, 3), dtype=float),
            np.empty((0, 3), dtype=float),
        )

    gyr_ts = [r.t_ms for r in gyr_rows]
    mag_ts = [r.t_ms for r in mag_rows]
    out_t: list[float] = []
    out_acc: list[tuple[float, float, float]] = []
    out_gyr: list[tuple[float, float, float]] = []
    out_mag: list[tuple[float, float, float]] = []

    for acc in acc_rows:
        j = bisect.bisect_left(gyr_ts, acc.t_ms)
        cands: list[Vec3Row] = []
        if j < len(gyr_rows):
            cands.append(gyr_rows[j])
        if j > 0:
            cands.append(gyr_rows[j - 1])
        if not cands:
            continue
        gyr = min(cands, key=lambda r: abs(r.t_ms - acc.t_ms))
        if abs(gyr.t_ms - acc.t_ms) > max_ag_gap_ms:
            continue

        t_pair_ms = int(round(0.5 * (acc.t_ms + gyr.t_ms)))
        mag = nearest_by_time(mag_rows, mag_ts, t_pair_ms)
        if mag is None or abs(mag.t_ms - t_pair_ms) > max_am_gap_ms:
            continue

        out_t.append(t_pair_ms / 1000.0)
        out_acc.append((acc.x, acc.y, acc.z))
        out_gyr.append((gyr.x, gyr.y, gyr.z))
        out_mag.append((mag.x, mag.y, mag.z))

    if not out_t:
        return (
            np.empty((0,), dtype=float),
            np.empty((0, 3), dtype=float),
            np.empty((0, 3), dtype=float),
            np.empty((0, 3), dtype=float),
        )

    t_arr = np.asarray(out_t, dtype=float)
    acc_arr = np.asarray(out_acc, dtype=float)
    gyr_arr = np.asarray(out_gyr, dtype=float)
    mag_arr = np.asarray(out_mag, dtype=float)
    if len(t_arr) > 1:
        keep = np.concatenate(([True], np.diff(t_arr) > 0))
        t_arr = t_arr[keep]
        acc_arr = acc_arr[keep]
        gyr_arr = gyr_arr[keep]
        mag_arr = mag_arr[keep]
    return t_arr, acc_arr, gyr_arr, mag_arr


def estimate_frequency_hz(t_sec: np.ndarray) -> float:
    if len(t_sec) < 2:
        return 200.0
    dt = np.diff(t_sec)
    dt = dt[dt > 0]
    if len(dt) == 0:
        return 200.0
    return float(1.0 / np.median(dt))


def resolve_calib_csv(calib_csv_arg: str | None, repo_root: Path) -> Path | None:
    if calib_csv_arg:
        p = Path(calib_csv_arg).resolve()
        return p if p.exists() else None
    candidates = [
        repo_root / "data" / "calib" / "calib.csv",
        repo_root / "PDR" / "MARG-Dead-reckoning" / "calib.csv",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def candidate_uimu_files(repo_root: Path) -> list[Path]:
    out: list[Path] = []
    scan_dirs = [
        repo_root / "data" / "GNSS-IMU-Logger" / "UIMU_Log",
        repo_root / "data" / "0317-GNSS-IMU-Logger" / "UIMU_Log",
        repo_root / "data" / "calib" / "UIMU_Log",
    ]
    for d in scan_dirs:
        if not d.exists():
            continue
        for p in sorted(d.glob("*.txt")):
            if p not in out:
                out.append(p)
    return out


def select_uimu_file(
    repo_root: Path,
    explicit_file: Path | None,
    start_ms: int,
    end_ms: int,
) -> tuple[Path, list[ImuRow], list[ImuRow], list[ImuRow], dict[str, object]]:
    if explicit_file is not None:
        acc_rows, gyr_rows, mag_rows = parse_uimu_window(explicit_file, start_ms, end_ms)
        return explicit_file, acc_rows, gyr_rows, mag_rows, {"mode": "explicit", "tested": 1}

    best: tuple[int, Path, list[ImuRow], list[ImuRow], list[ImuRow]] | None = None
    tested = 0
    for f in candidate_uimu_files(repo_root):
        tested += 1
        acc_rows, gyr_rows, mag_rows = parse_uimu_window(f, start_ms, end_ms)
        paired_n = min(len(acc_rows), len(gyr_rows), len(mag_rows))
        if best is None or paired_n > best[0]:
            best = (paired_n, f, acc_rows, gyr_rows, mag_rows)
    if best is None:
        raise FileNotFoundError("no UIMU_Log candidates found")
    return best[1], best[2], best[3], best[4], {
        "mode": "auto_best_overlap",
        "tested": tested,
        "best_paired_rows": int(best[0]),
    }


def load_calib(calib_csv: Path | None) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    zero = np.zeros(3, dtype=float)
    eye = np.eye(3, dtype=float)
    if calib_csv is None:
        return (
            {"acc_b": zero, "gyro_b": zero, "mag_b": zero, "mag_A": eye},
            {"loaded": False, "calib_csv": "", "fallback_identity": True},
        )
    try:
        calib = pd.read_csv(calib_csv, sep="\t", header=[0, 1], index_col=[0, 1]).T
        acc_b = calib.loc["b", "b"]["accel"].to_numpy(dtype=float)
        gyro_b = calib.loc["b", "b"]["gyro"].to_numpy(dtype=float)
        mag_b = calib.loc["b", "b"]["mag"].to_numpy(dtype=float)
        mag_A = calib.loc["A"]["mag"].to_numpy(dtype=float)
        return (
            {"acc_b": acc_b, "gyro_b": gyro_b, "mag_b": mag_b, "mag_A": mag_A},
            {"loaded": True, "calib_csv": str(calib_csv), "fallback_identity": False},
        )
    except Exception:
        return (
            {"acc_b": zero, "gyro_b": zero, "mag_b": zero, "mag_A": eye},
            {"loaded": False, "calib_csv": str(calib_csv), "fallback_identity": True},
        )


def apply_calib(
    acc_mps2: np.ndarray,
    gyr_radps: np.ndarray,
    mag_uT: np.ndarray,
    calib: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Match original PDR/pdr.py behavior: subtract per-sensor bias and apply mag soft-iron matrix.
    acc_corr = acc_mps2 - calib["acc_b"]
    gyr_corr = gyr_radps - calib["gyro_b"]
    mag_debiased = mag_uT - calib["mag_b"]
    mag_corr = (calib["mag_A"] @ mag_debiased.T).T
    for i in range(10):
        print("accel:", acc_mps2[300 + i], '->', acc_corr[300 + i])
        print("gyro:", gyr_radps[300 + i], '->', gyr_corr[300 + i])
        print("mag:", mag_uT[300 + i], '->', mag_corr[300 + i])
    return acc_corr, gyr_corr, mag_corr


def quat_to_rot_mat(q: np.ndarray) -> np.ndarray:
    r00 = 2 * (q[0] * q[0] + q[1] * q[1]) - 1
    r01 = 2 * (q[1] * q[2] - q[0] * q[3])
    r02 = 2 * (q[1] * q[3] + q[0] * q[2])
    r10 = 2 * (q[1] * q[2] + q[0] * q[3])
    r11 = 2 * (q[0] * q[0] + q[2] * q[2]) - 1
    r12 = 2 * (q[2] * q[3] - q[0] * q[1])
    r20 = 2 * (q[1] * q[3] - q[0] * q[2])
    r21 = 2 * (q[2] * q[3] + q[0] * q[1])
    r22 = 2 * (q[0] * q[0] + q[3] * q[3]) - 1
    return np.array([[r00, r01, r02], [r10, r11, r12], [r20, r21, r22]], dtype=float)


def integrate_kinematics(t_sec: np.ndarray, acc_linear: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = len(t_sec)
    vel = np.zeros((n, 3), dtype=float)
    pos = np.zeros((n, 3), dtype=float)
    for i in range(1, n):
        dt = t_sec[i] - t_sec[i - 1]
        if dt <= 0:
            continue
        vel[i] = vel[i - 1] + acc_linear[i] * dt
        pos[i] = pos[i - 1] + vel[i] * dt
    return vel, pos


def run_standard_pdr(t_sec: np.ndarray, acc_mps2: np.ndarray, freq_hz: float) -> dict[str, np.ndarray]:
    # Use startup stationary window as acceleration bias baseline.
    n_bias = max(1, min(len(acc_mps2), int(freq_hz * 2.0)))
    acc_bias = np.mean(acc_mps2[:n_bias], axis=0)
    acc_linear = acc_mps2 - acc_bias
    vel, pos = integrate_kinematics(t_sec, acc_linear)
    speed = np.linalg.norm(vel, axis=1)
    return {"acc_linear": acc_linear, "vel": vel, "pos": pos, "speed": speed}


def run_ahrs_pdr(
    t_sec: np.ndarray, acc_mps2: np.ndarray, gyr_radps: np.ndarray, mag_uT: np.ndarray, freq_hz: float
) -> dict[str, np.ndarray]:
    mad = Madgwick(frequency=freq_hz, gain=0.1)
    q = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    q_arr = np.zeros((len(t_sec), 4), dtype=float)
    acc_world = np.zeros((len(t_sec), 3), dtype=float)
    acc_linear = np.zeros((len(t_sec), 3), dtype=float)
    for i in range(len(t_sec)):
        q = mad.updateMARG(q, gyr=gyr_radps[i], acc=acc_mps2[i], mag=mag_uT[i])
        if q is None:
            q = q_arr[i - 1] if i > 0 else np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
        q = np.asarray(q, dtype=float)
        n = np.linalg.norm(q)
        if n > 0:
            q = q / n
        q_arr[i] = q
        r = quat_to_rot_mat(q)
        acc_world[i] = r @ acc_mps2[i]
        acc_linear[i] = acc_world[i] - np.array([0.0, 0.0, G0], dtype=float)
    vel, pos = integrate_kinematics(t_sec, acc_linear)
    speed = np.linalg.norm(vel, axis=1)
    return {
        "q_wxyz": q_arr,
        "acc_world": acc_world,
        "acc_linear": acc_linear,
        "vel": vel,
        "pos": pos,
        "speed": speed,
    }


def run_zupt_pdr(
    pdr_root: Path, t_sec: np.ndarray, acc_mps2: np.ndarray, gyr_radps: np.ndarray, mag_uT: np.ndarray, freq_hz: float, out_dir: Path
) -> dict[str, np.ndarray]:
    sys.path.insert(0, str(pdr_root))
    from PDR.zupt import zupt  # noqa: PLC0415

    cols = pd.MultiIndex.from_product([["accel", "gyro", "mag"], ["x", "y", "z"]])
    df = pd.DataFrame(np.concatenate([acc_mps2, gyr_radps, mag_uT], axis=1), columns=cols)
    ans = zupt(
        df=df,
        fn="seg_20260315_full_part04_zupt_aided",
        sample_rate=int(round(freq_hz)),
        zupt_tresh=3,
        margin=0.1,
        debug=False,
        lp_filter=False,
        save_dir=str(out_dir),
        show=False,
        return_data=True,
    )
    hf = ans["hf"]
    sf = ans["sf"]

    vel_obj = hf["velocity"]
    if isinstance(vel_obj, pd.DataFrame):
        vel_ned = vel_obj.to_numpy(dtype=float)
    else:
        vel_ned = np.vstack(vel_obj.to_numpy())
    vel_ned = np.asarray(vel_ned, dtype=float)
    if vel_ned.ndim != 2 or vel_ned.shape[1] != 3:
        raise ValueError(f"unexpected ZUPT velocity shape: {vel_ned.shape}")
    vel_ned = (
        pd.DataFrame(vel_ned)
        .interpolate(limit_direction="both")
        .fillna(0.0)
        .to_numpy(dtype=float)
    )
    acc_earth = hf["acceleration"].to_numpy(dtype=float)
    raw_is_moving = hf["is_moving"].to_numpy(dtype=bool)
    labels = derive_motion_labels(
        raw_is_moving=raw_is_moving,
        vel_ned=vel_ned,
        acc_earth=acc_earth,
        gyr_radps=gyr_radps,
        freq_hz=freq_hz,
    )
    hf["motion_context"] = labels["motion_context"]
    hf["step_event"] = labels["step_event"]
    hf["zupt_candidate"] = labels["zupt_candidate"]

    pos_ned = np.zeros_like(vel_ned)
    for i in range(1, len(t_sec)):
        dt = t_sec[i] - t_sec[i - 1]
        if dt <= 0:
            continue
        pos_ned[i] = pos_ned[i - 1] + vel_ned[i] * dt

    # imufusion is configured with CONVENTION_NED in PDR.zupt:
    # velocity columns are [North, East, Down]. Convert to ENU.
    vel_enu = np.column_stack([vel_ned[:, 1], vel_ned[:, 0], -vel_ned[:, 2]])

    pos = np.zeros_like(vel_enu)
    for i in range(1, len(t_sec)):
        dt = t_sec[i] - t_sec[i - 1]
        if dt <= 0:
            continue
        pos[i] = pos[i - 1] + vel_enu[i] * dt

    speed = np.linalg.norm(vel_enu, axis=1)
    return {
        "hf": hf,
        "sf": sf,
        "pos": pos,
        "vel": vel_enu,
        "speed": speed,
        "pos_ned": pos_ned,
        "vel_ned": vel_ned,
        "labels": labels,
        "meta": ans,
    }


def run_zupt_mag_pdr(
    t_sec: np.ndarray,
    acc_mps2: np.ndarray,
    gyr_radps: np.ndarray,
    mag_uT: np.ndarray,
    freq_hz: float,
    out_dir: Path,
) -> dict[str, np.ndarray]:
    dt = 1.0 / float(freq_hz)
    cols = pd.MultiIndex.from_product([["accel", "gyro", "mag"], ["x", "y", "z"]])
    df = pd.DataFrame(
        np.concatenate([acc_mps2, gyr_radps, mag_uT], axis=1),
        columns=cols,
    ).reset_index(drop=True)
    df.index = df.index / float(freq_hz)
    df["gyro"] *= 180.0 / np.pi
    df["accel"] /= G0

    ahrs = imufusion.Ahrs()
    ahrs.settings = imufusion.Settings(
        imufusion.CONVENTION_NED, 0.5, 2000, 10, 30, int(5 * freq_hz)
    )

    sf_rows: list[dict[str, object]] = []
    for _, x in df.iterrows():
        ahrs.update(
            x["gyro"].to_numpy(), x["accel"].to_numpy(), x["mag"].to_numpy(), dt
        )
        euler = ahrs.quaternion.to_euler()
        sf_rows.append(
            {
                "x": float(ahrs.earth_acceleration[0] * G0),
                "y": float(ahrs.earth_acceleration[1] * G0),
                "z": float(ahrs.earth_acceleration[2] * G0),
                "roll": float(euler[0]),
                "pitch": float(euler[1]),
                "yaw": float(euler[2]),
                "Q_T": np.asarray(ahrs.quaternion.wxyz, dtype=float),
            }
        )
    sf = pd.DataFrame(sf_rows, index=df.index)

    hf = sf[["x", "y", "z"]].to_numpy()
    hf = pd.DataFrame(
        hf, columns=pd.MultiIndex.from_product([["acceleration"], ["x", "y", "z"]])
    )
    g_end = np.linalg.norm(hf["acceleration"], axis=1)[-100:].mean()
    g_start = abs(hf["acceleration", "z"][-100:].mean())
    g = min(g_start, g_end)

    margin = int(0.1 * freq_hz)
    acc_earth = hf["acceleration"].to_numpy(dtype=float)
    raw_is_moving = np.linalg.norm(acc_earth, axis=1) > (3.0 + g)
    hf["is_moving"] = raw_is_moving
    for index in range(len(hf) - margin):
        hf.loc[index, "is_moving"] = any(
            hf.loc[index : (index + margin), "is_moving"]
        )
    for index in range(len(hf) - 1, margin, -1):
        hf.loc[index, "is_moving"] = any(
            hf.loc[(index - margin) : index, "is_moving"]
        )
    labels = derive_motion_labels(
        raw_is_moving=hf["is_moving"].to_numpy(dtype=bool),
        vel_ned=np.zeros((len(hf), 3), dtype=float),
        acc_earth=acc_earth,
        gyr_radps=gyr_radps,
        freq_hz=freq_hz,
    )
    hf["motion_context"] = labels["motion_context"]
    dt_arr = np.diff(t_sec, prepend=t_sec[0])
    dt_arr[dt_arr <= 0] = 0.0
    vel_ned, zupt_anchor = integrate_velocity_with_constraints(
        acc_earth=acc_earth,
        dt=dt_arr,
        motion_context=labels["motion_context"],
        g_est=float(g),
        gyr_radps=gyr_radps,
        cfg=ZuptIntegrationConfig(),
    )
    vel_cols = pd.MultiIndex.from_product([["velocity"], ["x", "y", "z"]])
    hf[vel_cols] = vel_ned
    hf["zupt_anchor"] = zupt_anchor.astype(bool)
    pos_ned = np.zeros_like(vel_ned)
    for i in range(1, len(t_sec)):
        dti = t_sec[i] - t_sec[i - 1]
        if dti <= 0:
            continue
        pos_ned[i] = pos_ned[i - 1] + vel_ned[i] * dti

    vel_enu = np.column_stack([vel_ned[:, 1], vel_ned[:, 0], -vel_ned[:, 2]])
    pos_enu = np.zeros_like(vel_enu)
    for i in range(1, len(t_sec)):
        dti = t_sec[i] - t_sec[i - 1]
        if dti <= 0:
            continue
        pos_enu[i] = pos_enu[i - 1] + vel_enu[i] * dti

    labels = derive_motion_labels(
        raw_is_moving=hf["is_moving"].to_numpy(dtype=bool),
        vel_ned=vel_ned,
        acc_earth=acc_earth,
        gyr_radps=gyr_radps,
        freq_hz=freq_hz,
    )
    hf["motion_context"] = labels["motion_context"]
    hf["step_event"] = labels["step_event"]
    hf["zupt_candidate"] = labels["zupt_candidate"]
    steps = int(np.sum(labels["step_event"]))
    speed = np.linalg.norm(vel_enu, axis=1)

    plt.figure(figsize=(8, 4))
    plt.plot(sf.index, sf["yaw"], label="yaw")
    plt.plot(sf.index, sf["pitch"], label="pitch")
    plt.plot(sf.index, sf["roll"], label="roll")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "ypr_mag.png", dpi=180)
    plt.close()

    plt.figure(figsize=(7, 7))
    plt.plot(pos_enu[:, 0], pos_enu[:, 1], label="zupt-aided-mag path")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.axis("equal")
    plt.tight_layout()
    plt.savefig(out_dir / "path2D_seg_20260315_full_part04_zupt_aided_mag.png", dpi=180)
    plt.close()

    return {
        "hf": hf,
        "sf": sf,
        "vel_ned": vel_ned,
        "pos_ned": pos_ned,
        "vel": vel_enu,
        "pos": pos_enu,
        "speed": speed,
        "steps": steps,
        "labels": labels,
        "g_est": float(g),
    }


def read_gnss_reference(segment_sat_csv: Path) -> pd.DataFrame:
    sat = pd.read_csv(segment_sat_csv)
    sat = sat.rename(columns={"Latitude": "lat_deg", "Longitude": "lon_deg", "Altitude": "alt_m"})
    sat["unix_ms"] = sat.apply(lambda r: gps_week_sow_to_unix_ms(float(r["week"]), float(r["seconds of week [s]"])), axis=1)
    sat["unix_sec"] = sat["unix_ms"] / 1000.0
    lat0 = np.deg2rad(float(sat["lat_deg"].iloc[0]))
    lon0 = np.deg2rad(float(sat["lon_deg"].iloc[0]))
    lat = np.deg2rad(sat["lat_deg"].to_numpy(dtype=float))
    lon = np.deg2rad(sat["lon_deg"].to_numpy(dtype=float))
    sat["east_m"] = (lon - lon0) * np.cos(lat0) * EARTH_RADIUS_M
    sat["north_m"] = (lat - lat0) * EARTH_RADIUS_M
    sat["up_m"] = sat["alt_m"] - float(sat["alt_m"].iloc[0])
    de = np.diff(sat["east_m"].to_numpy(dtype=float), prepend=np.nan)
    dn = np.diff(sat["north_m"].to_numpy(dtype=float), prepend=np.nan)
    du = np.diff(sat["up_m"].to_numpy(dtype=float), prepend=np.nan)
    dt = np.diff(sat["unix_sec"].to_numpy(dtype=float), prepend=np.nan)
    spd = np.sqrt(de * de + dn * dn + du * du)
    spd = np.where(dt > 0, spd / dt, np.nan)
    sat["speed_mps"] = pd.Series(spd).interpolate(limit_direction="both")
    return sat


def write_method_csv(path: Path, t_sec_abs: np.ndarray, vel: np.ndarray, pos: np.ndarray, speed: np.ndarray) -> None:
    df = pd.DataFrame(
        {
            "t_sec_abs": t_sec_abs,
            "t_sec_rel": t_sec_abs - t_sec_abs[0],
            "vel_x_mps": vel[:, 0],
            "vel_y_mps": vel[:, 1],
            "vel_z_mps": vel[:, 2],
            "speed_mps": speed,
            "pos_x_m": pos[:, 0],
            "pos_y_m": pos[:, 1],
            "pos_z_m": pos[:, 2],
        }
    )
    df.to_csv(path, index=False)


def write_zupt_internal_csv(
    out_dir: Path,
    prefix: str,
    t_sec_abs: np.ndarray,
    sf: pd.DataFrame,
    vel_ned: np.ndarray,
    pos_ned: np.ndarray,
    hf: pd.DataFrame,
) -> tuple[Path, Path]:
    q = np.vstack(sf["Q_T"].to_numpy())
    quat_csv = out_dir / f"{prefix}_quat_ned.csv"
    pd.DataFrame(
        {
            "t_sec_abs": t_sec_abs,
            "t_sec_rel": t_sec_abs - t_sec_abs[0],
            "q_w": q[:, 0],
            "q_x": q[:, 1],
            "q_y": q[:, 2],
            "q_z": q[:, 3],
            "roll_deg": sf["roll"].to_numpy(dtype=float),
            "pitch_deg": sf["pitch"].to_numpy(dtype=float),
            "yaw_deg": sf["yaw"].to_numpy(dtype=float),
        }
    ).to_csv(quat_csv, index=False)

    moving = hf["is_moving"].to_numpy(dtype=bool)
    motion_context = hf["motion_context"].to_numpy(dtype=bool) if "motion_context" in hf else moving
    step_event = hf["step_event"].to_numpy(dtype=bool) if "step_event" in hf else np.zeros_like(moving, dtype=bool)
    zupt_candidate = hf["zupt_candidate"].to_numpy(dtype=bool) if "zupt_candidate" in hf else (~moving)
    traj_csv = out_dir / f"{prefix}_traj_ned.csv"
    pd.DataFrame(
        {
            "t_sec_abs": t_sec_abs,
            "t_sec_rel": t_sec_abs - t_sec_abs[0],
            "vel_n_mps": vel_ned[:, 0],
            "vel_e_mps": vel_ned[:, 1],
            "vel_d_mps": vel_ned[:, 2],
            "speed_mps": np.linalg.norm(vel_ned, axis=1),
            "pos_n_m": pos_ned[:, 0],
            "pos_e_m": pos_ned[:, 1],
            "pos_d_m": pos_ned[:, 2],
            "is_moving": moving.astype(int),
            "motion_context": motion_context.astype(int),
            "step_event": step_event.astype(int),
            "zupt_candidate": zupt_candidate.astype(int),
        }
    ).to_csv(traj_csv, index=False)
    return quat_csv, traj_csv


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
    if len(src_t) == 0:
        return np.full_like(target_t, np.nan, dtype=float)
    keep = np.concatenate(([True], np.diff(src_t) > 0))
    t = src_t[keep]
    v = src_v[keep]
    if len(t) < 2:
        return np.full_like(target_t, np.nan, dtype=float)
    y = pd.Series(v).interpolate(limit_direction="both").to_numpy(dtype=float)
    return np.interp(target_t, t, y, left=np.nan, right=np.nan)


def plot_trajectory_compare(
    out_png: Path,
    gnss: pd.DataFrame,
    std_en: tuple[np.ndarray, np.ndarray],
    zupt_en: tuple[np.ndarray, np.ndarray],
    ahrs_en: tuple[np.ndarray, np.ndarray],
    zupt_mag_en: tuple[np.ndarray, np.ndarray] | None = None,
) -> None:
    plt.figure(figsize=(10, 9))
    plt.plot(gnss["east_m"], gnss["north_m"], label="GNSS", linewidth=2.0)
    plt.plot(std_en[0], std_en[1], label="Standard PDR", alpha=0.9)
    plt.plot(zupt_en[0], zupt_en[1], label="ZUPT-aided PDR", alpha=0.9)
    if zupt_mag_en is not None:
        plt.plot(
            zupt_mag_en[0],
            zupt_mag_en[1],
            label="ZUPT-aided-mag PDR",
            alpha=0.9,
        )
    plt.plot(ahrs_en[0], ahrs_en[1], label="AHRS+PDR", alpha=0.9)
    plt.xlabel("East [m]")
    plt.ylabel("North [m]")
    plt.title("Trajectory Comparison (Part04)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.axis("equal")
    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    plt.close()


def plot_speed_compare(
    out_png: Path,
    gnss_t: np.ndarray,
    gnss_speed: np.ndarray,
    std_speed: np.ndarray,
    zupt_speed: np.ndarray,
    ahrs_speed: np.ndarray,
    zupt_mag_speed: np.ndarray | None = None,
) -> None:
    t0 = float(gnss_t[0])
    tx = gnss_t - t0
    plt.figure(figsize=(12, 6))
    plt.plot(tx, gnss_speed, label="GNSS speed", linewidth=2.0)
    plt.plot(tx, std_speed, label="Standard PDR speed", alpha=0.9)
    plt.plot(tx, zupt_speed, label="ZUPT-aided PDR speed", alpha=0.9)
    if zupt_mag_speed is not None:
        plt.plot(
            tx,
            zupt_mag_speed,
            label="ZUPT-aided-mag PDR speed",
            alpha=0.9,
        )
    plt.plot(tx, ahrs_speed, label="AHRS+PDR speed", alpha=0.9)
    plt.xlabel("Time from start [s]")
    plt.ylabel("Speed [m/s]")
    plt.title("Speed Trend Comparison (Part04)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    plt.close()


def main() -> None:
    p = argparse.ArgumentParser(description="Run standard/AHRS/ZUPT/ZUPT-mag PDR pipeline for a segment.")
    p.add_argument("--segment", default="seg_20260315_full_part04")
    p.add_argument("--satellite-csv", default=None, help="Override satellite_lla.csv path")
    p.add_argument("--uimu-file", default=None, help="Override UIMU_Log file path")
    p.add_argument("--calib-csv", default=None, help="Override calib.csv path")
    p.add_argument("--in-dir", default=None, help="Override input directory for prepared PDR CSVs")
    p.add_argument("--out-dir", default=None, help="Override output directory for PDR results")
    args = p.parse_args()

    repo_root = REPO_ROOT
    pdr_root = PDR_LIB_ROOT
    seg = args.segment
    segment_sat_csv = (
        Path(args.satellite_csv).resolve()
        if args.satellite_csv
        else repo_root / "data" / "bison_input" / seg / "satellite_lla.csv"
    )
    explicit_uimu_file = Path(args.uimu_file).resolve() if args.uimu_file else None

    in_dir = Path(args.in_dir).resolve() if args.in_dir else (repo_root / "data" / "pdr_input" / seg)
    out_dir = Path(args.out_dir).resolve() if args.out_dir else (repo_root / "PDR" / "output" / f"{seg}_three_impl")
    in_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    imu_input_csv = in_dir / "paired_imu_for_pdr.csv"
    start_ms, end_ms = read_satellite_time_window_ms(segment_sat_csv)
    uimu_file, acc_rows, gyr_rows, mag_rows, uimu_select_meta = select_uimu_file(
        repo_root, explicit_uimu_file, start_ms, end_ms
    )
    t_sec_abs, acc_mps2, gyr_radps, mag_uT = pair_acc_gyro_mag(acc_rows, gyr_rows, mag_rows)
    if len(t_sec_abs) < 10:
        if imu_input_csv.exists():
            cached = pd.read_csv(imu_input_csv)
            req = [
                "t_sec_abs",
                "ax_mps2",
                "ay_mps2",
                "az_mps2",
                "gx_radps",
                "gy_radps",
                "gz_radps",
                "mx_uT",
                "my_uT",
                "mz_uT",
            ]
            miss = [c for c in req if c not in cached.columns]
            if miss:
                raise RuntimeError(f"paired rows are too few; cached paired imu missing columns: {miss}")
            t_sec_abs = cached["t_sec_abs"].to_numpy(dtype=float)
            acc_mps2 = cached[["ax_mps2", "ay_mps2", "az_mps2"]].to_numpy(dtype=float)
            gyr_radps = cached[["gx_radps", "gy_radps", "gz_radps"]].to_numpy(dtype=float)
            mag_uT = cached[["mx_uT", "my_uT", "mz_uT"]].to_numpy(dtype=float)
        else:
            raise RuntimeError(
                f"paired rows are too few (count={len(t_sec_abs)}); raw={uimu_file}, sat={segment_sat_csv}"
            )
    if len(t_sec_abs) < 10:
        raise RuntimeError("paired rows are too few after fallback")
    calib_csv = resolve_calib_csv(args.calib_csv, repo_root)
    calib_data, calib_meta = load_calib(calib_csv)
    acc_mps2, gyr_radps, mag_uT = apply_calib(acc_mps2, gyr_radps, mag_uT, calib_data)
    freq_hz = estimate_frequency_hz(t_sec_abs)

    pd.DataFrame(
        {
            "t_sec_abs": t_sec_abs,
            "t_sec_rel": t_sec_abs - t_sec_abs[0],
            "ax_mps2": acc_mps2[:, 0],
            "ay_mps2": acc_mps2[:, 1],
            "az_mps2": acc_mps2[:, 2],
            "gx_radps": gyr_radps[:, 0],
            "gy_radps": gyr_radps[:, 1],
            "gz_radps": gyr_radps[:, 2],
            "mx_uT": mag_uT[:, 0],
            "my_uT": mag_uT[:, 1],
            "mz_uT": mag_uT[:, 2],
        }
    ).to_csv(imu_input_csv, index=False)

    gnss = read_gnss_reference(segment_sat_csv)
    gnss_csv = in_dir / "gnss_reference_part04.csv"
    gnss.to_csv(gnss_csv, index=False)

    std = run_standard_pdr(t_sec_abs, acc_mps2, freq_hz)
    ahrs = run_ahrs_pdr(t_sec_abs, acc_mps2, gyr_radps, mag_uT, freq_hz)
    zupt = run_zupt_pdr(
        pdr_root, t_sec_abs, acc_mps2, gyr_radps, mag_uT, freq_hz, out_dir
    )
    zupt_mag = run_zupt_mag_pdr(
        t_sec_abs, acc_mps2, gyr_radps, mag_uT, freq_hz, out_dir
    )

    write_method_csv(
        out_dir / "standard_pdr_traj_vel.csv",
        t_sec_abs,
        std["vel"],
        std["pos"],
        std["speed"],
    )
    write_method_csv(
        out_dir / "ahrs_pdr_traj_vel.csv",
        t_sec_abs,
        ahrs["vel"],
        ahrs["pos"],
        ahrs["speed"],
    )
    write_method_csv(
        out_dir / "zupt_aided_pdr_traj_vel.csv",
        t_sec_abs,
        zupt["vel"],
        zupt["pos"],
        zupt["speed"],
    )
    write_method_csv(
        out_dir / "zupt_aided_mag_pdr_traj_vel.csv",
        t_sec_abs,
        zupt_mag["vel"],
        zupt_mag["pos"],
        zupt_mag["speed"],
    )
    pd.DataFrame(
        {
            "t_sec_abs": t_sec_abs,
            "q_w": ahrs["q_wxyz"][:, 0],
            "q_x": ahrs["q_wxyz"][:, 1],
            "q_y": ahrs["q_wxyz"][:, 2],
            "q_z": ahrs["q_wxyz"][:, 3],
        }
    ).to_csv(out_dir / "ahrs_quaternion.csv", index=False)
    zupt_quat_csv, zupt_ned_traj_csv = write_zupt_internal_csv(
        out_dir,
        "zupt_aided_internal_ned",
        t_sec_abs,
        zupt["sf"],
        zupt["vel_ned"],
        zupt["pos_ned"],
        zupt["hf"],
    )
    zupt_mag_quat_csv, zupt_mag_ned_traj_csv = write_zupt_internal_csv(
        out_dir,
        "zupt_aided_mag_internal_ned",
        t_sec_abs,
        zupt_mag["sf"],
        zupt_mag["vel_ned"],
        zupt_mag["pos_ned"],
        zupt_mag["hf"],
    )

    gnss_t = gnss["unix_sec"].to_numpy(dtype=float)
    std_e = interp_to_times(t_sec_abs, std["pos"][:, 0], gnss_t)
    std_n = interp_to_times(t_sec_abs, std["pos"][:, 1], gnss_t)
    zupt_e = interp_to_times(t_sec_abs, zupt["pos"][:, 0], gnss_t)
    zupt_n = interp_to_times(t_sec_abs, zupt["pos"][:, 1], gnss_t)
    zupt_mag_e = interp_to_times(t_sec_abs, zupt_mag["pos"][:, 0], gnss_t)
    zupt_mag_n = interp_to_times(t_sec_abs, zupt_mag["pos"][:, 1], gnss_t)
    ahrs_e = interp_to_times(t_sec_abs, ahrs["pos"][:, 0], gnss_t)
    ahrs_n = interp_to_times(t_sec_abs, ahrs["pos"][:, 1], gnss_t)
    std_speed = interp_to_times(t_sec_abs, std["speed"], gnss_t)
    zupt_speed = interp_to_times(t_sec_abs, zupt["speed"], gnss_t)
    zupt_mag_speed = interp_to_times(t_sec_abs, zupt_mag["speed"], gnss_t)
    ahrs_speed = interp_to_times(t_sec_abs, ahrs["speed"], gnss_t)
    gnss_speed = gnss["speed_mps"].to_numpy(dtype=float)

    aligned_csv = out_dir / "gnss_time_aligned_compare.csv"
    pd.DataFrame(
        {
            "unix_sec": gnss_t,
            "gnss_east_m": gnss["east_m"],
            "gnss_north_m": gnss["north_m"],
            "gnss_speed_mps": gnss_speed,
            "std_east_m": std_e,
            "std_north_m": std_n,
            "std_speed_mps": std_speed,
            "zupt_east_m": zupt_e,
            "zupt_north_m": zupt_n,
            "zupt_speed_mps": zupt_speed,
            "zupt_mag_east_m": zupt_mag_e,
            "zupt_mag_north_m": zupt_mag_n,
            "zupt_mag_speed_mps": zupt_mag_speed,
            "ahrs_east_m": ahrs_e,
            "ahrs_north_m": ahrs_n,
            "ahrs_speed_mps": ahrs_speed,
        }
    ).to_csv(aligned_csv, index=False)

    traj_png = out_dir / "traj_compare_three_pdr_vs_gnss.png"
    speed_png = out_dir / "speed_trend_compare_three_pdr_vs_gnss.png"
    plot_trajectory_compare(
        traj_png,
        gnss,
        (std_e, std_n),
        (zupt_e, zupt_n),
        (ahrs_e, ahrs_n),
        (zupt_mag_e, zupt_mag_n),
    )
    plot_speed_compare(
        speed_png,
        gnss_t,
        gnss_speed,
        std_speed,
        zupt_speed,
        ahrs_speed,
        zupt_mag_speed,
    )

    traj_zupt_cmp_png = out_dir / "traj_compare_zupt_vs_zupt_mag_vs_gnss.png"
    plt.figure(figsize=(10, 9))
    plt.plot(gnss["east_m"], gnss["north_m"], label="GNSS", linewidth=2.0)
    plt.plot(zupt_e, zupt_n, label="ZUPT-aided PDR")
    plt.plot(zupt_mag_e, zupt_mag_n, label="ZUPT-aided-mag PDR")
    plt.xlabel("East [m]")
    plt.ylabel("North [m]")
    plt.title("Trajectory Comparison: ZUPT vs ZUPT-mag (Part04)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.axis("equal")
    plt.tight_layout()
    plt.savefig(traj_zupt_cmp_png, dpi=180)
    plt.close()

    speed_zupt_cmp_png = out_dir / "speed_trend_compare_zupt_vs_zupt_mag_vs_gnss.png"
    t0 = float(gnss_t[0])
    tx = gnss_t - t0
    plt.figure(figsize=(12, 6))
    plt.plot(tx, gnss_speed, label="GNSS speed", linewidth=2.0)
    plt.plot(tx, zupt_speed, label="ZUPT-aided PDR speed")
    plt.plot(tx, zupt_mag_speed, label="ZUPT-aided-mag PDR speed")
    plt.xlabel("Time from start [s]")
    plt.ylabel("Speed [m/s]")
    plt.title("Speed Trend: ZUPT vs ZUPT-mag (Part04)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(speed_zupt_cmp_png, dpi=180)
    plt.close()

    speed_metrics = {
        "standard_pdr": compute_speed_metrics(gnss_speed, std_speed),
        "zupt_aided_pdr": compute_speed_metrics(gnss_speed, zupt_speed),
        "zupt_aided_mag_pdr": compute_speed_metrics(gnss_speed, zupt_mag_speed),
        "ahrs_pdr": compute_speed_metrics(gnss_speed, ahrs_speed),
    }

    report = {
        "segment": seg,
        "inputs": {
            "uimu_file": str(uimu_file),
            "uimu_select_mode": str(uimu_select_meta.get("mode", "")),
            "uimu_select_tested_files": int(uimu_select_meta.get("tested", 0)),
            "uimu_best_paired_rows": int(uimu_select_meta.get("best_paired_rows", len(t_sec_abs))),
            "satellite_lla": str(segment_sat_csv),
            "calib_csv": calib_meta["calib_csv"],
            "calib_loaded": bool(calib_meta["loaded"]),
            "calib_fallback_identity": bool(calib_meta["fallback_identity"]),
            "prepared_imu_csv": str(imu_input_csv),
            "prepared_gnss_csv": str(gnss_csv),
        },
        "raw_counts": {
            "acc": int(len(acc_rows)),
            "gyro": int(len(gyr_rows)),
            "mag": int(len(mag_rows)),
        },
        "paired_rows": int(len(t_sec_abs)),
        "sample_rate_hz": float(freq_hz),
        "outputs": {
            "standard_pdr_csv": str(out_dir / "standard_pdr_traj_vel.csv"),
            "ahrs_pdr_csv": str(out_dir / "ahrs_pdr_traj_vel.csv"),
            "zupt_aided_pdr_csv": str(out_dir / "zupt_aided_pdr_traj_vel.csv"),
            "zupt_aided_mag_pdr_csv": str(out_dir / "zupt_aided_mag_pdr_traj_vel.csv"),
            "ahrs_quaternion_csv": str(out_dir / "ahrs_quaternion.csv"),
            "zupt_internal_ned_quat_csv": str(zupt_quat_csv),
            "zupt_internal_ned_traj_csv": str(zupt_ned_traj_csv),
            "zupt_mag_internal_ned_quat_csv": str(zupt_mag_quat_csv),
            "zupt_mag_internal_ned_traj_csv": str(zupt_mag_ned_traj_csv),
            "zupt_ypr_png": str(out_dir / "ypr.png"),
            "zupt_mag_ypr_png": str(out_dir / "ypr_mag.png"),
            "zupt_path_png": zupt["meta"]["path_png"],
            "zupt_path2d_png": zupt["meta"]["path2d_png"],
            "zupt_mag_path2d_png": str(
                out_dir / "path2D_seg_20260315_full_part04_zupt_aided_mag.png"
            ),
            "aligned_compare_csv": str(aligned_csv),
            "traj_compare_png": str(traj_png),
            "speed_compare_png": str(speed_png),
            "traj_zupt_compare_png": str(traj_zupt_cmp_png),
            "speed_zupt_compare_png": str(speed_zupt_cmp_png),
        },
        "speed_metrics_vs_gnss": speed_metrics,
        "zupt_meta": {
            "steps": int(np.sum(zupt["labels"]["step_event"])),
            "g_est": float(zupt["meta"]["g_est"]),
        },
        "zupt_mag_meta": {
            "steps": int(zupt_mag["steps"]),
            "g_est": float(zupt_mag["g_est"]),
        },
    }
    (out_dir / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
