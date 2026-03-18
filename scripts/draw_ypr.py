#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import imufusion
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.use("Agg")


def find_repo_root(start: Path) -> Path:
    for p in [start, *start.parents]:
        if (p / "capture").exists() and (p / "PDR").exists():
            return p
    raise FileNotFoundError(f"cannot locate repo root from {start}")


def load_capture_csv(csv_path: Path, sample_rate: int) -> pd.DataFrame:
    # Keep the same loading style used in debug/zupt.ipynb.
    df = pd.read_csv(csv_path, index_col=[0], header=[0, 1]).reset_index(drop=True)
    need = [("accel", "x"), ("accel", "y"), ("accel", "z"), ("gyro", "x"), ("gyro", "y"), ("gyro", "z"), ("mag", "x"), ("mag", "y"), ("mag", "z")]
    miss = [c for c in need if c not in df.columns]
    if miss:
        raise ValueError(f"input csv missing columns: {miss}")
    df = df.loc[:, need].copy()
    df.columns = pd.MultiIndex.from_tuples(need)
    df["gyro"] *= 180.0 / np.pi
    df.index = df.index / float(sample_rate)
    return df


def build_sf(df: pd.DataFrame, sample_rate: int) -> pd.DataFrame:
    # Follow debug/zupt.ipynb logic: imufusion + update_no_magnetometer with fixed 0.005 dt.
    ahrs = imufusion.Ahrs()
    ahrs.settings = imufusion.Settings(
        imufusion.CONVENTION_NED,
        5,
        2000,
        10,
        10,
        5 * sample_rate,
    )

    def update(x: pd.Series) -> dict[str, object]:
        ahrs.update_no_magnetometer(x["gyro"].to_numpy(), x["accel"].to_numpy(), 0.005)
        euler = ahrs.quaternion.to_euler()
        q = ahrs.quaternion.wxyz
        acceleration = ahrs.earth_acceleration
        return {
            "x": acceleration[0],
            "y": acceleration[1],
            "z": acceleration[2],
            "roll": euler[0],
            "pitch": euler[1],
            "yaw": euler[2],
            "Q_T": q,
            "accel_err": ahrs.internal_states.acceleration_error,
            "accel_igr": ahrs.internal_states.accelerometer_ignored,
            "accel_rec": ahrs.internal_states.acceleration_recovery_trigger,
            "ang_rrec": ahrs.flags.angular_rate_recovery,
            "accel_rrec": ahrs.flags.acceleration_recovery,
        }

    sf = df.apply(update, axis=1)
    return pd.DataFrame(list(sf), index=df.index)


def draw_ypr(df: pd.DataFrame, sf: pd.DataFrame, out_png: Path) -> None:
    plt.style.use("default")
    fig, ax = plt.subplots(
        nrows=6,
        sharex=True,
        figsize=(20, 15),
        tight_layout=True,
        gridspec_kw={"height_ratios": [6, 6, 6, 2, 1, 1]},
    )

    ax[0].plot(df.index, df["gyro", "x"], "tab:red", label="gyro x")
    ax[0].plot(df.index, df["gyro", "y"], "tab:green", label="gyro y")
    ax[0].plot(df.index, df["gyro", "z"], "tab:blue", label="gyro z")
    ax[0].set_ylabel("Degrees/s")
    ax[0].set_title("Gyroscope")
    ax[0].legend()
    ax[0].grid()

    ax[1].plot(df.index, df["accel", "x"], "tab:red", label="Accelerometer x")
    ax[1].plot(df.index, df["accel", "y"], "tab:green", label="Accelerometer y")
    ax[1].plot(df.index, df["accel", "z"], "tab:blue", label="Accelerometer z")
    ax[1].set_ylabel("Acceleration [m/s/s]")
    ax[1].set_title("Accelerometer")
    ax[1].legend()
    ax[1].grid()

    ax[2].plot(sf["yaw"], "tab:red", label="Yaw")
    ax[2].plot(sf["pitch"], "tab:green", label="Pitch")
    ax[2].plot(sf["roll"], "tab:blue", label="Roll")
    ax[2].set_ylabel("Degrees")
    ax[2].grid()
    ax[2].legend()

    ax[3].plot(sf["accel_err"], "tab:olive", label="Acceleration error")
    ax[3].set_ylabel("Error")
    ax[3].set_xlabel("Time [s]")
    ax[3].legend()

    ax[4].plot(sf["accel_igr"], "tab:cyan", label="Acceleration ignored")
    ax[4].legend()

    ax[5].plot(sf["accel_rec"], "tab:orange", label="Acceleration recovery trigger")
    ax[5].legend()

    for axes in ax:
        axes.set_xlim(0, float(sf.index.max()))

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=300)
    plt.close(fig)


def main() -> None:
    repo_root = find_repo_root(Path(__file__).resolve().parent)
    parser = argparse.ArgumentParser(description="Draw ypr.png from capture csv using zupt.ipynb logic.")
    parser.add_argument("--csv", required=True, help="input capture csv path")
    parser.add_argument("--sample-rate", type=int, default=200, help="sample rate used to map index to seconds")
    parser.add_argument("--output", default=None, help="output png path (default: <repo>/output/ypr.png)")
    args = parser.parse_args()

    csv_path = Path(args.csv).resolve()
    if not csv_path.exists():
        raise FileNotFoundError(f"missing csv: {csv_path}")
    out_png = Path(args.output).resolve() if args.output else (repo_root / "output" / "ypr.png")

    df = load_capture_csv(csv_path, sample_rate=int(args.sample_rate))
    sf = build_sf(df, sample_rate=int(args.sample_rate))
    draw_ypr(df, sf, out_png)
    print(f'{{"input_csv":"{csv_path}","output_png":"{out_png}"}}')


if __name__ == "__main__":
    main()
