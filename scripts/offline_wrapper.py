#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Callable
import sys

import numpy as np
import pandas as pd
from ahrs.filters import EKF, Madgwick, Mahony

G0 = 9.80665


def find_repo_root(start: Path) -> Path:
    for p in [start, *start.parents]:
        if (p / "PDR").exists() and (p / "capture").exists():
            return p
    raise FileNotFoundError(f"cannot locate repo root from {start}")


REPO_ROOT = find_repo_root(Path(__file__).resolve().parent)
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from PDR.pdr import PDR


def quat_to_euler_deg(q: np.ndarray) -> np.ndarray:
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


@dataclass
class CalibEvent:
    idx: int
    kind: str


class OfflineUdpReader:
    def __init__(self, data: pd.DataFrame, chunk_size: int = 20):
        self._data = data
        self._chunk_size = max(1, int(chunk_size))
        self.cursor = 0

    def read(self) -> pd.DataFrame:
        if self.cursor >= len(self._data):
            raise StopIteration
        end = min(len(self._data), self.cursor + self._chunk_size)
        out = self._data.iloc[self.cursor:end].copy()
        self.cursor = end
        return out

    def close(self) -> None:
        return


class OfflinePDR(PDR):
    def udp_init(self) -> None:
        self.udp = None

    def attach_offline_reader(self, reader: OfflineUdpReader, raw_data: pd.DataFrame) -> None:
        self.udp = reader
        self._offline_reader = reader
        self._offline_raw = raw_data

    def _calib_window(self, n: int) -> pd.DataFrame:
        start = int(self._offline_reader.cursor)
        end = min(len(self._offline_raw), start + max(1, int(n)))
        return self._offline_raw.iloc[start:end]

    def calibrate_gyro(self):
        w = self._calib_window(1000)["gyro"]
        if len(w) == 0:
            return self.calib_b["gyro"].to_numpy(dtype=float)
        bg = w.mean().to_numpy(dtype=float)
        self.calib_b["gyro"] = bg
        self.save_calib()
        return bg

    def calibrate_accel(self):
        w = self._calib_window(100)["accel"]
        if len(w) == 0:
            return self.calib_b["accel"].to_numpy(dtype=float), self.calib_A["accel"].to_numpy(dtype=float)
        m = w.mean().to_numpy(dtype=float)
        n = float(np.linalg.norm(m))
        if n > 1e-9:
            b = m - G0 * (m / n)
        else:
            b = np.zeros(3, dtype=float)
        a = np.eye(3, dtype=float)
        self.calib_b["accel"] = b
        self.calib_A["accel"] = a
        self.save_calib()
        return b, a

    def calibrate_mag(self):
        w = self._calib_window(2000)["mag"]
        if len(w) == 0:
            return self.calib_b["mag"].to_numpy(dtype=float), self.calib_A["mag"].to_numpy(dtype=float)
        x = w.to_numpy(dtype=float)
        b = np.mean(x, axis=0)
        xc = x - b[None, :]
        cov = np.cov(xc, rowvar=False) if len(xc) > 2 else np.eye(3, dtype=float)
        ew, ev = np.linalg.eigh(cov)
        ew = np.clip(ew, 1e-8, None)
        inv_sqrt = ev @ np.diag(1.0 / np.sqrt(ew)) @ ev.T
        y = (inv_sqrt @ xc.T).T
        mean_norm = float(np.mean(np.linalg.norm(y, axis=1)))
        target = float(self.norm if np.isfinite(self.norm) and self.norm > 1e-6 else 55.0)
        scale = target / max(mean_norm, 1e-6)
        a = scale * inv_sqrt
        self.calib_b["mag"] = b
        self.calib_A["mag"] = a
        self.save_calib()
        return b, a


def parse_events(spec: str) -> list[CalibEvent]:
    out: list[CalibEvent] = []
    if not spec.strip():
        return out
    for token in spec.split(","):
        t = token.strip()
        if not t:
            continue
        if "@" not in t:
            raise ValueError(f"invalid event '{t}', expected kind@index")
        kind, idx_s = t.split("@", 1)
        kind = kind.strip().lower()
        if kind not in {"gyro", "accel", "mag"}:
            raise ValueError(f"unsupported calibration kind '{kind}'")
        out.append(CalibEvent(idx=int(idx_s), kind=kind))
    out.sort(key=lambda e: e.idx)
    return out


def filter_cls(name: str):
    n = name.strip().lower()
    m = {"mahony": Mahony, "madgwick": Madgwick, "ekf": EKF}
    if n not in m:
        raise ValueError(f"unsupported filter '{name}', choose one of: {', '.join(sorted(m))}")
    return m[n]


def load_capture_csv(path: Path, frequency: float) -> pd.DataFrame:
    # same loading style as debug/zupt.ipynb
    df = pd.read_csv(path, index_col=[0], header=[0, 1]).reset_index(drop=True)
    core = [("accel", "x"), ("accel", "y"), ("accel", "z"), ("gyro", "x"), ("gyro", "y"), ("gyro", "z"), ("mag", "x"), ("mag", "y"), ("mag", "z")]
    miss = [c for c in core if c not in df.columns]
    if miss:
        raise ValueError(f"input csv missing columns: {miss}")

    if ("time", "") in df.columns:
        t_ms = df[("time", "")].to_numpy(dtype=float)
    else:
        dt_ms = 1000.0 / max(float(frequency), 1e-9)
        t_ms = np.arange(len(df), dtype=float) * dt_ms

    out = pd.DataFrame(index=df.index)
    out[("time", "")] = t_ms
    for c in core:
        out[c] = df[c].to_numpy(dtype=float)
    out.columns = pd.MultiIndex.from_tuples([("time", ""), *core])
    return out


def run_pipeline(
    input_csv: Path,
    out_dir: Path,
    filter_name: str,
    frequency: float,
    chunk_size: int,
    lp: bool,
    cutoff: float,
    order: int,
    events: list[CalibEvent],
) -> tuple[Path, Path]:
    raw = load_capture_csv(input_csv, frequency=frequency)
    reader = OfflineUdpReader(raw, chunk_size=chunk_size)
    pdr = OfflinePDR(filter=filter_cls(filter_name), frequency=frequency, lp=lp, cutoff=cutoff, order=order)
    pdr.attach_offline_reader(reader, raw)
    pdr.capture = True

    dispatch: dict[str, Callable[[], object]] = {
        "gyro": pdr.calibrate_gyro,
        "accel": pdr.calibrate_accel,
        "mag": pdr.calibrate_mag,
    }
    ev_i = 0
    while reader.cursor < len(raw):
        while ev_i < len(events) and events[ev_i].idx <= reader.cursor:
            dispatch[events[ev_i].kind]()
            ev_i += 1
        try:
            pdr.update()
        except StopIteration:
            break
    while ev_i < len(events):
        dispatch[events[ev_i].kind]()
        ev_i += 1

    if len(pdr.data) == 0:
        raise RuntimeError("no samples were processed")

    q = np.vstack(pdr.data["Q"].to_numpy())
    eul = np.vstack([quat_to_euler_deg(v) for v in q])
    t_raw = pdr.data[("time", "")].to_numpy(dtype=float)
    t_rel = np.arange(len(t_raw), dtype=float) / float(frequency)
    if len(t_raw) > 1 and np.all(np.diff(t_raw) >= 0):
        t_rel = (t_raw - t_raw[0]) / 1000.0

    out_df = pd.DataFrame(
        {
            "t_ms": t_raw,
            "t_sec_rel": t_rel,
            "q_w": q[:, 0],
            "q_x": q[:, 1],
            "q_y": q[:, 2],
            "q_z": q[:, 3],
            "roll_deg": eul[:, 0],
            "pitch_deg": eul[:, 1],
            "yaw_deg": eul[:, 2],
        }
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    result_csv = out_dir / f"{input_csv.stem}_result.csv"
    out_df.to_csv(result_csv, index=False)

    calib_root = find_repo_root(Path(__file__).resolve().parent) / "calib.csv"
    calib_out = out_dir / "calib.csv"
    if calib_root.exists():
        shutil.copy2(calib_root, calib_out)
    else:
        pdr.calib.T.to_csv(calib_out, sep="\t")
    return result_csv, calib_out


def main() -> None:
    repo_root = find_repo_root(Path(__file__).resolve().parent)
    p = argparse.ArgumentParser(description="Offline wrapper: replay capture/*.csv through PDR.update() and export attitude.")
    p.add_argument("--input-csv", required=True, help="capture csv path")
    p.add_argument("--output-dir", default=str(repo_root / "output"), help="output directory (default: <repo>/output)")
    p.add_argument("--filter", default="Mahony", choices=["Mahony", "Madgwick", "EKF"], help="attitude filter")
    p.add_argument("--frequency", type=float, default=200.0, help="sample rate (Hz)")
    p.add_argument("--chunk-size", type=int, default=20, help="samples per offline read")
    p.add_argument("--lp", action="store_true", help="enable live low-pass filter")
    p.add_argument("--cutoff", type=float, default=5.0, help="low-pass cutoff")
    p.add_argument("--order", type=int, default=10, help="low-pass order")
    p.add_argument(
        "--calib-events",
        default="gyro@0,accel@0,mag@0",
        help="calibration events, e.g. 'gyro@0,accel@0,mag@2000' (sample index)",
    )
    args = p.parse_args()

    input_csv = Path(args.input_csv).resolve()
    if not input_csv.exists():
        raise FileNotFoundError(f"missing input csv: {input_csv}")
    out_dir = Path(args.output_dir).resolve()
    events = parse_events(args.calib_events)
    result_csv, calib_csv = run_pipeline(
        input_csv=input_csv,
        out_dir=out_dir,
        filter_name=args.filter,
        frequency=float(args.frequency),
        chunk_size=int(args.chunk_size),
        lp=bool(args.lp),
        cutoff=float(args.cutoff),
        order=int(args.order),
        events=events,
    )
    print(
        f'{{"result_csv":"{result_csv}","calib_csv":"{calib_csv}","events":"{args.calib_events}"}}'
    )


if __name__ == "__main__":
    main()
