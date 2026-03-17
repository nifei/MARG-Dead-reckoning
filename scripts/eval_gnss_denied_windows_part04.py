#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def interp(ts: np.ndarray, ys: np.ndarray, tq: np.ndarray) -> np.ndarray:
    keep = np.concatenate(([True], np.diff(ts) > 0))
    t = ts[keep]
    y = ys[keep]
    if len(t) < 2:
        return np.full_like(tq, np.nan, dtype=float)
    y = pd.Series(y).interpolate(limit_direction="both").to_numpy(dtype=float)
    return np.interp(tq, t, y, left=np.nan, right=np.nan)


def heading_deg(dx: np.ndarray, dy: np.ndarray) -> np.ndarray:
    return np.degrees(np.arctan2(dx, dy))


def wrap180(x: np.ndarray) -> np.ndarray:
    return (x + 180.0) % 360.0 - 180.0


def stats(x: np.ndarray) -> dict[str, float]:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return {"n": 0, "mean": float("nan"), "median": float("nan"), "p90": float("nan"), "p95": float("nan"), "max": float("nan")}
    return {
        "n": int(len(x)),
        "mean": float(np.mean(x)),
        "median": float(np.median(x)),
        "p90": float(np.percentile(x, 90)),
        "p95": float(np.percentile(x, 95)),
        "max": float(np.max(x)),
    }


def evaluate_for_duration(
    t: np.ndarray,
    g_e: np.ndarray,
    g_n: np.ndarray,
    e_e: np.ndarray,
    e_n: np.ndarray,
    motion_context: np.ndarray,
    duration_s: float,
) -> dict[str, object]:
    rows: list[dict[str, float]] = []
    n = len(t)
    for i in range(n):
        t_end = t[i] + duration_s
        j = int(np.searchsorted(t, t_end, side="left"))
        if j <= i or j >= n:
            continue
        dge = g_e[j] - g_e[i]
        dgn = g_n[j] - g_n[i]
        dee = e_e[j] - e_e[i]
        den = e_n[j] - e_n[i]
        err = float(np.hypot((dee - dge), (den - dgn)))
        hd_g = float(heading_deg(np.array([dge]), np.array([dgn]))[0])
        hd_e = float(heading_deg(np.array([dee]), np.array([den]))[0])
        hd_err = float(abs(wrap180(np.array([hd_e - hd_g]))[0]))
        moving_ratio = float(np.mean(motion_context[i : j + 1]))
        rows.append(
            {
                "start_idx": float(i),
                "end_idx": float(j),
                "start_t": float(t[i]),
                "end_t": float(t[j]),
                "window_s": float(t[j] - t[i]),
                "pos_err_m": err,
                "heading_err_deg": hd_err,
                "moving_ratio": moving_ratio,
            }
        )
    if not rows:
        return {"windows": 0, "pos_err_m": stats(np.array([])), "heading_err_deg": stats(np.array([])), "moving_ratio": stats(np.array([]))}
    df = pd.DataFrame(rows)
    return {
        "windows": int(len(df)),
        "pos_err_m": stats(df["pos_err_m"].to_numpy(dtype=float)),
        "heading_err_deg": stats(df["heading_err_deg"].to_numpy(dtype=float)),
        "moving_ratio": stats(df["moving_ratio"].to_numpy(dtype=float)),
        "rows": df,
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate GNSS-denied windows (10/20/30s by default) for a segment.")
    p.add_argument("--segment", default="seg_20260315_full_part04")
    p.add_argument("--gnss-fix-csv", default=None, help="Override gnss_raw_fix csv path")
    p.add_argument("--est-csv", default=None, help="Override estimated trajectory csv path")
    p.add_argument("--diag-dir", default=None, help="Override output diagnose directory")
    p.add_argument("--durations", default="10,20,30", help="Comma-separated durations in seconds")
    args = p.parse_args()

    root = Path(__file__).resolve().parents[1]
    seg = args.segment
    out_dir = root / "PDR" / "output" / f"{seg}_three_impl"
    diag_dir = Path(args.diag_dir).resolve() if args.diag_dir else (out_dir / "diagnose_zupt_anomaly")
    diag_dir.mkdir(parents=True, exist_ok=True)

    gnss_fix_csv = Path(args.gnss_fix_csv).resolve() if args.gnss_fix_csv else (root / "data" / "pdr_input" / seg / "gnss_raw_fix_part04.csv")
    est_csv = Path(args.est_csv).resolve() if args.est_csv else (out_dir / "zupt_aided_mag_gnss_pdr_traj_vel.csv")
    gnss = pd.read_csv(gnss_fix_csv).sort_values("unix_sec")
    est = pd.read_csv(est_csv)

    gt = gnss["unix_sec"].to_numpy(dtype=float)
    ge = gnss["east_m"].to_numpy(dtype=float)
    gn = gnss["north_m"].to_numpy(dtype=float)

    t = gt
    ee = interp(est["t_sec_abs"].to_numpy(dtype=float), est["pos_x_m"].to_numpy(dtype=float), t)
    en = interp(est["t_sec_abs"].to_numpy(dtype=float), est["pos_y_m"].to_numpy(dtype=float), t)

    if "motion_context" in est.columns:
        mc = interp(est["t_sec_abs"].to_numpy(dtype=float), est["motion_context"].to_numpy(dtype=float), t) > 0.5
    else:
        mc = np.ones_like(t, dtype=bool)

    durations = [float(x) for x in args.durations.split(",") if x.strip()]
    summary: dict[str, object] = {
        "segment": seg,
        "durations_s": durations,
        "inputs": {
            "gnss_fix_csv": str(gnss_fix_csv),
            "est_csv": str(est_csv),
        },
        "metrics": {},
    }
    csv_rows: list[dict[str, float]] = []
    for d in durations:
        ans = evaluate_for_duration(t, ge, gn, ee, en, mc, d)
        summary["metrics"][f"{int(d)}s"] = {
            "windows": ans["windows"],
            "pos_err_m": ans["pos_err_m"],
            "heading_err_deg": ans["heading_err_deg"],
            "moving_ratio": ans["moving_ratio"],
        }
        csv_rows.append(
            {
                "duration_s": d,
                "windows": float(ans["windows"]),
                "pos_err_mean_m": float(ans["pos_err_m"]["mean"]),
                "pos_err_p95_m": float(ans["pos_err_m"]["p95"]),
                "pos_err_max_m": float(ans["pos_err_m"]["max"]),
                "heading_err_mean_deg": float(ans["heading_err_deg"]["mean"]),
                "heading_err_p95_deg": float(ans["heading_err_deg"]["p95"]),
            }
        )

    out_json = diag_dir / "gnss_denied_window_eval_part04.json"
    out_csv = diag_dir / "gnss_denied_window_eval_part04.csv"
    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    pd.DataFrame(csv_rows).to_csv(out_csv, index=False)
    print(json.dumps({"json": str(out_json), "csv": str(out_csv)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
