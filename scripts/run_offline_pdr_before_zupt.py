import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path
import csv
import itertools
import logging
from scipy import signal

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import imufusion


def load_calib_csv(calib_path):
    try:
        calib = pd.read_csv(
            calib_path, sep="\t", header=[0, 1], index_col=[0, 1]
        ).T
        print(f"已读取校准文件: {calib_path}")
        return calib
    except Exception as e:
        print(f"警告: 无法读取校准文件 {calib_path}: {e}")
        print("使用默认校准参数（单位矩阵 + 零偏移）")
        
        columns = tuple(
            itertools.product(("accel", "gyro", "mag"), ("x", "y", "z"))
        )
        index = tuple(itertools.product(("A"), ("x", "y", "z"))) + (
            ("b", "b"),
        )
        data = np.array([np.eye(3)] * 3).reshape(9, 3).T
        data = np.append(data, np.zeros(shape=(1, 9)), axis=0)
        calib = pd.DataFrame(data, columns=columns, index=index)
        return calib


def load_capture_csv(csv_path):
    df = pd.read_csv(csv_path, index_col=[0], header=[0, 1])
    return df


def apply_calibration(df, calib_b, calib_A):
    df_calib = df.copy()
    
    df_calib -= calib_b
    
    df_calib["mag"] = (calib_A["mag"] @ df_calib["mag"].T).T
    
    return df_calib


def apply_lowpass_filter(df, cutoff=5, order=10, fs=200):
    b, a = signal.butter(order, cutoff, fs=fs, btype='lowpass', analog=False)
    df_filtered = df.apply(lambda x: signal.filtfilt(b, a, x))
    return df_filtered


def compute_quaternions_offline(df, frequency=200):
    ahrs = imufusion.Ahrs()
    
    Q_list = []
    
    def update(x):
        ahrs.update_no_magnetometer(x['gyro'].to_numpy(), x['accel'].to_numpy(), 0.005)
        Q = ahrs.quaternion.wxyz
        return Q
    
    sf = df.apply(update, axis=1)
    
    return np.array(list(sf))


def save_with_capture_format(df_original, df_processed, Q_array, output_path, original_time):
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        
        writer.writerow(["", "accel", "accel", "accel", "gyro", "gyro", "gyro", "mag", "mag", "mag", "Q", "Q", "Q", "Q"])
        writer.writerow(["", "x", "y", "z", "x", "y", "z", "x", "y", "z", "w", "x", "y", "z"])
        writer.writerow(["time", "", "", "", "", "", "", "", "", "", "", "", "", ""])
        
        for idx in range(len(df_original)):
            row = [
                original_time[idx],
                df_processed[("accel", "x")].iloc[idx],
                df_processed[("accel", "y")].iloc[idx],
                df_processed[("accel", "z")].iloc[idx],
                df_processed[("gyro", "x")].iloc[idx],
                df_processed[("gyro", "y")].iloc[idx],
                df_processed[("gyro", "z")].iloc[idx],
                df_processed[("mag", "x")].iloc[idx],
                df_processed[("mag", "y")].iloc[idx],
                df_processed[("mag", "z")].iloc[idx],
                Q_array[idx, 0],
                Q_array[idx, 1],
                Q_array[idx, 2],
                Q_array[idx, 3],
            ]
            writer.writerow(row)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="离线 PDR 处理，支持读取校准参数"
    )
    parser.add_argument(
        "--csv", 
        required=True, 
        help="输入的 capture CSV 文件路径"
    )
    parser.add_argument(
        "--calib", 
        default=str(REPO_ROOT / "calib.csv"),
        help="校准文件 calib.csv 路径 (默认: calib.csv)"
    )
    parser.add_argument(
        "--output-dir", 
        default=str(REPO_ROOT / "output"),
        help="输出目录 (默认: output)"
    )
    parser.add_argument(
        "--frequency", 
        type=float, 
        default=200.0,
        help="采样频率 Hz (默认: 200)"
    )
    parser.add_argument(
        "--lp", 
        action="store_true",
        help="启用低通滤波 (默认: 不启用)"
    )
    parser.add_argument(
        "--cutoff", 
        type=float, 
        default=5.0,
        help="低通滤波截止频率 Hz (默认: 5)"
    )
    parser.add_argument(
        "--order", 
        type=int, 
        default=10,
        help="低通滤波器阶数 (默认: 10)"
    )
    
    args = parser.parse_args()
    
    csv_path = Path(args.csv).resolve()
    if not csv_path.exists():
        print(f"错误: 文件不存在: {csv_path}")
        return 1
    
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("离线 PDR 处理")
    print("=" * 60)
    
    calib_path = Path(args.calib).resolve()
    calib = load_calib_csv(calib_path)
    
    calib_b = calib.loc["b", "b"]
    calib_A = calib.loc["A"]
    
    print(f"\n校准参数:")
    print(f"  偏移量 (calib_b):")
    print(f"    accel: {calib_b['accel'].tolist()}")
    print(f"    gyro:  {calib_b['gyro'].tolist()}")
    print(f"    mag:   {calib_b['mag'].tolist()}")
    print(f"\n  缩放矩阵 (calib_A):")
    print(f"    accel:\n{calib_A['accel'].to_string()}")
    print(f"    gyro:\n{calib_A['gyro'].to_string()}")
    print(f"    mag:\n{calib_A['mag'].to_string()}")
    
    fn = csv_path.stem
    print(f"\n正在处理: {csv_path.name}")
    print(f"文件名将是: fill_q_{fn}.csv")
    
    df = load_capture_csv(csv_path)
    print(f"原始数据形状: {df.shape}")
    
    print("\n正在应用校准参数...")
    df_calib = apply_calibration(df, calib_b, calib_A)
    print("校准参数已应用")
    
    df_processed = df_calib
    if args.lp:
        print(f"\n正在应用低通滤波 (cutoff={args.cutoff}Hz, order={args.order})...")
        df_processed = apply_lowpass_filter(df_calib, cutoff=args.cutoff, order=args.order, fs=args.frequency)
        print("低通滤波已应用")
    
    original_time = df.index.to_numpy()
    
    orig_q = None
    if ("Q", "w") in df.columns:
        orig_q = df[[("Q", "w"), ("Q", "x"), ("Q", "y"), ("Q", "z")]].to_numpy()
        print(f"原始 Q 列已读取")
    else:
        print("注意: 原始 CSV 中没有 Q 列")
    
    print(f"正在使用 imufusion 计算四元数...")
    Q_array = compute_quaternions_offline(df_processed, frequency=args.frequency)
    print(f"计算完成")
    
    output_path = output_dir / f"fill_q_{fn}.csv"
    save_with_capture_format(df, df_processed, Q_array, output_path, original_time)
    print(f"结果已保存到: {output_path}")
    
    if orig_q is not None:
        diff = np.abs(orig_q - Q_array).mean()
        print(f"\n原始 Q 与计算 Q 的平均差异: {diff:.6f}")
        if diff < 1e-3:
            print("✓ 验证通过: 计算的 Q 与原始 Q 一致")
        else:
            print("⚠ 注意: 存在较大差异")
    
    print("\n" + "=" * 60)
    if args.lp:
        print("校准参数和低通滤波已应用到数据")
    else:
        print("校准参数已应用到数据")
    print("=" * 60)
    
    return 0


if __name__ == "__main__":
    exit(main())
