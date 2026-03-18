import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path
import csv

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import imufusion


def load_capture_csv(csv_path):
    df = pd.read_csv(csv_path, index_col=[0], header=[0, 1])
    return df


def compute_quaternions_offline(df, frequency=200):
    ahrs = imufusion.Ahrs()
    
    Q_list = []
    
    def update(x):
        ahrs.update_no_magnetometer(x['gyro'].to_numpy(), x['accel'].to_numpy(), 0.005)
        Q = ahrs.quaternion.wxyz
        return Q
    
    sf = df.apply(update, axis=1)
    
    return np.array(list(sf))


def save_with_capture_format(df, Q_array, output_path, original_time):
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        
        writer.writerow(["", "accel", "accel", "accel", "gyro", "gyro", "gyro", "mag", "mag", "mag", "Q", "Q", "Q", "Q"])
        writer.writerow(["", "x", "y", "z", "x", "y", "z", "x", "y", "z", "w", "x", "y", "z"])
        writer.writerow(["time", "", "", "", "", "", "", "", "", "", "", "", "", ""])
        
        for idx in range(len(df)):
            row = [
                original_time[idx],
                df[("accel", "x")].iloc[idx],
                df[("accel", "y")].iloc[idx],
                df[("accel", "z")].iloc[idx],
                df[("gyro", "x")].iloc[idx],
                df[("gyro", "y")].iloc[idx],
                df[("gyro", "z")].iloc[idx],
                df[("mag", "x")].iloc[idx],
                df[("mag", "y")].iloc[idx],
                df[("mag", "z")].iloc[idx],
                Q_array[idx, 0],
                Q_array[idx, 1],
                Q_array[idx, 2],
                Q_array[idx, 3],
            ]
            writer.writerow(row)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="离线计算四元数 Q，验证 pdr.py 的计算方法"
    )
    parser.add_argument(
        "--csv", 
        required=True, 
        help="输入的 capture CSV 文件路径"
    )
    parser.add_argument(
        "--output-dir", 
        default=str(REPO_ROOT / "capture"),
        help="输出目录 (默认: offline_q_output)"
    )
    parser.add_argument(
        "--frequency", 
        type=float, 
        default=200.0,
        help="采样频率 Hz (默认: 200)"
    )
    
    args = parser.parse_args()
    
    csv_path = Path(args.csv).resolve()
    if not csv_path.exists():
        print(f"错误: 文件不存在: {csv_path}")
        return 1
    
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    
    fn = csv_path.stem
    print(f"正在处理: {csv_path.name}")
    print(f"文件名将是: fill_q_{fn}.csv")
    
    df = load_capture_csv(csv_path)
    print(f"原始数据形状: {df.shape}")
    
    original_time = df.index.to_numpy()
    
    orig_q = None
    if ("Q", "w") in df.columns:
        orig_q = df[[("Q", "w"), ("Q", "x"), ("Q", "y"), ("Q", "z")]].to_numpy()
        print(f"原始 Q 列已读取")
    else:
        print("注意: 原始 CSV 中没有 Q 列")
    
    print(f"正在使用 imufusion 计算四元数...")
    Q_array = compute_quaternions_offline(df, frequency=args.frequency)
    print(f"计算完成")
    
    output_path = output_dir / f"fill_q_{fn}.csv"
    save_with_capture_format(df, Q_array, output_path, original_time)
    print(f"结果已保存到: {output_path}")
    
    if orig_q is not None:
        diff = np.abs(orig_q - Q_array).mean()
        print(f"\n原始 Q 与计算 Q 的平均差异: {diff:.6f}")
        if diff < 1e-3:
            print("✓ 验证通过: 计算的 Q 与原始 Q 一致")
        else:
            print("⚠ 注意: 存在较大差异")
    
    return 0


if __name__ == "__main__":
    exit(main())
