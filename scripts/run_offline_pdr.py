import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path
import csv
import itertools
import logging
from scipy import signal
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

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
    df = pd.read_csv(csv_path, index_col=[0], header=[0, 1]).reset_index(drop=True)
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

def compute_ahrs_offline(df, frequency=200):
    ahrs = imufusion.Ahrs()
    sample_rate = frequency

    ahrs.settings = imufusion.Settings(imufusion.CONVENTION_NED, 5, 2000, 10, 10, 5 * 200)

    results = []
    
    def update(x):
        ahrs.update_no_magnetometer(x['gyro'].to_numpy(), x['accel'].to_numpy(), 0.005)
        euler = ahrs.quaternion.to_euler()
        Q = ahrs.quaternion.wxyz
        acceleration = ahrs.earth_acceleration
        
        ans = {}
        ans.update({'x': acceleration[0], 'y': acceleration[1], 'z': acceleration[2]})
        ans.update({'roll': euler[0], 'pitch': euler[1], 'yaw': euler[2]})
        ans.update({'Q_T': Q})
        ans.update({"accel_err": 0.0})
        ans.update({"accel_igr": False})
        ans.update({"accel_rec": 0})
        ans.update({"ang_rrec": False})
        ans.update({"accel_rrec": False})
        return ans
    print('--' * 100)
    print('df range', df.index.min(), df.index.max())
    print('--' * 100)
    sf = df.apply(update, axis=1)
    
    return pd.DataFrame(list(sf), index=df.index)

def compute_quaternions_offline(df, frequency=200):
    ahrs = imufusion.Ahrs()
    
    Q_list = []
    
    def update(x):
        ahrs.update_no_magnetometer(x['gyro'].to_numpy(), x['accel'].to_numpy(), 0.005)
        Q = ahrs.quaternion.wxyz
        return Q
    
    sf = df.apply(update, axis=1)
    
    return np.array(list(sf))

def run_zupt_analysis(sf, df, frequency=200, zupt_tresh=3, margin=0.15):
    sample_rate = frequency
    dt = 1 / sample_rate
    margin_samples = int(margin * sample_rate)
    
    hf = sf[['x', 'y', 'z']].to_numpy()
    cols = itertools.product(['acceleration'], ('x', 'y', 'z'))
    hf = pd.DataFrame(hf, columns=pd.MultiIndex.from_tuples(cols))
    # hf = hf.reset_index(drop=True)
    
    g_end = np.linalg.norm(hf['acceleration'], axis=1)[-100:].mean()
    g_start = -hf['acceleration', 'z'][:100].mean()
    g = min(g_start, g_end)
    print(f'calculated g : {g}')
    
    hf['is_moving'] = hf['acceleration'].apply(np.linalg.norm, axis=1) > zupt_tresh + g
    
    for index in range(len(hf) - margin_samples):
        hf.loc[index, 'is_moving'] = any(hf.loc[index:(index + margin_samples), 'is_moving'])
    
    for index in range(len(hf) - 1, margin_samples, -1):
        hf.loc[index, 'is_moving'] = any(hf.loc[(index - margin_samples):index, 'is_moving'])
    
    peaks, _ = find_peaks(hf['is_moving'].astype(int))
    steps = len(peaks)
    print(f'Detected {steps} steps')
    
    velocity = np.zeros((len(hf), 3))
    cols = pd.MultiIndex.from_product([['velocity'], ['x', 'y', 'z']])
    hf[cols] = hf['acceleration'] * dt
    for idx in range(1, len(hf)):
        if hf.loc[idx, 'is_moving']:
            velocity[idx] = velocity[idx - 1] + hf.loc[idx, 'velocity']
    
    hf['velocity'] = velocity
    
    is_moving_diff = hf['is_moving'].astype(int).diff().fillna(0)
    idx_shift_diff = is_moving_diff[is_moving_diff < 0].index
    is_moving_diff[idx_shift_diff] = 0
    is_moving_diff[idx_shift_diff - 1] = 1
    is_moving_diff = is_moving_diff.astype(bool)
    
    cols = pd.MultiIndex.from_product([['velocity_drift'], ['x', 'y', 'z']])
    hf[cols] = hf['velocity'].apply(lambda x: x * is_moving_diff)
    idx_to_interp = hf[hf['is_moving']]['is_moving'].index.symmetric_difference(is_moving_diff[is_moving_diff].index)
    hf.loc[idx_to_interp, 'velocity_drift'] = np.nan
    hf['velocity_drift'] = hf['velocity_drift'].interpolate()
    hf['velocity'] = hf['velocity'] - hf['velocity_drift']
    
    cols = pd.MultiIndex.from_product([['position'], ['x', 'y', 'z']])
    hf[cols] = hf['velocity'] * dt
    pos = np.zeros((len(hf), 3))
    for idx in range(1, len(hf)):
        pos[idx] = pos[idx - 1] + hf.loc[idx, 'position']
    
    hf['position'] = pos
    
    return hf, steps, idx_shift_diff


def plot_ypr(sf, df, output_dir, fn):
    plt.style.use('default')
    
    fig, ax = plt.subplots(nrows=6, sharex=True, figsize=(20,15), tight_layout=True, gridspec_kw={"height_ratios": [6, 6, 6, 2, 1, 1]})
    
    ax[0].plot(df.index, df['gyro', 'x'], "tab:red", label='gyro x')
    ax[0].plot(df.index, df['gyro', 'y'], "tab:green", label='gyro y')
    ax[0].plot(df.index, df['gyro', 'z'], "tab:blue", label='gyro z')
    ax[0].set_ylabel('Degrees/s')
    ax[0].set_title('Gyroscope')
    ax[0].legend()
    ax[0].grid()
    
    ax[1].plot(df.index, df['accel', 'x'], "tab:red", label='Accelerometer x')
    ax[1].plot(df.index, df['accel', 'y'], "tab:green", label='Accelerometer y')
    ax[1].plot(df.index, df['accel', 'z'], "tab:blue", label='Accelerometer z')
    ax[1].set_ylabel('Acceleration [m/s/s]')
    ax[1].set_title('Accelerometer')
    ax[1].legend()
    ax[1].grid()
    
    ax[2].plot(sf['yaw'], "tab:red", label='Yaw')
    ax[2].plot(sf['pitch'], "tab:green", label='Pitch')
    ax[2].plot(sf['roll'], "tab:blue", label='Roll')
    ax[2].set_ylabel('Degrees')
    ax[2].grid()
    ax[2].legend()
    
    ax[3].plot(sf['accel_err'], "tab:olive", label='Acceleration error')
    ax[3].set_ylabel('Error')
    ax[3].set_xlabel('Time [s]')
    ax[3].legend()
    
    ax[4].plot(sf['accel_igr'], "tab:cyan", label='Acceleration ignored')
    
    ax[5].plot(sf['accel_rec'], "tab:orange", label='Acceleration recovery trigger')
    
    for axes in ax:
        axes.set_xlim(0, sf.index.max())
    
    fig.savefig(output_dir / f'ypr_{fn}.png', dpi=300)
    plt.close()
    print("Saved ypr.png")


def plot_zupt_path(hf, steps, idx_shift_diff, output_dir, fn):
    fig, ax = plt.subplots(nrows=5, sharex=True, figsize=(20,10), tight_layout=True, gridspec_kw={"height_ratios": [6, 1, 6, 6, 6]})
    
    ax[0].plot(hf['acceleration', 'x'], "tab:red", label="X")
    ax[0].plot(hf['acceleration', 'y'], "tab:green", label="Y")
    ax[0].plot(hf['acceleration', 'z'], "tab:blue", label="Z")
    ax[0].set_title("Acceleration")
    ax[0].set_ylabel("m/s/s")
    ax[0].grid()
    ax[0].legend()
    
    ax[1].plot(hf['is_moving'], "tab:cyan", label="Is moving")
    ax[1].grid()
    ax[1].legend()
    
    ax[2].plot(hf['velocity', 'x'], "tab:red", label="X")
    ax[2].plot(hf['velocity', 'y'], "tab:green", label="Y")
    ax[2].plot(hf['velocity', 'z'], "tab:blue", label="Z")
    ax[2].set_title("Velocity")
    ax[2].set_ylabel("m/s")
    ax[2].grid()
    ax[2].legend()
    
    ax[3].plot(hf['velocity_drift', 'x'], "tab:red", label="X")
    ax[3].plot(hf['velocity_drift', 'y'], "tab:green", label="Y")
    ax[3].plot(hf['velocity_drift', 'z'], "tab:blue", label="Z")
    ax[3].set_title("Velocity Drift")
    ax[3].set_ylabel("m/s")
    ax[3].grid()
    ax[3].legend()
    
    ax[4].plot(hf['position', 'x'], "tab:red", label="X")
    ax[4].plot(hf['position', 'y'], "tab:green", label="Y")
    ax[4].plot(hf['position', 'z'], "tab:blue", label="Z")
    ax[4].set_title("Position")
    ax[4].set_ylabel("m")
    ax[4].grid()
    ax[4].legend()
    
    fig.savefig(output_dir / f'path_{steps}_{fn}.png', dpi=300)
    plt.close()
    print(f"Saved path_{steps}_{fn}.png")


def plot_2d_path(hf, idx_shift_diff, output_dir, fn):
    fig, axes = plt.subplots(nrows=1, figsize=(10,10))
    
    axes.plot(hf['position', 'x'], hf['position', 'y'])
    axes.scatter(hf.loc[idx_shift_diff, 'position']['x'], hf.loc[idx_shift_diff, 'position']['y'], color='red', label='steps')
    axes.set_xlabel('X [m]')
    axes.set_ylabel('Y [m]')
    axes.set_title('position 2D')
    axes.legend()
    axes.grid()
    
    fig.savefig(output_dir / f'path2D_{fn}.png', dpi=300)
    plt.close()
    print(f"Saved path2D_{fn}.png")


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
    parser.add_argument(
        "--zupt", 
        action="store_true",
        help="启用零速检测 (默认: 不启用)"
    )
    parser.add_argument(
        "--zupt-tresh", 
        type=float, 
        default=3.0,
        help="零速检测阈值 (默认: 3)"
    )
    parser.add_argument(
        "--zupt-margin", 
        type=float, 
        default=0.15,
        help="零速检测边距秒数 (默认: 0.15)"
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
    df['gyro'] *= -1
    df['accel'] *= -1
    df_for_save = df.copy()
    print(f"原始数据形状: {df.shape}")
    print('--' * 25, '原始数据', '--' * 50 )
    print('df range', df.index.min(), df.index.max())
    print('--' * 100)

    # TODO 
    sample_rate = args.frequency
    df['gyro'] *= 180 / np.pi
    df.index /= sample_rate
    print('--' * 25, '校准前', '--' * 50)
    print('df range', df.index.min(), df.index.max())
    print('--' * 100)
    print("\n正在应用校准参数...")
    df_calib = apply_calibration(df, calib_b, calib_A)

    print('--' * 25, '校准后', '--' * 50)
    print('df_calib range', df_calib.index.min(), df_calib.index.max())
    print('--' * 100)
    print("校准参数已应用")
    
    df_processed = df_calib
    if args.lp:
        print(f"\n正在应用低通滤波 (cutoff={args.cutoff}Hz, order={args.order})...")
        df_processed = apply_lowpass_filter(df_calib, cutoff=args.cutoff, order=args.order, fs=args.frequency)
        print("低通滤波已应用")

        print('--' * 25, '低通滤波后', '--' * 50)
        print('df_processed range', df_processed.index.min(), df_processed.index.max())
        print('--' * 100)
        print("低通滤波已应用") 

    original_time = df.index.to_numpy()
    
    orig_q = None
    if ("Q", "w") in df.columns:
        orig_q = df[[("Q", "w"), ("Q", "x"), ("Q", "y"), ("Q", "z")]].to_numpy()
        print(f"原始 Q 列已读取")
    else:
        print("注意: 原始 CSV 中没有 Q 列")
    
    print(f"正在使用 imufusion 计算四元数...")
    # Q_array = compute_quaternions_offline(df_processed, frequency=args.frequency)
    
    sf = compute_ahrs_offline(df_processed, frequency=args.frequency)
    Q_array = np.array([q for q in sf['Q_T']])
    
    print(f"计算完成")
    
    output_path = output_dir / f"fill_q_{fn}.csv"
    save_with_capture_format(df, df_for_save, Q_array, output_path, original_time)
    print(f"结果已保存到: {output_path}")
    
    if args.zupt:
        print(f"\n正在绘制 YPR 图..., len(sf), len(df), sf.index.max", len(sf), len(df), sf.index.max())

        print('--' * 100)
        print('sf range', sf.index.min(), sf.index.max())

        print('yaw', sf['yaw'].values[-10:])
        print('roll', sf['yaw'].values[-10:])
        print('pitch', sf['yaw'].values[-10:])
        print('--' * 100)

        plot_ypr(sf, df_processed, output_dir, fn)
        print(f"\n正在运行零速检测 (thresh={args.zupt_tresh}, margin={args.zupt_margin}s)...")
        hf, steps, idx_shift_diff = run_zupt_analysis(sf, df_processed, frequency=args.frequency, zupt_tresh=args.zupt_tresh, margin=args.zupt_margin)

        print(f"零速检测完成，检测到 {steps} 步")
        
        
        print(f"\n正在绘制路径图...")
        plot_zupt_path(hf, steps, idx_shift_diff, output_dir, fn)
        
        print(f"\n正在绘制 2D 路径图...")
        plot_2d_path(hf, idx_shift_diff, output_dir, fn)
        
        print(f"\n生成的文件:")
        print(f"  - ypr.png")
        print(f"  - path_{steps}.png")
        print(f"  - path2D_{fn}.png")

    if orig_q is not None:
        diff = np.abs(orig_q - Q_array).mean()
        print(f"\n原始 Q 与计算 Q 的平均差异: {diff:.6f}")
        if diff < 1e-3:
            print("✓ 验证通过: 计算的 Q 与原始 Q 一致")
        else:
            print("⚠ 注意: 存在较大差异")
    
    print("\n" + "=" * 60)
    if args.lp and args.zupt:
        print("校准参数、低通滤波和零速检测已应用到数据")
    elif args.lp:
        print("校准参数和低通滤波已应用到数据")
    elif args.zupt:
        print("校准参数和零速检测已应用到数据")
    else:
        print("校准参数已应用到数据")
    print("=" * 60)
    
    return 0


if __name__ == "__main__":
    exit(main())
