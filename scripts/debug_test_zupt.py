import pandas as pd
import numpy as np
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

fn = 'handheld_male_rect'

# 读取两个脚本的中间数据进行对比
print("=== 对比数据流程 ===")

# 读取原始数据
df_orig = pd.read_csv(f'capture/{fn}.csv', index_col=[0], header=[0,1])

print(f"\n原始数据前 5 行:")
print(df_orig[['accel', 'gyro']].head())

# run_zupt.py 的流程
print("\n=== run_zupt.py 流程 ===")
df_zupt = df_orig.copy()
df_zupt['gyro'] *= 180 / np.pi
df_zupt.index /= 200

from scipy import signal
b, a = signal.butter(10, 20, fs=200, btype='lowpass', analog=False)
df_zupt = df_zupt.apply(lambda x: signal.filtfilt(b, a, x))

print(f"\nrun_zupt.py 滤波后的前 5 行:")
print(df_zupt[['accel', 'gyro']].head())

# run_offline_pdr.py 的流程
print("\n=== run_offline_pdr.py 流程 ===")
df_offline = df_orig.copy()
df_offline['gyro'] *= 180 / np.pi
df_offline.index /= 200

# 应用校准
calib = pd.read_csv("calib.csv", sep="\t", header=[0, 1], index_col=[0, 1]).T
calib_b = calib.loc["b", "b"]
print(f"\ncalib_b:")
print(f"  accel: {calib_b['accel'].tolist()}")
print(f"  gyro:  {calib_b['gyro'].tolist()}")
print(f"  mag:   {calib_b['mag'].tolist()}")

df_calib = df_offline.copy()
df_calib -= calib_b

print(f"\n减去校准后的前 5 行:")
print(df_calib[['accel', 'gyro']].head())

# 对比差异
print("\n=== 陀螺仪差异 ===")
gyro_diff = df_zupt['gyro'] - df_calib['gyro']
print(f"平均差异: {gyro_diff.mean().tolist()}")
print(f"最大差异: {gyro_diff.abs().max().tolist()}")

print("\n结论: run_offline_pdr.py 减去了陀螺仪偏移，导致输入 AHRS 的数据与 run_zupt.py 不同！")
