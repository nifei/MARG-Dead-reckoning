import pandas as pd
import numpy as np
from pathlib import Path
import csv


def load_imu_log(imu_log_path):
    with open(imu_log_path, 'r') as f:
        lines = f.readlines()
    
    data_start = 0
    for i, line in enumerate(lines):
        if not line.startswith('*') and not line.startswith('N/A') and ',' in line:
            data_start = i
            break
    
    data_lines = lines[data_start:]
    
    imu_data = []
    for line in data_lines:
        line = line.strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split(',')]
        if len(parts) < 6:
            continue
        try:
            timestamp = int(parts[0])
            sensor_flag = int(parts[2])
            x = float(parts[3])
            y = float(parts[4])
            z = float(parts[5])
            imu_data.append((timestamp, sensor_flag, x, y, z))
        except (ValueError, IndexError):
            continue
    
    return imu_data


def merge_sensors_to_200hz(imu_data, max_seconds=None):
    accel_dict = {}
    gyro_dict = {}
    mag_dict = {}
    
    for ts, flag, x, y, z in imu_data:
        if flag == 1:
            accel_dict[ts] = (x, y, z)
        elif flag == 2:
            gyro_dict[ts] = (x, y, z)
        elif flag == 3:
            mag_dict[ts] = (x, y, z)
    
    all_timestamps = sorted(set(accel_dict.keys()) | set(gyro_dict.keys()) | set(mag_dict.keys()))
    
    if not all_timestamps:
        return []
    
    start_ts = all_timestamps[0]
    
    if max_seconds is not None:
        end_ts = start_ts + max_seconds * 1000
    else:
        end_ts = all_timestamps[-1]
    
    target_interval_ms = 5
    num_samples = int((end_ts - start_ts) / target_interval_ms) + 1
    
    merged_data = []
    
    for i in range(num_samples):
        target_ts = start_ts + i * target_interval_ms
        
        closest_ts_accel = min(accel_dict.keys(), key=lambda t: abs(t - target_ts), default=None)
        closest_ts_gyro = min(gyro_dict.keys(), key=lambda t: abs(t - target_ts), default=None)
        closest_ts_mag = min(mag_dict.keys(), key=lambda t: abs(t - target_ts), default=None)
        
        accel = accel_dict.get(closest_ts_accel, (0.0, 0.0, 0.0))
        gyro = gyro_dict.get(closest_ts_gyro, (0.0, 0.0, 0.0))
        mag = mag_dict.get(closest_ts_mag, (0.0, 0.0, 0.0))
        
        merged_data.append({
            'time': target_ts,
            'accel_x': accel[0],
            'accel_y': accel[1],
            'accel_z': accel[2],
            'gyro_x': gyro[0],
            'gyro_y': gyro[1],
            'gyro_z': gyro[2],
            'mag_x': mag[0],
            'mag_y': mag[1],
            'mag_z': mag[2]
        })
    
    return merged_data


def save_as_capture_format(merged_data, output_path):
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        
        writer.writerow(["", "accel", "accel", "accel", "gyro", "gyro", "gyro", "mag", "mag", "mag", "Q", "Q", "Q", "Q"])
        writer.writerow(["", "x", "y", "z", "x", "y", "z", "x", "y", "z", "w", "x", "y", "z"])
        writer.writerow(["time", "", "", "", "", "", "", "", "", "", "", "", "", ""])
        
        for row in merged_data:
            csv_row = [
                row['time'],
                row['accel_x'],
                row['accel_y'],
                row['accel_z'],
                row['gyro_x'],
                row['gyro_y'],
                row['gyro_z'],
                row['mag_x'],
                row['mag_y'],
                row['mag_z'],
                "",
                "",
                "",
                ""
            ]
            writer.writerow(csv_row)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="将 custom_data 中的 GNSS/IMU Logger 数据转换为 capture 格式"
    )
    parser.add_argument(
        "--imu-log", 
        required=True, 
        help="输入的 IMU 日志文件路径 (UIMU_Log/*.txt)"
    )
    parser.add_argument(
        "--output", 
        default="capture/handheld_nifei_rect.csv",
        help="输出文件路径 (默认: capture/handheld_nifei_rect.csv)"
    )
    parser.add_argument(
        "--max-seconds", 
        type=float, 
        default=None,
        help="只转换前 N 秒的数据 (默认: 全部)"
    )
    
    args = parser.parse_args()
    
    imu_log_path = Path(args.imu_log).resolve()
    if not imu_log_path.exists():
        print(f"错误: 文件不存在: {imu_log_path}")
        return 1
    
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"正在读取 IMU 日志: {imu_log_path.name}")
    imu_data = load_imu_log(imu_log_path)
    print(f"读取到 {len(imu_data)} 条数据")
    
    if args.max_seconds:
        print(f"只转换前 {args.max_seconds} 秒的数据")
    
    print(f"正在合并传感器数据为 200Hz...")
    merged_data = merge_sensors_to_200hz(imu_data, max_seconds=args.max_seconds)
    print(f"合并后 {len(merged_data)} 个样本")
    
    save_as_capture_format(merged_data, output_path)
    print(f"已保存到: {output_path}")
    
    return 0


if __name__ == "__main__":
    exit(main())
