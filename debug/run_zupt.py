import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools
import os
from scipy import signal
from scipy.signal import find_peaks
import imufusion

fn = 'handheld_male_rect'
fn = 'fill_q_handheld_nifei_rect'
fn = 'from_run_q_fill_q_handheld_nifei_rect'
fn = 'fill_q_handheld_male_rect'
fn = 'fill_q_zupt_handheld_male_rect'
fn = 'fill_q_zupt_handheld_male_rect_20hz'
fn = 'fill_q_handheld_male_rect_no_lp_no_zupt_dft'
fn = 'fill_q_handheld_nifei_init'
fn = 'handheld_nifei_init'

while ('offline_imgs' in os.getcwd()) or ('debug' in os.getcwd()):
    os.chdir("..")

df = pd.read_csv(f'capture/{fn}.csv', index_col=[0], header=[0,1]).reset_index(drop=True)
print('--' * 25, '原始数据', '--' * 50 )
print('df range', df.index.min(), df.index.max())
print('--' * 100)
df['gyro'] *= -1
df['accel'] *= -1
if not os.path.exists('./offline_imgs/' + fn):
    print('Creating folder ' + fn)
    os.makedirs('./offline_imgs/' + fn)

os.chdir(f"offline_imgs/{fn}")

df['gyro'] *= 180 / np.pi

sample_rate = 200
dt = 1/sample_rate
zupt_tresh = 3
margin = int(0.15 * sample_rate)
debug_ZUPT = False

df.index /= sample_rate

print("Filtering data...")
b, a = signal.butter(10, 20, fs=200, btype='lowpass', analog=False)
df = df.apply(lambda x: signal.filtfilt(b, a, x))

print("Running AHRS...")
offset = imufusion.Offset(sample_rate)
ahrs = imufusion.Ahrs()
ahrs.settings = imufusion.Settings(imufusion.CONVENTION_NED,
                                   5,
                                   2000,
                                   10,
                                   10,
                                   5 * sample_rate)

def update(x):
    ahrs.update_no_magnetometer(x['gyro'].to_numpy(), x['accel'].to_numpy(), 0.005)
    euler = ahrs.quaternion.to_euler()
    Q = ahrs.quaternion.wxyz
    acceleration = ahrs.earth_acceleration
    
    ans = {}
    ans.update({'x': acceleration[0], 'y':acceleration[1], 'z':acceleration[2]})
    ans.update({'roll': euler[0], 'pitch' : euler[1], 'yaw' : euler[2]})
    ans.update({'Q_T' : Q})
    ans.update({"accel_err" : ahrs.internal_states.acceleration_error})
    ans.update({"accel_igr" : ahrs.internal_states.accelerometer_ignored})
    ans.update({"accel_rec" : ahrs.internal_states.acceleration_recovery_trigger})
    ans.update({"ang_rrec" : ahrs.flags.angular_rate_recovery})
    ans.update({"accel_rrec" : ahrs.flags.acceleration_recovery})
    return ans

print('--' * 100)
print('df range', df.index.min(), df.index.max())
print('--' * 100)
sf = df.apply(update, axis=1)
sf = pd.DataFrame(list(sf), index=df.index)
print('--' * 100)
print('sf range', sf.index.min(), sf.index.max())
print('yaw', sf['yaw'].values[-10:])
print('roll', sf['yaw'].values[-10:])
print('pitch', sf['yaw'].values[-10:])
print('--' * 100)

print("Plotting YPR...", 'len(sf)', len(sf), 'len(df)', len(df), 'sf.index.max', sf.index.max())
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

fig.savefig(f'ypr.png',  dpi=300)
plt.close()
print("Saved ypr.png")

print("Running ZUPT...")
hf = sf[['x', 'y', 'z']].to_numpy()
cols = itertools.product(['acceleration'], ('x', 'y', 'z'))
hf = pd.DataFrame(hf, columns=pd.MultiIndex.from_tuples(cols))

g_end = np.linalg.norm(hf['acceleration'], axis=1)[-100:].mean()
g_start = -hf['acceleration', 'z'][:100].mean()
g = min(g_start, g_end)
print(f'calculated g : {g}')

hf['is_moving'] = hf['acceleration'].apply(np.linalg.norm, axis=1) > zupt_tresh + g

for index in range(len(hf) - margin):
    hf.loc[index, 'is_moving'] = any(hf.loc[index:(index + margin), 'is_moving'])

for index in range(len(hf) - 1, margin, -1):
    hf.loc[index, 'is_moving'] = any(hf.loc[(index - margin):index, 'is_moving'])

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
hf[cols] = hf['velocity'].apply(lambda x: x*is_moving_diff)
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

print("Plotting path...")
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

fig.savefig(f'path_{len(peaks)}.png', dpi=300)
plt.close()
print(f"Saved path_{len(peaks)}.png")

print("Plotting 2D path...")
fig, axes = plt.subplots(nrows=1, figsize=(10,10))

axes.plot(hf['position', 'x'], hf['position', 'y'])
axes.scatter(hf.loc[idx_shift_diff, 'position']['x'], hf.loc[idx_shift_diff, 'position']['y'], color='red', label='steps')
axes.set_xlabel('X [m]')
axes.set_ylabel('Y [m]')
axes.set_title('position 2D')
axes.legend()
axes.grid()

fig.savefig(f'path2D_{fn}.png', dpi=300)
plt.close()
print(f"Saved path2D_{fn}.png")

print("\nZUPT analysis complete!")
print("Generated files:")
print(f"  - ypr.png")
print(f"  - path_{len(peaks)}.png")
print(f"  - path2D_{fn}.png")
