#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

def gcd(a, b):
    while b != 0:
        a, b = b, a % b
    return a

def interpolate(df, kind):
    temp_df = df.copy()
    for column in ['X', 'Y', 'Z']:
        valid = ~temp_df[column].isnull()
        if valid.any():
            interpolator = interp1d(temp_df.loc[valid, 'Time'], temp_df.loc[valid, column], 
                                    kind=kind, fill_value='extrapolate')
            temp_df.loc[:, column] = interpolator(df['Time'])
    
    return temp_df

def main():
    df = pd.read_csv("AT50_e.csv")

    df_filtered_origin = df.loc[~((df['Time'] > 20.) & (df['Time'] < 50.) |
                                  (df['Time'] > 130.) & (df['Time'] < 170.) |
                                  (df['Time'] > 230.) & (df['Time'] < 260.))]

    blanc_mask = ~np.in1d(np.arange(df_filtered_origin.index[-1]+1), df_filtered_origin.index)

    df_filtered = df_filtered_origin.copy()
    df_filtered = df_filtered.reset_index(drop=True)

    time_square = max(len(str(time).split('.')[1]) for time in df_filtered.Time if '.' in str(time))

    df_filtered.Time = (df_filtered.Time * 10 ** time_square).round().astype(int)

    interval_list = [df_filtered.Time[i] - df_filtered.Time[i-1] for i in range(1, len(df_filtered))]
    
    temp = interval_list[0]
    for i in range(1, len(interval_list)):
        temp = gcd(temp, interval_list[i])

    interval = temp

    time_cursor = df_filtered.Time[0]
    last_time = df_filtered.iloc[-1].Time
    time_set = set(df_filtered['Time'])
    while last_time != time_cursor:
        time_cursor += interval
        if time_cursor not in time_set:
            temp_df = pd.DataFrame([time_cursor], columns=['Time'])
            df_filtered = pd.concat([df_filtered, temp_df])

    df_filtered = df_filtered.sort_values(by='Time').reset_index(drop=True)

    kinds = ['linear', 'nearest', 'slinear', 'quadratic', 'cubic']
    results = {}

    for kind in kinds:
        results[kind] = interpolate(df_filtered, kind)

    df_filtered2 = df_filtered.copy()

    # 각 보간 방법에 따른 X 값의 오프셋을 설정
    offsets = {
        'origin': 0,
        'linear': 800000,    
        'nearest': 1600000,
        'quadratic': 2400000,
        'cubic': 3200000,
        'extrapolate': 4000000
    }

    linear_interpolated = results['linear']
    nearest_interpolated = results['nearest']
    slinear_interpolated = results['slinear']
    quadratic_interpolated = results['quadratic']
    cubic_interpolated = results['cubic']

    empty_idx = df_filtered[df_filtered['X'].isna()].index
    empty_idx1 = empty_idx[:29]
    empty_idx2 = empty_idx[30:68]
    empty_idx3 = empty_idx[69:]

    # 3D 그래프를 그립니다.
    fig = plt.figure(figsize=(16, 7))
    ax = fig.add_subplot(111, projection='3d')

    ax.plot3D(df_filtered['X'], df_filtered['Y'], df_filtered['Z'], 'black', label='origin')
    ax.plot3D(linear_interpolated['X'] + offsets['linear'], linear_interpolated['Y'], linear_interpolated['Z'], 'orange', label='linear')
    ax.plot3D(linear_interpolated['X'].iloc[empty_idx1] + offsets['linear'], linear_interpolated['Y'].iloc[empty_idx1], linear_interpolated['Z'].iloc[empty_idx1], 'red', linewidth='2')
    ax.plot3D(linear_interpolated['X'].iloc[empty_idx2] + offsets['linear'], linear_interpolated['Y'].iloc[empty_idx2], linear_interpolated['Z'].iloc[empty_idx2], 'red', linewidth='2')
    ax.plot3D(linear_interpolated['X'].iloc[empty_idx3] + offsets['linear'], linear_interpolated['Y'].iloc[empty_idx3], linear_interpolated['Z'].iloc[empty_idx3], 'red', linewidth='2')

    ax.plot3D(nearest_interpolated['X'] + offsets['nearest'], nearest_interpolated['Y'], nearest_interpolated['Z'], 'green', label='nearest')
    ax.plot3D(nearest_interpolated['X'].iloc[empty_idx1] + offsets['nearest'], nearest_interpolated['Y'].iloc[empty_idx1], nearest_interpolated['Z'].iloc[empty_idx1], 'red', linewidth='2')
    ax.plot3D(nearest_interpolated['X'].iloc[empty_idx2] + offsets['nearest'], nearest_interpolated['Y'].iloc[empty_idx2], nearest_interpolated['Z'].iloc[empty_idx2], 'red', linewidth='2')
    ax.plot3D(nearest_interpolated['X'].iloc[empty_idx3] + offsets['nearest'], nearest_interpolated['Y'].iloc[empty_idx3], nearest_interpolated['Z'].iloc[empty_idx3], 'red', linewidth='2')

    ax.plot3D(quadratic_interpolated['X'] + offsets['quadratic'], quadratic_interpolated['Y'], quadratic_interpolated['Z'], 'blue', label='quadratic')
    ax.plot3D(quadratic_interpolated['X'].iloc[empty_idx1] + offsets['quadratic'], quadratic_interpolated['Y'].iloc[empty_idx1], quadratic_interpolated['Z'].iloc[empty_idx1], 'red', linewidth='2')
    ax.plot3D(quadratic_interpolated['X'].iloc[empty_idx2] + offsets['quadratic'], quadratic_interpolated['Y'].iloc[empty_idx2], quadratic_interpolated['Z'].iloc[empty_idx2], 'red', linewidth='2')
    ax.plot3D(quadratic_interpolated['X'].iloc[empty_idx3] + offsets['quadratic'], quadratic_interpolated['Y'].iloc[empty_idx3], quadratic_interpolated['Z'].iloc[empty_idx3], 'red', linewidth='2')

    ax.plot3D(cubic_interpolated['X'] + offsets['cubic'], cubic_interpolated['Y'], cubic_interpolated['Z'], 'purple', label='cubic')
    ax.plot3D(cubic_interpolated['X'].iloc[empty_idx1] + offsets['cubic'], cubic_interpolated['Y'].iloc[empty_idx1], cubic_interpolated['Z'].iloc[empty_idx1], 'red', linewidth='2')
    ax.plot3D(cubic_interpolated['X'].iloc[empty_idx2] + offsets['cubic'], cubic_interpolated['Y'].iloc[empty_idx2], cubic_interpolated['Z'].iloc[empty_idx2], 'red', linewidth='2')
    ax.plot3D(cubic_interpolated['X'].iloc[empty_idx3] + offsets['cubic'], cubic_interpolated['Y'].iloc[empty_idx3], cubic_interpolated['Z'].iloc[empty_idx3], 'red', linewidth='2')

    ax.plot3D(df_filtered2['X'] + offsets['extrapolate'], df_filtered2['Y'], df_filtered2['Z'], '#4B9DFA', label='extrapolate')

    ax.set_xlabel('X Value')
    ax.set_ylabel('Y Value')
    ax.set_zlabel('Z Value')
    ax.legend()

    plt.show()

if __name__ == "__main__":
    main()
    plt.show()

# %%
