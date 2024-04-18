#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("AT50_e.csv")

plt.figure(figsize=(12, 8))

# 'X' 값에 대한 그래프
plt.subplot(3, 1, 1)
plt.plot(df['Time'], df['X'], label='X', color='red')
plt.title('Time vs X')
plt.xlabel('Time')
plt.ylabel('X Value')
plt.grid(True)

# 'Y' 값에 대한 그래프
plt.subplot(3, 1, 2)
plt.plot(df['Time'], df['Y'], label='Y', color='green')
plt.title('Time vs Y')
plt.xlabel('Time')
plt.ylabel('Y Value')
plt.grid(True)

# 'Z' 값에 대한 그래프
plt.subplot(3, 1, 3)
plt.plot(df['Time'], df['Z'], label='Z', color='blue')
plt.title('Time vs Z')
plt.xlabel('Time')
plt.ylabel('Z Value')
plt.grid(True)

plt.tight_layout()
plt.show()

# %%
