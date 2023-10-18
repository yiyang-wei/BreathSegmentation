import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


file = "data/20220418_EVLP818_converted.csv"

df = pd.read_csv(file, header=0, sep=',')

df.index = df.index - df.index[0]

print(df.head())
print(df.columns) # ['Timestamp', 'Date', 'Time', 'B_phase', 'Pressure', 'Flow']
print(df.shape[0]) # 1155636

# part = df.iloc[3*6000:4*6000]
# part.plot(subplots=True)
# plt.savefig("fig/normal_breath.png")

# part = df.iloc[31*6000-500:31*6000+2500]
# part.plot(subplots=True)
# plt.savefig("fig/two_stage_exhaling.png")

# part = df.iloc[36*6000+500:36*6000+6500]
# part.plot(subplots=True)
# plt.savefig("fig/flat_head_inhaling.png")

# part = df.iloc[39*6000:39*6000+6000]
# part.plot(subplots=True)
# plt.savefig("fig/fluctuating_breath.png")

# part = df.iloc[41*6000-3000:41*6000+3000]
# part.plot(subplots=True)
# plt.savefig("fig/fluctuating_pressure.png")

# part = df.iloc[42*6000:42*6000+3000]
# part.plot(subplots=True)
# plt.savefig("fig/time_gap.png")

# part = df.iloc[100+36*6000:6100+44*6000]
# part.plot(subplots=True)
# plt.show()


phase = df['B_phase'].to_numpy()
phase_diff = np.diff(phase)
boundry = np.where(phase_diff == 1)[0]
switch = np.where(phase_diff == -1)[0]


# plot the pressure and flow of a section containing the first 7 breaths with subplots 2,1
# plt.subplots(2, 1, figsize=(10, 5))
# plt.subplot(2, 1, 1)
# plt.plot(df['Pressure'].iloc[boundry[0]:boundry[5]], color='C1')
# plt.title("Pressure")
# for i in range(5):
#     plt.axvline(x=boundry[i], color='grey', linestyle='-', alpha=0.5)
#     plt.axvline(x=switch[i], color='grey', linestyle='--', linewidth=1, alpha=0.5)
#
# plt.subplot(2, 1, 2)
# plt.plot(df['Flow'].iloc[boundry[0]:boundry[5]])
# plt.title("Flow")
# for i in range(5):
#     plt.axvline(x=boundry[i], color='grey', linestyle='-', alpha=0.5)
#     plt.axvline(x=switch[i], color='grey', linestyle='--', linewidth=1, alpha=0.5)
#
# plt.tight_layout()
# plt.savefig("fig/breath_segmentation.png")

flow = df['Flow'].to_numpy()
timestamp = df['Timestamp'].to_numpy()
time_diff = np.diff(timestamp)

for i in range(330,335):
    start = boundry[i] + 1
    mid = switch[i] + 1
    end = boundry[i+1] + 1
    # numpy inner product of flow[start:mid] and time_diff[start:mid]
    # print(f"Breath {i+1}: {np.inner(flow[start:mid], time_diff[start-1:mid-1])/100000:.4f} vol inhaling, {np.inner(flow[mid:end], time_diff[mid-1:end-1])/100000:.4f} vol exhaling")
    print(f"Breath {i+1}: {np.sum(flow[start:mid])/1000:.4f} vol inhaling, {np.sum(flow[mid:end])/1000:.4f} vol exhaling")