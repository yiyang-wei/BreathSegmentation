import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os


FLOW_THRESHOLD = 600


def segment_breath(flow):
    left = 1
    right = left + 1
    B_phase = np.zeros(flow.shape[0])
    increasing = False
    decreasing = False
    phase = 0
    while right < flow.shape[0] - 1:
        if flow[right] >= flow[right-1] and decreasing:
            decreasing = False
            if flow[right] < 0 and phase == 1:
                phase = 0
            B_phase[left+1:right+1] = phase
            left = right
            right = left + 1
            increasing = True
        elif flow[right] <= flow[right-1] and increasing:
            increasing = False
            if flow[right] > FLOW_THRESHOLD and phase == 0:
                phase = 1
            B_phase[left+1:right+1] = phase
            left = right
            right = left + 1
            decreasing = True
        elif flow[right] > flow[right-1]:
            increasing = True
            decreasing = False
            right += 1
        elif flow[right] < flow[right-1]:
            increasing = False
            decreasing = True
            right += 1
        else:
            right += 1

    return B_phase




raw_ventilator_folder = r"..\ventilator converted files"

files = [file for file in os.listdir(raw_ventilator_folder) if file.endswith(".csv")]

use_files = files[:]

accuracies = []
for file in use_files:
    df = pd.read_csv(os.path.join(raw_ventilator_folder, file), header=0)
    flow = df["Flow"].to_numpy()
    pred_B_phase = segment_breath(flow)
    B_phase = df["B_phase"].to_numpy()
    # evaluate accuracy
    diff = pred_B_phase - B_phase
    diff[diff != 0] = 1
    accuracy = 1 - np.sum(diff) / diff.shape[0]
    accuracies.append(accuracy)
    print(f"Accuracy of {file}: {accuracy}")
    # for i in range(6000*0, flow.shape[0], 6000):
    #     plt.subplot(3, 1, 1)
    #     plt.plot(flow[i:i+6000], label="flow")
    #     plt.subplot(3, 1, 2)
    #     plt.plot(B_phase[i:i+6000], label="B_phase")
    #     plt.subplot(3, 1, 3)
    #     plt.plot(pred_B_phase[i:i+6000], label="pred_B_phase")
    #     plt.legend()
    #     plt.show()

print(f"Average accuracy: {np.mean(accuracies)}")