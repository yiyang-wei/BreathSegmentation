import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import os
import re

from sklearn.model_selection import ParameterGrid
import multiprocessing as mp


CLIP = 800


def read_cases(folder):
    cases = {}
    for file_name in os.listdir(folder):
        if file_name.endswith(".csv"):
            case_id = int(re.search(r"EVLP\d+", file_name).group()[4:])
            df = pd.read_csv(os.path.join(folder, file_name), header=0)
            cases[case_id] = df
    return cases

def segment_breath_A(flow, flow_threshold_0=0, flow_threshold_1=600):
    left = 1
    right = left + 1
    B_phase = np.zeros(flow.shape[0])
    increasing = False
    decreasing = False
    phase = 0
    while right < flow.shape[0] - 1:
        if flow[right] >= flow[right-1] and decreasing:
            decreasing = False
            if flow[right] < flow_threshold_0 and phase == 1:
                phase = 0
            B_phase[left+1:right] = phase
            left = right
            right = left + 1
            increasing = True
        elif flow[right] <= flow[right-1] and increasing:
            increasing = False
            if flow[right] > flow_threshold_1 and phase == 0:
                phase = 1
            B_phase[left:right+1] = phase
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


def segment_breath_B(flow, threshold01=250, threshold10=250, slope_threshold01=80, slope_threshold10=80, forward=3, flow_threshold01=375, flow_threshold10=0, flow_hard_threshold01=520):
    B_phase = np.zeros(flow.shape[0])
    phase = 0
    idx = 1
    while idx < flow.shape[0] - 10:
        if ((flow[idx] < threshold01 <= flow[idx+1] or flow[idx+1] - flow[idx] > slope_threshold01) and flow[idx+forward] > flow_threshold01) or flow[idx] > flow_hard_threshold01:
            phase = 1
        elif ((flow[idx-1] > threshold10 >= flow[idx] or flow[idx] - flow[idx-1] < -slope_threshold10) and flow[idx+forward] < flow_threshold10) or flow[idx] < flow_threshold10:
            phase = 0
        B_phase[idx] = phase
        idx += 1
    return B_phase


def segment_breath_C(flow, threshold01=5, threshold10=5, slope_threshold01=1, slope_threshold10=1, forward=4, flow_threshold01=9, flow_threshold10=-0.5):
    B_phase = np.zeros(flow.shape[0])
    phase = 0
    idx = 1
    while idx < flow.shape[0] - 10:
        if ((flow[idx] < threshold01 <= flow[idx+1] or flow[idx+1] - flow[idx] > slope_threshold01) and flow[idx+forward] > flow_threshold01) or flow[idx] > flow_threshold01:
            phase = 1
        elif ((flow[idx-1] > threshold10 >= flow[idx] or flow[idx] - flow[idx-1] < -slope_threshold10) and flow[idx+forward] < flow_threshold10) or flow[idx] < flow_threshold10:
            phase = 0
        B_phase[idx] = phase
        idx += 1
    return B_phase

def segment_breath_D(volume, ):
    B_phase = np.zeros(volume.shape[0])
    phase = 0
    idx = 1
    while idx < volume.shape[0] - 10:
        if volume[idx] > volume[idx-1]:
            phase = 1
        elif volume[idx] < volume[idx-1]:
            phase = 0
        B_phase[idx] = phase
        idx += 1


def segment_servo_i(cases, log=True, plot=False, kwargs=None):
    if kwargs is not None:
        print(kwargs)
    accuracies = []
    for case in cases:
        df = cases[case]
        flow = df["Flow"].to_numpy()
        if kwargs is None:
            pred_B_phase = segment_breath_B(flow)
        else:
            pred_B_phase = segment_breath_B(flow, **kwargs)
        flow = flow[CLIP:-CLIP]
        B_phase = df["B_phase"].to_numpy()
        B_phase = B_phase[CLIP:-CLIP]
        pred_B_phase = pred_B_phase[CLIP:-CLIP]
        # evaluate accuracy
        diff = pred_B_phase - B_phase
        diff[diff != 0] = 1
        accuracy = 1 - np.sum(diff) / diff.shape[0]
        accuracies.append(accuracy)
        if log:
            print(f"Accuracy of EVLP{case}: {accuracy}")
        if plot:
            window_size = 6000
            for i in range(window_size*38, flow.shape[0], window_size):
                err = B_phase[i:i+window_size] - pred_B_phase[i:i+window_size]
                mistakes = np.where(err != 0)[0]
                if mistakes.shape[0] < 16:
                    continue
                j = 0
                while j < mistakes.shape[0]:
                    start = mistakes[j]
                    while j < mistakes.shape[0] - 1 and mistakes[j + 1] == mistakes[j] + 1:
                        j += 1
                    end = mistakes[j]
                    padding = 1
                    print(np.vstack((flow[i + start - padding:i + end + padding + 1],
                                     err[start - padding:end + padding + 1].astype(int),
                                     B_phase[i + start - padding:i + end + padding + 1].astype(int),
                                     pred_B_phase[i + start - padding:i + end + padding + 1].astype(int))))
                    j += 1
                print()
                plt.subplot(3, 1, 1)
                plt.plot(flow[i:i+window_size])
                plt.subplot(3, 1, 2)
                plt.plot(pred_B_phase[i:i+window_size])
                plt.subplot(3, 1, 3)
                plt.plot(err, color="red")
                plt.show()
        if len(accuracies) % 10 == 0:
            if np.mean(accuracies) < 0.996:
                print(f"{kwargs} Low accuracy, aborting...")
                break
    else:
        print(kwargs)
        print(f"Average accuracy: {np.mean(accuracies)}")
        print("Worst 3 accuracies:")
        worst_3 = np.argsort(accuracies)[:3]
        for i in worst_3:
            print(f"EVLP{list(cases.keys())[i]}: {accuracies[i]}")

    return kwargs, accuracies


def segment_bellavista():
    bellavista_folder = r"..\EVLP data\Bellavista data"

    files = [file for file in os.listdir(bellavista_folder) if file.endswith(".csv")]

    use_files = files[:]

    for file in use_files:
        df = pd.read_csv(os.path.join(bellavista_folder, file), header=0)
        flow = df["Flow"].to_numpy()
        pred_B_phase = segment_breath_C(flow)
        flow = flow[CLIP:-CLIP]
        pred_B_phase = pred_B_phase[CLIP:-CLIP]
        window_size = 18000
        for i in range(window_size*0, flow.shape[0], window_size):
            plt.subplot(2, 1, 1)
            plt.plot(flow[i:i+window_size])
            plt.subplot(2, 1, 2)
            plt.plot(pred_B_phase[i:i+window_size])
            plt.show()



if __name__ == "__main__":
    param_grid = ParameterGrid({"threshold01": [200, 250, 300, 350],
                                "slope_threshold01": [20, 40, 60, 80],
                                "forward": [2, 3, 4],
                                "flow_threshold01": [350, 375, 400, 425],
                                "flow_hard_threshold01": [450, 500, 520, 550]})

    cases = read_cases(r"..\ventilator converted files")

    with mp.Pool(processes=4) as pool:
        results = pool.starmap(segment_servo_i, [(cases, False, False, kwargs) for kwargs in param_grid])

    grid_search = {}
    for result in results:
        grid_search[tuple(result[0])] = result[1]

    with open("grid_search.pkl", "wb") as f:
        pickle.dump(grid_search, f)


    # with open("grid_search.pkl", "rb") as f:
    #     grid_search = pickle.load(f)

    # print top 5 mean accuracy settings and their accuracies
    mean_accuracies = [(key, np.mean(value)) for key, value in grid_search.items()]
    mean_accuracies.sort(key=lambda x: x[1], reverse=True)
    print(mean_accuracies[:5])

    # segment_bellavista()