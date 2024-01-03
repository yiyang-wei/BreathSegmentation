import pandas as pd
import numpy as np


# flow and B_phase only
def read_file(file):
    # read selected columns
    breath = pd.read_csv(file, header=0, sep=',', usecols=["Pressure", "Flow", "B_phase"])

    # extract flow and phase
    pressure = breath['Pressure'].to_numpy()
    flow = breath['Flow'].to_numpy()
    phase = breath['B_phase'].to_numpy()
    return pressure, flow, phase


def fixed_window(flow, phase, clip=1000, window_size=601):

    # remove the first 1000 points and the last 1000 points
    flow = flow[clip:-clip]
    phase = phase[clip:-clip]

    # apply fixed size sliding window split
    L = len(flow)
    X = np.zeros((L - window_size, window_size))
    y = np.zeros((L - window_size, 1))

    for i in range(X.shape[0]):
        X[i] = flow[i:i + window_size]
        y[i] = phase[i + window_size // 3]

    return X, y


def concat_files(files, clip=1000, window_size=601):
    X = None
    y = None
    for file in files:
        _, flow, phase = read_file(file)
        X_new, y_new = fixed_window(flow, phase, clip, window_size)
        if X is None:
            X = X_new
            y = y_new
        else:
            X = np.concatenate((X, X_new))
            y = np.concatenate((y, y_new))
    return X, y


def breath_window(pressure, flow, phase, window_size=1200):
    # find each occurrence of 1,0 in phase (switching from inhaling to exhaling)
    phase_diff = np.diff(phase)
    switches = np.where(phase_diff == -1)[0]
    L = flow.shape[0]
    # drop switches that are too close to the beginning or the end
    switches = switches[window_size <= switches]
    switches = switches[switches <= L - window_size]
    N = switches.shape[0]
    # create a window for each switch
    pressure_window = np.zeros((N, window_size))
    flow_window = np.zeros((N, window_size))
    for i in range(N):
        start = switches[i] - window_size // 2 + 1
        end = switches[i] + window_size // 2 + 1
        pressure_window[i] = pressure[start:end]
        flow_window[i] = flow[start:end]
    return pressure_window, flow_window


# pressure, flow, phase = read_file("data/20220418_EVLP818_converted.csv")
