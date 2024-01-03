import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, log_loss
import time


def read_file(file):
    print("reading file: ", file)
    # read selected columns
    breath = pd.read_csv(file, header=0, sep=',')

    # extract flow and phase
    pressure = breath['Pressure'].to_numpy()
    flow = breath['Flow'].to_numpy()
    if 'B_phase' in breath.columns:
        phase = breath['B_phase'].to_numpy()
    else:
        phase = np.random.randint(0, 2, size=flow.shape)
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


def train(X_train, y_train, X_test, y_test):

    # print(X_train.shape)
    # print(y_train.shape)
    # print(X_test.shape)
    # print(y_test.shape)

    start = time.time()

    # clf = RandomForestClassifier(n_estimators=64, max_depth=8, random_state=0, n_jobs=-1, verbose=2, criterion="gini")
    # clf = RandomForestClassifier(n_estimators=64, max_depth=8, random_state=0, n_jobs=-1, verbose=2, criterion="entropy")
    # clf = RandomForestClassifier(n_estimators=64, max_depth=8, random_state=0, n_jobs=-1, verbose=2, criterion="log_loss")
    clf = RandomForestClassifier(n_estimators=64, max_depth=8, random_state=0, n_jobs=-1, criterion="log_loss", verbose=2)
    clf.fit(X_train, y_train.ravel())

    y_pred = clf.predict(X_test)

    end = time.time()
    print("time elapsed: ", end - start)

    # use suitable score for data segmentation
    print("accuracy score: ", accuracy_score(y_test, y_pred))
    print("log loss: ", log_loss(y_test, y_pred))
    print()

    return clf, y_pred


if __name__ == "__main__":

    WINDOW_SIZE = 1201

    folder = r'..\EVLP Data\raw ventilator ts'
    bellavista_folder = r'..\EVLP Data\Bellavista data'

    files = ["20220418_EVLP818_converted.csv",
             "20220417_EVLP817_converted.csv",
             "20220218_EVLP803_converted.csv",
             # "20210925_EVLP782_converted.csv",
             # "20210620_EVLP762_converted.csv",
             # "20210521_EVLP753_converted.csv",
             # "20210410_EVLP738_converted.csv",
             # "20210408_EVLP737_converted.csv",
             # "20210405_EVLP736_converted.csv",
             "20210403_EVLP735_converted.csv"]

    bellavista_files = ["combined_evlp887.csv",
                        "combined_evlp895.csv"]

    files = [os.path.join(folder, file) for file in files]
    bellavista_files = [os.path.join(bellavista_folder, file) for file in bellavista_files]

    X_train, y_train = concat_files(files, window_size=WINDOW_SIZE)
    X_test, y_test = concat_files(bellavista_files, window_size=WINDOW_SIZE)
    clf, y_pred = train(X_train, y_train, X_test, y_test)

    change01 = np.where(np.diff(y_pred) == 1)[0]
    change10 = np.where(np.diff(y_pred) == -1)[0]

    for i in range(len(change01)):
        plt.subplot(2, 1, 1)
        plt.plot(X_test[change01[i]], label='0->1')
        plt.axvline(x=WINDOW_SIZE//3, color='r')
        plt.legend()
        plt.subplot(2, 1, 2)
        plt.plot(X_test[change10[i]], label='1->0')
        plt.axvline(x=WINDOW_SIZE//3, color='r')
        plt.legend()
        plt.show()

