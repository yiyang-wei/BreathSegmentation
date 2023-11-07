from BreathLoader import BreathLoader
from SlidingWindow import SlidingWindow
import numpy as np
import pandas as pd
from enum import Enum
import pickle
import os
import re

VENTILATOR_DATA_FOLDER = r"C:\Users\yiyan\WorkSpace\UHN\ventilator converted files"
BREATH_LABEL_FOLDER = r"C:\Users\yiyan\WorkSpace\UHN\ventilator breathlabels"


# Get files in each folder and store in two dictionaries
# The key is the EVLP number, the value is the file name

def get_files_in_folder(folder):
    files = {}
    for file in os.listdir(folder):
        if file.endswith(".csv"):
            EVLP_id = re.search("EVLP\d+", file).group()
            files[EVLP_id] = file
    return files


ventilator_files = get_files_in_folder(VENTILATOR_DATA_FOLDER)
breath_label_files = get_files_in_folder(BREATH_LABEL_FOLDER)


class BreathLabel(Enum):
    def __new__(cls, value, color, shortcuts):
        obj = object.__new__(cls)
        obj._value_ = value
        obj.color = color
        obj.shortcuts = shortcuts
        return obj

    Unvisited = (0, "black", None)
    Normal = (1, "black", ["0", "1"])
    Assessment = (2, "green", ["2"])
    Bronch = (3, "blue", ["3"])
    Deflation = (4, "violet", ["4"])
    Question = (5, "orange", ["5"])
    InPause = (6, "gold", ["6"])
    ExPause = (7, "indigo", ["7"])
    Recruitment = (8, "aqua", ["8"])
    Noise = (9, "red", ["9"])


N_LABELS = len(BreathLabel)


def load_label(filename):
    """Load the breath label file."""
    file_path = os.path.join(BREATH_LABEL_FOLDER, filename)
    try:
        df = pd.read_csv(file_path, header=0, usecols=["Breath_num", "Label"])
        breath_num = df["Breath_num"].to_numpy()
        labels = df["Label"].to_numpy()
        return breath_num, labels
    except Exception as e:
        print(f"Error reading file: {e}")
        exit()


def compute_y(breath_data, labels, left_focus, mid, right_focus):
    """Compute the y vector of the given window."""
    y = np.zeros(N_LABELS)
    for i in range(left_focus, right_focus, 10):
        breath_index = breath_data.find_breath_by_index(i)
        label = BreathLabel[labels[breath_index]].value
        y[label] += 0.5 if i < mid else 1
    y /= np.sum(y)
    return y


def build_training(breath_data, breath_label, sw):
    """Build the training data and labels."""
    breath_num, labels = breath_label
    windows = sw.get_windows(breath_data.flow.shape[0])
    X = np.zeros((windows.shape[0], sw.window_size))
    y = np.zeros((windows.shape[0], N_LABELS))
    for i in range(windows.shape[0]):
        window_left, window_left_focus, window_mid, window_right_focus, window_right = windows[i]
        X[i] = breath_data.pressure[window_left:window_right]
        y[i] = compute_y(breath_data, labels, window_left_focus, window_mid, window_right_focus)
    return X, y


sw = SlidingWindow(400, 800, 200, 400, 1000, 1000, 100)
# breath_data = BreathLoader(os.path.join(VENTILATOR_DATA_FOLDER, ventilator_files["EVLP550"]))
# breath_label = load_label(breath_label_files["EVLP550"])
# X, y = build_training(breath_data, breath_label, sw)
Xs = []
ys = []
for case in breath_label_files.keys():
    breath_data = BreathLoader(os.path.join(VENTILATOR_DATA_FOLDER, ventilator_files[case]))
    breath_label = load_label(breath_label_files[case])
    X, y = build_training(breath_data, breath_label, sw)
    Xs.append(X)
    ys.append(y)

# split the data into training and testing sets
X_train = np.concatenate(Xs[:3])
y_train = np.concatenate(ys[:3])
X_test = Xs[4]
y_test = ys[4]

X_train_drop = []
y_train_drop = []
for i in range(y_train.shape[0]):
    if y_train[i, 1] < 0.999 or np.random.rand() >= 0.7:
        X_train_drop.append(X_train[i])
        y_train_drop.append(y_train[i])

# drop test data with more than 1 non-zero label
X_test_drop = []
y_test_drop = []
for i in range(y_test.shape[0]):
    if np.where(y_test[i] > 0)[0].shape[0] == 1:
        X_test_drop.append(X_test[i])
        y_test_drop.append(y_test[i])

X_train = np.array(X_train_drop)
y_train = np.array(y_train_drop)
X_test = np.array(X_test_drop)
y_test = np.array(y_test_drop)
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

input("Press Enter to continue...")

# Random Forest Regression on each label
from sklearn.ensemble import RandomForestRegressor

regressors = []
for i in range(N_LABELS):
    print(f"Training label {i}")
    # print the progress
    regressor = RandomForestRegressor(n_estimators=80, random_state=0, n_jobs=20, verbose=1)
    regressor.fit(X_train, y_train[:, i])
    regressors.append(regressor)

# save the regressors
with open("regressors.pkl", "wb") as f:
    pickle.dump(regressors, f)

# load the regressors
# with open("regressors.pkl", "rb") as f:
#     regressors = pickle.load(f)

# Predict the labels of the test data
y_pred = np.zeros((y_test.shape[0], N_LABELS))
for i in range(N_LABELS):
    y_pred[:, i] = regressors[i].predict(X_test)

# Compute the accuracy
for i in range(N_LABELS):
    test_labels = y_test[:, i]
    pred_labels = np.argmax(y_pred, axis=1) == i
    accuracy = np.sum(test_labels == pred_labels) / test_labels.shape[0]
    correct_catch = np.sum(np.logical_and(test_labels == 1, pred_labels == 1)) / np.sum(test_labels == 1)
    correct_reject = np.sum(np.logical_and(test_labels == 0, pred_labels == 0)) / np.sum(test_labels == 0)
    wrong_catch = np.sum(np.logical_and(test_labels == 0, pred_labels == 1)) / np.sum(pred_labels == 1)
    wrong_reject = np.sum(np.logical_and(test_labels == 1, pred_labels == 0)) / np.sum(pred_labels == 0)
    print(BreathLabel(i).name)
    print(f"Accuracy: {accuracy}")
    print(f"Correct catch: {correct_catch}")
    print(f"Correct reject: {correct_reject}")
    print(f"Wrong catch: {wrong_catch}")
    print(f"Wrong reject: {wrong_reject}")
    print()
