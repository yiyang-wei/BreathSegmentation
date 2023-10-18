from prepare_data import read_file, breath_window

import numpy as np
import matplotlib.pyplot as plt

pressure, flow, phase = read_file("data/20220418_EVLP818_converted.csv")
pressure_window, flow_window = breath_window(pressure, flow, phase, window_size=1200)

N = pressure_window.shape[0]
X = flow_window

# ask for user input to label the first 100 window and save in y
# 0: normal breath
# 1: assessment breath
# 2: other noise

y = np.zeros(N)

idx = 0
counts = [0] * 3
# loop until all labels appear at least three times
while min(counts) < 3:
    # show 3x3 plots at a time
    for i in range(3):
        for j in range(3):
            plt.subplot(3, 3, i * 3 + j + 1)
            plt.plot(pressure_window[idx + i * 3 + j])
    plt.show()
    # ask for user input, empty if all 0
    labels = input("labels: ")
    if labels == "":
        idx += 9
        continue
    for label in labels.split(" "):
        y[idx] = int(label)
        counts[int(label)] += 1
        idx += 1
    # print count of each label
    print(np.unique(y[:idx], return_counts=True))

# use modAL to query the rest of the labels when not confident

from modAL.models import ActiveLearner
from modAL.uncertainty import uncertainty_sampling
from sklearn.ensemble import RandomForestClassifier

# initialize ActiveLearner
learner = ActiveLearner(
    estimator=RandomForestClassifier(n_estimators=64, max_depth=8, random_state=0, n_jobs=-1, verbose=2, criterion="gini"),
    query_strategy=uncertainty_sampling,
    X_training=X[:idx],
    y_training=y[:idx]
)

prediction = np.zeros((N, 1))
prediction[:idx] = y[:idx]

# query for labels when not confident
threshold = 0.90
for i in range(idx, N):
    query_idx, query_inst = learner.query(X[i].reshape(1, -1))
    prediction[i] = learner.predict(X[i].reshape(1, -1))
    if query_inst[0, prediction[i]] < threshold:
        plt.subplot(2, 1, 1)
        plt.plot(pressure_window[i])
        plt.subplot(2, 1, 2)
        plt.plot(flow_window[i])
        plt.show()
        y[i] = int(input("label: "))
        # print count of each label
        print(np.unique(y[:i+1], return_counts=True))
        learner.teach(X[i].reshape(1, -1), y[i].reshape(1, ))
    else:
        learner.teach(X[i].reshape(1, -1), prediction[i].reshape(1, ))
