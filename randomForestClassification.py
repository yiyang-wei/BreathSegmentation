import matplotlib.pyplot as plt

from prepare_data import read_file, concat_files

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, log_loss

import time

files = ["../EVLP data/raw ventilator ts/20220418_EVLP818_converted.csv",
         "../EVLP data/raw ventilator ts/20220417_EVLP817_converted.csv",
         "../EVLP data/raw ventilator ts/20220218_EVLP803_converted.csv",
         "../EVLP data/raw ventilator ts/20210925_EVLP782_converted.csv",
         "../EVLP data/raw ventilator ts/20210620_EVLP762_converted.csv",
         "../EVLP data/raw ventilator ts/20210521_EVLP753_converted.csv"]

# 550, 552, 556, 606, 681 are single
# 599, 701, 737 double to single
# 556, 696 auto triggering


WINDOW_SIZE = 5

def train(X_train, y_train, X_test, y_test):

    # print(X_train.shape)
    # print(y_train.shape)
    # print(X_test.shape)
    # print(y_test.shape)

    start = time.time()

    # clf = RandomForestClassifier(n_estimators=64, max_depth=8, random_state=0, n_jobs=-1, verbose=2, criterion="gini")
    # clf = RandomForestClassifier(n_estimators=64, max_depth=8, random_state=0, n_jobs=-1, verbose=2, criterion="entropy")
    # clf = RandomForestClassifier(n_estimators=64, max_depth=8, random_state=0, n_jobs=-1, verbose=2, criterion="log_loss")
    clf = RandomForestClassifier(n_estimators=64, max_depth=8, random_state=0, n_jobs=-1, criterion="log_loss")
    clf.fit(X_train, y_train.ravel())

    y_pred = clf.predict(X_test)

    end = time.time()
    print("time elapsed: ", end - start)

    # use suitable score for data segmentation
    print("accuracy score: ", accuracy_score(y_test, y_pred))
    print("log loss: ", log_loss(y_test, y_pred))

    return clf, y_pred

for i in range(4):
    print("test file: ", files[i])
    X_train, y_train = concat_files(files[:i]+files[i+1:], window_size=WINDOW_SIZE)
    X_test, y_test = concat_files(files[i:i+1], window_size=WINDOW_SIZE)
    clf, y_pred = train(X_train, y_train, X_test, y_test)

    # TODO: check if misclassifications concentrate on noise data

    error = y_pred-y_test.reshape(-1)
    #
    # N = error.shape[0]
    # for i in range(10):
    #     plt.plot(error[i*N//10:(i+1)*N//10])
    #     plt.title(f"Accurate rate of {i*10} to {i*10+10} of dataset: {accuracy_score(y_test[i*N//10:(i+1)*N//10], y_pred[i*N//10:(i+1)*N//10])}")
    #     plt.show()
    #     print(f"Accurate rate of {i*10} to {i*10+10} of dataset: {accuracy_score(y_test[i*N//10:(i+1)*N//10], y_pred[i*N//10:(i+1)*N//10])}")

    misclassified = np.where(y_pred != y_test.reshape(-1))[0]

    step = 30000
    for i in range(40):
        # set figure size
        plt.figure(figsize=(16, 6))
        plt.subplot(3, 1, 1)
        plt.plot(y_pred[i*step:(i+1)*step], label="y_pred")
        plt.legend()
        plt.subplot(3, 1, 2)
        plt.plot(y_test.reshape(-1)[i*step:(i+1)*step], color="C1", label="y_test")
        plt.legend()
        plt.subplot(3, 1, 3)
        plt.plot(error[i*step:(i+1)*step], color="C2", label="error")
        plt.legend()
        plt.tight_layout()
        plt.show()

    # d_miss = np.diff(misclassified)
    # plt.hist(d_miss, bins=100)
    # plt.show()
    #
    # step = 400
    # for i in range(20):
    #     plt.subplot(2, 1, 1)
    #     plt.plot(d_miss[i * step:(i + 1)*step], ".-")
    #     plt.subplot(2, 1, 2)
    #     plt.plot(d_miss[i * step:(i + 1) * step], ".-")
    #     plt.ylim(0, 20)
    #     plt.show()

    break

# accuracy > 0.999 on sections with no noise
# more misclassifications on actual 1s

# TODO: filter out rapid change

# use re to find consecutive 1s with length < 10 in y_pred
# use re to find consecutive 0s with length < 10 in y_pred



# TODO: histgram of breath length 01+0+1

# predict_proba = clf.predict_proba(X_test)
# predict_proba = predict_proba[:, 1]
# for p in [0.25, 0.1, 0.05]:
#     y_pred_2 = np.zeros(y_pred.shape)
#     y_pred_2[predict_proba < p] = 0
#     y_pred_2[predict_proba > 1-p] = 1
#     y_pred_2[(predict_proba >= p) & (predict_proba <= 1-p)] = 0.5
#
#     # count percentage of predict_proba value between 0.25 and 0.75
#     print("percentage between 0.25 and 0.75: ",
#           np.sum((predict_proba > p) & (predict_proba < 1-p)) / predict_proba.shape[0])
#
#     # accuracy score for y_pred_2 but ignore 0.5
#     print("confident accuracy score: ", accuracy_score(y_test[y_pred_2 != 0.5], y_pred_2[y_pred_2 != 0.5]))