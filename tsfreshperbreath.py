import matplotlib.pyplot as plt

from BreathLoader import BreathLoader
from tsfresh import extract_relevant_features
from tsfresh.feature_selection import significance_tests
from sklearn.linear_model import LassoCV, LinearRegression
from sklearn.metrics import r2_score
import numpy as np
import pandas as pd
import os
import re

# 818 - 762

def load_and_prepare_data(start=0, end=None):
    VENTILATOR_FOLDER = r"C:\Users\yiyan\WorkSpace\UHN\ventilator converted files"

    # do not use 550, 556
    # try 551, 552, 557 first

    ventilator_files = {}
    for file in os.listdir(VENTILATOR_FOLDER):
        if file.endswith(".csv"):
            evlp_id = int(re.search(r"EVLP\d+", file).group()[4:])
            ventilator_files[evlp_id] = os.path.join(VENTILATOR_FOLDER, file)

    evlp_cases = list(ventilator_files.keys())
    evlp_cases.sort()
    if end is None:
        selected_cases = evlp_cases[start:]
    else:
        selected_cases = evlp_cases[start:end]
    print("Selected Cases:", selected_cases)

    Xs = []
    ys = []
    for evlp_id in selected_cases:
        breath_data = BreathLoader(ventilator_files[evlp_id])
        timestamps = breath_data.timestamp
        flow = breath_data.flow
        pressure = breath_data.pressure

        breath_id = np.zeros_like(timestamps)
        n_breaths = breath_data.n_breaths
        y = {}
        for i in range(n_breaths):
            start_idx, end_idx = breath_data.get_breath_boundary(i)
            breath_id[start_idx:end_idx] += i + 1
            params = breath_data.calc_params(i)
            y[evlp_id * 10000 + i + 1] = params["Dy_comp"]

        # remove leading 0s in breath_id
        timestamps = timestamps[breath_id > 0]
        flow = flow[breath_id > 0]
        pressure = pressure[breath_id > 0]
        breath_id = breath_id[breath_id > 0] + evlp_id * 10000

        Xs.append(pd.DataFrame({"Id": breath_id, "Timestamp": timestamps, "Flow": flow, "Pressure": pressure}))
        ys.append(pd.Series(y))

    X = pd.concat(Xs)
    y = pd.concat(ys)

    print(X.shape)
    print(y)

    return X, y


def decomposition(X, y):
    features = extract_relevant_features(X, y, column_id='Id', column_sort='Timestamp')
    return features


def check_significance(features, y):
    p_values = pd.Series(index=features.columns)
    errors = []
    for feature in features.columns:
        try:
            feature_series = features[feature]
            p_value = significance_tests.target_real_feature_real_test(feature_series, y)
            print(p_value)
            p_values[feature] = p_value
        except Exception as e:
            print(e)
            errors.append(feature)
        print()

    print("Number of Errors:", len(errors))

    print(p_values.sort_values().tail(10))

def lasso_feature_selection(features, y):
    lasso = LassoCV(max_iter=100000, n_jobs=24).fit(features, y)
    importance = np.abs(lasso.coef_)
    feature_names = np.array(features.columns)
    selected_features = feature_names[importance > 0]
    print("Selected Features:", selected_features)
    print("Number of Selected Features:", len(selected_features))
    return selected_features

def adjusted_R_square(features, selected, y):
    # return the adjusted R square of the linear regression model using the selected features
    X = features[selected]
    model = LinearRegression().fit(X, y)
    y_pred = model.predict(X)
    r_square = r2_score(y, y_pred)
    print("R Square:", r_square)
    n = len(y)
    print("Number of Samples:", n)
    p = len(selected)
    print("Number of Features:", p)
    r_square = 1 - (1 - r_square) * (n - 1) / (n - p - 1)
    print("Adjusted R Square:", r_square)
    return r_square

if __name__ == "__main__":
    # SAVEPATH = {"y": "tf_y.csv", "features": "tf_features.csv"}
    SAVEPATH = {"y": "tf_y_551.csv", "features": "tf_features_551.csv"}
    # X, y = load_and_prepare_data(1,2)
    # y.to_csv(SAVEPATH["y"])
    # features = decomposition(X, y)
    # print(features.shape)
    # features.to_csv(SAVEPATH["features"])
    y = pd.read_csv(SAVEPATH["y"], header=0, index_col=0).squeeze()
    features = pd.read_csv(SAVEPATH["features"], header=0, index_col=0)
    # check_significance(features, y)
    r_square = 0
    for i in range(5):
        selected = lasso_feature_selection(features, y)
        r_square = adjusted_R_square(features, selected, y)
        if r_square > 0.8:
            break


"""
'Pressure__c3__lag_1' 'Flow__c3__lag_3'
"""
