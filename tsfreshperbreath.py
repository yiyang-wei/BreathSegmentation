from BreathLoader import BreathLoader
from tsfresh import extract_relevant_features
from tsfresh.feature_selection import significance_tests
import numpy as np
import pandas as pd
import os


def load_and_prepare_data():
    VENTILATOR_FOLDER = r"C:\Users\yiyan\WorkSpace\UHN\ventilator converted files"

    # do not use 550, 556
    # try 551, 552, 557 first

    ventilator_file = r"20190616_EVLP551_converted.csv"

    breath_data = BreathLoader(os.path.join(VENTILATOR_FOLDER, ventilator_file))
    timestamps = breath_data.timestamp
    flow = breath_data.flow
    pressure = breath_data.pressure

    breath_id = np.zeros_like(timestamps)
    n_breaths = breath_data.n_breaths
    y = {}
    for i in range(n_breaths):
        start_idx, end_idx = breath_data.get_breath_boundary(i)
        breath_id[start_idx:end_idx] = i + 1
        params = breath_data.calc_params(i)
        y[i + 1] = params["Dy_comp"]

    # remove leading 0s in breath_id
    timestamps = timestamps[breath_id > 0]
    flow = flow[breath_id > 0]
    pressure = pressure[breath_id > 0]
    breath_id = breath_id[breath_id > 0]

    X = pd.DataFrame({"Id": breath_id, "Timestamp": timestamps, "Flow": flow, "Pressure": pressure})
    y = pd.Series(y)

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

if __name__ == "__main__":
    # X, y = load_and_prepare_data()
    # features = decomposition(X, y)
    # features.to_csv("tf_features.csv")
    features = pd.read_csv("tf_features.csv", header=0, index_col=0)
    y = pd.read_csv("tf_y.csv", header=0, index_col=0).squeeze()
    check_significance(features, y)


