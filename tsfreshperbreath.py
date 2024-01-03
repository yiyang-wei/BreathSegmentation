import matplotlib.pyplot as plt

from BreathLoader import BreathLoader
from tsfresh import extract_relevant_features, extract_features
from tsfresh.feature_selection import significance_tests
from sklearn.linear_model import LassoCV, Lasso
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
import numpy as np
import pandas as pd
import os
import re

# 818 - 762

VENTILATOR_FOLDER = r"C:\Users\yiyan\WorkSpace\UHN\ventilator converted files"
DONOR_FILE = r"C:\Users\yiyan\WorkSpace\UHN\EVLP data\EVLP#1-879_donor.csv"
SAVEFOLER = "tsfresh feature tables assessment"


def save_to_csv(df, out_file_name):
    df.to_csv(os.path.join(SAVEFOLER, out_file_name))

def file_name_identifier(evlp_ids):
    if len(evlp_ids) == 1:
        file_names_identifier = f"{evlp_ids[0]}"
    else:
        file_names_identifier = f"{evlp_ids[0]}_{evlp_ids[-1]}"
    return file_names_identifier

def get_filenames_by_ids(evlp_ids):
    id_to_filename = {}
    for file in os.listdir(VENTILATOR_FOLDER):
        if file.endswith(".csv"):
            evlp_id = int(re.search(r"EVLP\d+", file).group()[4:])
            if evlp_id in evlp_ids:
                id_to_filename[evlp_id] = os.path.join(VENTILATOR_FOLDER, file)
    return id_to_filename


def process_selected_cases(id_to_filename):
    evlp_cases = list(id_to_filename.keys())
    evlp_cases.sort()
    file_name = file_name_identifier(evlp_cases)
    y_file_name = f"dy_comp_{file_name}.csv"
    features_file_name = f"features_{file_name}.csv"
    selected_features_file_name = f"selected_features_{file_name}.txt"
    if os.path.exists(os.path.join(SAVEFOLER, selected_features_file_name)):
        print("Read Selected Features from File:", selected_features_file_name)
        with open(os.path.join(SAVEFOLER, selected_features_file_name), "r") as f:
            lines = f.readlines()
            r_square = float(lines[0].split(":")[1])
            print("Adjusted R Square:", r_square)
            selected = [line.strip() for line in lines[1:]]
            print("Number of Selected Features:", len(selected))
            return r_square, selected
    if os.path.exists(os.path.join(SAVEFOLER, y_file_name)) and os.path.exists(os.path.join(SAVEFOLER, features_file_name)):
        print("Read Data from File:", y_file_name, features_file_name)
        y = pd.read_csv(os.path.join(SAVEFOLER, y_file_name), header=0, index_col=0).squeeze()
        features = pd.read_csv(os.path.join(SAVEFOLER, features_file_name), header=0, index_col=0)
    else:
        print("Loading Selected Cases:", evlp_cases)
        X, y = load_and_prepare_data(id_to_filename)
        save_to_csv(y, y_file_name)
        features = decomposition(X, y)
        save_to_csv(features, features_file_name)
    selected, r_square = lasso_feature_selection(features, y)
    with open(os.path.join(SAVEFOLER, selected_features_file_name), "w") as f:
        lines = [f"Adjusted R Square: {r_square}\n"] + [f"{feature}\n" for feature in selected]
        f.writelines(lines)
    return r_square, selected


def load_and_prepare_data(id_to_filename):
    donor_df = pd.read_csv(DONOR_FILE, header=0, usecols=["Evlp Id No", "Donor Weight Kg"])

    Xs = []
    ys = []
    for evlp_id in id_to_filename.keys():
        body_weight = donor_df.loc[donor_df["Evlp Id No"] == evlp_id, "Donor Weight Kg"].values[0]
        breath_data = BreathLoader(id_to_filename[evlp_id])
        timestamps = breath_data.timestamp
        flow = breath_data.flow / body_weight
        pressure = breath_data.pressure

        breath_id = np.zeros_like(timestamps)
        n_breaths = breath_data.n_breaths
        y = {}
        for i in range(n_breaths):
            start_idx, end_idx = breath_data.get_breath_boundary(i)
            duration = (timestamps[end_idx] - timestamps[start_idx]) / 10000
            if 5.5 <= duration <= 6.5:
                breath_id[start_idx:end_idx] = i + 1
                y[evlp_id * 10000 + i + 1] = breath_data.dynamic_compliance(i)

        # remove leading 0s in breath_id
        timestamps = timestamps[breath_id > 0]
        flow = flow[breath_id > 0]
        pressure = pressure[breath_id > 0]
        breath_id = breath_id[breath_id > 0] + evlp_id * 10000

        Xs.append(pd.DataFrame({"Id": breath_id, "Timestamp": timestamps, "Flow": flow, "Pressure": pressure}))
        ys.append(pd.Series(y))

    X = pd.concat(Xs)
    y = pd.concat(ys)

    print("X shape:", X.shape)
    print("y shape:", y.shape)

    return X, y


def decomposition(X, y):
    features = extract_relevant_features(X, y, column_id='Id', column_sort='Timestamp')
    print("Number of Features:", features.shape[1])
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

def forward_selection(features, y, significance_test=0.001):
    initial_features = features.columns.tolist()
    best_features = []
    current_model = None

    while initial_features:
        pvals = []
        for candidate in initial_features:
            # Fit model with the current best features plus the new candidate feature
            tested_features = best_features + [candidate]
            new_model = sm.OLS(y, sm.add_constant(features[tested_features])).fit()
            if current_model is None:
                p_value = 0
            else:
                # check if the new model is significantly better than the current model
                p_value = sm.stats.anova_lm(current_model, new_model)["Pr(>F)"][1]
            pvals.append((p_value, candidate))

        # Sort the p-values and get the one for the best candidate
        pvals.sort()
        best_pval, best_candidate = pvals[0]

        # If the best p-value is less than our significance threshold, we add the feature to our model
        if best_pval < significance_test:
            best_features.append(best_candidate)
            current_model = new_model
            initial_features.remove(best_candidate)
            print("Selected:", best_candidate, "\t\tP-value:", best_pval, "\t\tNew Length:", len(best_features))
        else:
            print("P-value:", best_pval)
            break  # If no new features are significant, we stop the selection process

    print("Number of Selected Features:", len(best_features))

    adj_r_square = current_model.rsquared_adj
    print("Adjusted R Square:", adj_r_square)

    return best_features, adj_r_square

def adjust_r_square(r_square, n, p):
    adj_r_square = 1 - (1 - r_square) * (n - 1) / (n - p - 1)
    print("Adjusted R Square:", adj_r_square)
    return adj_r_square

def lasso_feature_selection(features, y):
    scaler = StandardScaler()
    feature_names = np.array(features.columns)
    features_scaled = scaler.fit_transform(features)
    n = len(y)

    lasso_cv = LassoCV(max_iter=100000, n_jobs=24, cv=10, n_alphas=100, eps=1e-2).fit(features_scaled, y)

    important_features = feature_names[lasso_cv.coef_ != 0]
    print("Number of Selected Features using the Optimal Model:", len(important_features))

    if len(important_features) <= 40:
        # calculate the adjusted R square
        r_square = lasso_cv.score(features_scaled, y)
        print("R Square:", r_square)
        p = len(important_features)
        adj_r_square = adjust_r_square(r_square, n, p)
        return important_features, adj_r_square

    # Calculate mean and standard error of the mean squared errors
    mse_mean = np.mean(lasso_cv.mse_path_, axis=1)
    mse_se = np.std(lasso_cv.mse_path_, axis=1) / np.sqrt(lasso_cv.mse_path_.shape[1])

    # Find the index of the minimum MSE
    min_mse_index = np.argmin(mse_mean)

    # Find the maximum alpha within one standard error of the minimum MSE
    alpha_1se = lasso_cv.alphas_[np.where(mse_mean <= mse_mean[min_mse_index] + mse_se[min_mse_index])[0][0]]

    # Fit Lasso model with the chosen alpha_1se
    lasso_1se = Lasso(alpha=alpha_1se, max_iter=100000)
    lasso_1se.fit(features_scaled, y)

    # Extract coefficients using the alpha_1se
    coef_1se = lasso_1se.coef_

    # Select features where the coefficient is non-zero
    selected_features_1se = feature_names[coef_1se != 0]

    # Print the selected features
    print("Number of Selected Features using 1se Rule:", len(selected_features_1se))

    # calculate the adjusted R square
    r_square = lasso_1se.score(features_scaled, y)
    print("R Square:", r_square)
    p = len(selected_features_1se)
    adj_r_square = adjust_r_square(r_square, n, p)
    return selected_features_1se, adj_r_square

def ols_evaluation(features, y):
    X = sm.add_constant(features)
    model = sm.OLS(y, X).fit()
    return model.rsquared_adj

if __name__ == "__main__":
    # cases = [762, 782, 803, 817, 818]
    # id_to_filename = get_filenames_by_ids(cases)
    # r_square, selected = process_selected_cases(id_to_filename)
    # print(selected)
    # 621 has NaN in Flow
    cases = [551, 553, 554, 555, 557, 558, 560, 563, 564, 565, 568, 572, 573,
             574, 575, 577, 579, 592, 593, 595, 598, 600, 603, 610, 615, 616,
             617, 618, 619, 631, 682, 685, 686, 694, 698, 730, 731, 736,
             738, 753, 762, 782, 803, 817, 818]

    # selected_features = ["Flow__index_mass_quantile__q_0.7",
    #                      "Pressure__index_mass_quantile__q_0.7",
    #                      "Pressure__energy_ratio_by_chunks__num_segments_10__segment_focus_9",
    #                      "Flow__energy_ratio_by_chunks__num_segments_10__segment_focus_6",
    #                      "Flow__autocorrelation__lag_3",
    #                      "Pressure__minimum",
    #                      "Flow__mean"]

    feature_parameters = {
        "index_mass_quantile": [{'q': 0.7}],
        "energy_ratio_by_chunks": [{'num_segments': 10, 'segment_focus': 9}, {'num_segments': 10, 'segment_focus': 6}],
        "autocorrelation": [{'lag': 3}],
        "minimum": None,
        "mean": None
    }

    adj_r_squares = []

    for c in cases:
        # read features and y from file
        id_to_filename = get_filenames_by_ids([c])
        X, y = load_and_prepare_data(id_to_filename)
        features = extract_features(X, column_id='Id', column_sort='Timestamp', n_jobs=8, default_fc_parameters=feature_parameters)
        adj_r_square = ols_evaluation(features, y)
        adj_r_squares.append(adj_r_square)
        print("Adjusted R Square:", adj_r_square)

    plt.hist(adj_r_squares, bins=20)
    plt.show()


    # count_selected = {}
    # for c in cases:
    #     id_to_filename = get_filenames_by_ids([c])
    #     r_square, selected = process_selected_cases(id_to_filename)
    #
    #     for feature in selected:
    #         if feature not in count_selected:
    #             count_selected[feature] = 0
    #         count_selected[feature] += 1
    #
    # print()
    # print("Selected Features Frequency:")
    # # print count_selected in descending order
    # count = 20
    # for feature, count in sorted(count_selected.items(), key=lambda x: x[1], reverse=True):
    #     print(f"{feature}: {count}")
    #     count -= 1
    #     if count == 0:
    #         break
    # print()
    # # save to csv
    # frequency = pd.Series(count_selected)
    # save_to_csv(frequency, "frequency.csv")


"""EVLP551
'Pressure__c3__lag_1' 'Flow__c3__lag_3'
"""

"""EVLP557
'Pressure__c3__lag_1' 'Pressure__c3__lag_2' 'Pressure__abs_energy'
"""
