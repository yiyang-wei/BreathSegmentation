from tsfresh import extract_relevant_features
import numpy as np
import pandas as pd
import os
import re


FOLDER = r"C:\Users\yiyan\WorkSpace\UHN\ventilator converted files"

outcome_df = pd.read_csv(r"C:\Users\yiyan\WorkSpace\UHN\EVLP data\EVLP#1-879_disposition.csv",
                             header=0, usecols=["Evlp Id No", "Extubation Time Hrs"],
                             dtype={"Evlp Id No": np.int16, "Extubation Time Hrs": np.float32})

ventilator_files = []
for file in os.listdir(FOLDER):
    if file.endswith(".csv"):
        evlp_id = int(re.search(r"EVLP\d+", file).group()[4:])
        if evlp_id in outcome_df["Evlp Id No"].values:
            hrs = outcome_df.loc[outcome_df["Evlp Id No"] == evlp_id, "Extubation Time Hrs"].values[0]
            if hrs >= 72:
                outcome = 1
            else:
                outcome = 0
        else:
            outcome = 2
        ventilator_files.append((evlp_id, os.path.join(FOLDER, file), outcome))

y = {}
X = []
for evlp_id, in_file_path, outcome in ventilator_files[4:16]:
    breath = pd.read_csv(in_file_path, header=0, usecols=["Timestamp", "Pressure", "Flow", "B_phase"])
    breath["Id"] = evlp_id
    X.append(breath)
    y[evlp_id] = outcome


X = pd.concat(X, ignore_index=True)
y = pd.Series(y)

print(X.shape)
print(y)

print("Extracting features...")

features = extract_relevant_features(X, y, column_id='Id', column_sort='Timestamp', ml_task='classification', n_jobs=1)

print("Saving features...")

print(features.shape)

# show the first five rows of the features dataframe
print(features.head(5))

# save the features dataframe to a csv file
features.to_csv("tf_features.csv")