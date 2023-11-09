import numpy as np
import pandas as pd


class BreathLoader:
    """Raw data handler."""

    def __init__(self, in_file_path):
        """Read data from a CSV file and segment each breath."""
        self.in_file_path = in_file_path
        try:
            print(f"Reading data from {in_file_path}")
            breath = pd.read_csv(in_file_path, header=0, usecols=["Timestamp", "Pressure", "Flow", "B_phase"])
            # extract the columns as numpy arrays
            self.timestamp = breath['Timestamp'].to_numpy()
            self.pressure = breath['Pressure'].to_numpy()
            self.flow = breath['Flow'].to_numpy()
            self.phase = breath['B_phase'].to_numpy()
            # segment each breath
            diff_phase = np.diff(self.phase)
            self.boundary = np.where(diff_phase == 1)[0] + 1
            self.switch = np.where(diff_phase == -1)[0] + 1
            self.switch = self.switch[self.switch > self.boundary[0]]
            self.n_breaths = self.boundary.shape[0] - 1
            print(f"Successfully loaded {self.n_breaths} breaths from {in_file_path}\n")
        except Exception as e:
            print(f"Error reading file: {e}")
            exit()

    def get_breath_boundary(self, start_breath_index, end_breath_index=None):
        """Get the start and end data index of the breaths in the given range."""
        if end_breath_index is None:
            end_breath_index = start_breath_index
        start_breath_index = np.clip(start_breath_index, 0, self.n_breaths - 1)
        end_breath_index = np.clip(end_breath_index, 0, self.n_breaths - 1)
        start_idx = self.boundary[start_breath_index]
        end_idx = self.boundary[end_breath_index + 1]
        return start_idx, end_idx

    def get_breath_data(self, start_breath_index, end_breath_index=None):
        """Get the partial data in the given breaths range."""
        start_idx, end_idx = self.get_breath_boundary(start_breath_index, end_breath_index)
        return self.timestamp[start_idx:end_idx], self.pressure[start_idx:end_idx], self.flow[start_idx:end_idx]

    def get_phase(self, breath_index):
        """Get the inhaling and exhaling data of the breath at the given index."""
        start_idx, end_idx = self.get_breath_boundary(breath_index)
        mid_idx = self.switch[breath_index]
        return start_idx, mid_idx, end_idx

    def find_breath_by_index(self, index):
        return np.searchsorted(self.boundary, index, side='right') - 1

    def find_breath_by_timestamp(self, timestamp):
        clicked_index = np.searchsorted(self.timestamp, timestamp, side='left')
        return self.find_breath_by_index(clicked_index)

    def dynamic_compliance(self, breath_index):
        """Get the dynamic compliance of the breath at the given index."""
        start_idx, mid_idx, end_idx = self.get_phase(breath_index)
        in_timestamp = self.timestamp[start_idx:mid_idx]
        in_pressure = self.pressure[start_idx:mid_idx]
        ex_flow = self.flow[mid_idx:end_idx]

        ex_vol = np.trapz(ex_flow, in_timestamp) / 100000
        dy_comp = - ex_vol / (np.max(in_pressure) - 5)
        return dy_comp

    def calc_params(self, breath_index):
        """Get the volume of the breath at the given index."""
        start_idx, mid_idx, end_idx = self.get_phase(breath_index)
        in_timestamp = self.timestamp[start_idx:mid_idx]
        ex_timestamp = self.timestamp[mid_idx:end_idx]
        in_pressure = self.pressure[start_idx:mid_idx]
        in_flow = self.flow[start_idx:mid_idx]
        ex_flow = self.flow[mid_idx:end_idx]

        params = {}
        params["Max_gap(ms)"] = np.max(np.diff(self.timestamp[start_idx:end_idx]))
        params["In_vol(ml)"] = np.trapz(in_flow, in_timestamp) / 100000
        params["Ex_vol(ml)"] = np.trapz(ex_flow, ex_timestamp) / 100000
        params["IE_vol_ratio"] = - params["In_vol(ml)"] / params["Ex_vol(ml)"]
        params["Duration(s)"] = (self.timestamp[end_idx] - self.timestamp[start_idx]) / 10000
        params["In_duration(s)"] = (self.timestamp[mid_idx] - self.timestamp[start_idx]) / 10000
        params["Ex_duration(s)"] = (self.timestamp[end_idx] - self.timestamp[mid_idx]) / 10000
        params["IE_duration_ratio"] = params["In_duration(s)"] / params["Ex_duration(s)"]
        params["P_peak"] = np.max(in_pressure) / 100
        params["F_min"] = min(np.min(in_flow), np.min(ex_flow)) / 100
        params["PEEP"] = self.pressure[start_idx - 1] / 100
        params["Dy_comp"] = - params["Ex_vol(ml)"] / (params["P_peak"] - 5)
        return params


if __name__ == "__main__":\

    import os
    import re


    FOLDER = r"C:\Users\yiyan\WorkSpace\UHN\ventilator converted files"


    def get_files_in_folder(folder):
        files = {}
        for file in os.listdir(folder):
            if file.endswith(".csv"):
                EVLP_id = re.search(r"EVLP\d+", file).group()
                files[EVLP_id] = file
        return files

    ventilator_files = get_files_in_folder(FOLDER)

    outcome_df = pd.read_csv(r"C:\Users\yiyan\WorkSpace\UHN\EVLP data\EVLP#1-879_disposition.csv",
                             header=0, usecols=["Evlp Id No", "Extubation Time Hrs"],
                             dtype={"Evlp Id No": np.int16, "Extubation Time Hrs": np.float32})
    F_min_mean = []
    outcomes = []
    for EVLP_id, file in get_files_in_folder(FOLDER).items():
        try:
            evlp_id = int(EVLP_id[4:])
            print(evlp_id)
            INFILE = os.path.join(FOLDER, file)
            breath = BreathLoader(INFILE)
            F_mins = []
            for i in range(breath.n_breaths):
                params = breath.calc_params(i)
                F_mins.append(params["F_min"])
            F_min_mean.append(np.mean(F_mins))
            if evlp_id in outcome_df["Evlp Id No"].values:
                hrs = outcome_df.loc[outcome_df["Evlp Id No"] == evlp_id, "Extubation Time Hrs"].values[0]
                if hrs >= 72:
                    outcomes.append(1)
                else:
                    outcomes.append(0)
            else:
                outcomes.append(2)
        except Exception as e:
            print(f"Error reading file: {e}")
            continue

    print(len(F_min_mean))
    print(np.mean(F_min_mean))

    import matplotlib.pyplot as plt
    plt.plot(F_min_mean, outcomes, 'o')
    plt.show()


