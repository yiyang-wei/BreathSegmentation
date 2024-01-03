import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tsfresh.feature_extraction.feature_calculators import index_mass_quantile, energy_ratio_by_chunks, autocorrelation
import os
import re
import pickle

NOT_TRANPLANTED = -999


class VentilatorRawTS:
    """Raw ventilator data handler."""

    def __init__(self, raw_ventilator_file_path, cols=None):
        """Read data from a CSV file and segment each breath."""
        self.in_file_path = raw_ventilator_file_path
        if cols is None:
            self.cols = ["Timestamp", "Pressure", "Flow", "B_phase"]
        print(f"Reading data from {raw_ventilator_file_path}")
        self.raw_df = pd.read_csv(raw_ventilator_file_path, header=0, usecols=self.cols)
        # check if there is any NaN
        if self.raw_df.isnull().values.any():
            num_nan_rows = self.raw_df.isnull().sum().sum()
            print(f"Warning: NaN found, dropping {num_nan_rows} rows")
            self.raw_df.dropna(inplace=True)
        self.length = self.raw_df.shape[0]
        print(f"Successfully loaded {self.length} data points")

    def get_ts_in_range(self, start_index, end_index, cols=None):
        """Get the partial data in the given range."""
        start_index = np.clip(start_index, 0, self.length - 1)
        end_index = np.clip(end_index, 0, self.length - 1)
        if cols is None:
            cols = self.cols
        return self.raw_df.loc[start_index:end_index-1, cols]

    def find_index_by_timestamp(self, timestamp):
        """Find the index of the data point with the given timestamp."""
        return np.searchsorted(self.raw_df["Timestamp"], timestamp, side='left')

    def plot(self, start_index=0, end_index=None, cols=None):
        """Plot the partial data in the given range."""
        if end_index is None:
            end_index = self.length
        start_index = np.clip(start_index, 0, self.length - 1)
        end_index = np.clip(end_index, 0, self.length - 1)
        if cols is None:
            cols = ["Flow", "Pressure"]
        # use timestamp as x-axis
        x = self.raw_df["Timestamp"][start_index:end_index].to_numpy()
        for col in cols:
            plt.subplot(len(cols), 1, cols.index(col) + 1)
            y = self.raw_df[col][start_index:end_index].to_numpy()
            plt.plot(x, y, label=col)
            plt.legend()
        plt.tight_layout()
        plt.show()



class VentilatorPerBreathTS:
    """Per breath ventilator data handler."""

    def __init__(self, raw_ventilator_ts):
        """Read data from a CSV file and segment each breath."""
        self.raw_ts = raw_ventilator_ts
        # segment each breath
        diff_phase = np.diff(self.raw_ts.raw_df["B_phase"])
        boundary = np.where(diff_phase == 1)[0] + 1
        switch = np.where(diff_phase == -1)[0] + 1
        if switch[0] < boundary[0]:
            switch = switch[1:]
        self.n_breaths = boundary.shape[0] - 1
        self.breath_segments = pd.DataFrame()
        self.breath_segments["Breath"] = np.arange(self.n_breaths) + 1
        self.breath_segments["Start"] = boundary[:-1]
        self.breath_segments["Mid"] = switch[:self.n_breaths]
        self.breath_segments["End"] = boundary[1:]
        self.breath_segments.set_index("Breath", inplace=True)

        # breath id for each data point
        self.breath_id = np.zeros(self.raw_ts.length, dtype=int)
        for i in range(self.n_breaths):
            start_idx, end_idx = self.get_breath_boundary(i + 1)
            self.breath_id[start_idx:end_idx] = i + 1

        print(f"Successfully segmented {self.n_breaths} breaths")

    def get_breath_boundary(self, start_breath_index, end_breath_index=None):
        """Get the start and end data index of the breaths in the given range."""
        if end_breath_index is None:
            end_breath_index = start_breath_index
        start_breath_index = np.clip(start_breath_index, 1, self.n_breaths)
        end_breath_index = np.clip(end_breath_index, 1, self.n_breaths)
        start_idx = self.breath_segments.loc[start_breath_index, "Start"]
        end_idx = self.breath_segments.loc[end_breath_index, "End"]
        return start_idx, end_idx

    def get_breath_ts(self, start_breath_index, end_breath_index=None):
        """Get the partial data in the given breaths range."""
        start_idx, end_idx = self.get_breath_boundary(start_breath_index, end_breath_index)
        return self.raw_ts.get_ts_in_range(start_idx, end_idx)

    def get_segments(self, breath_index):
        """Get the segmentation points of the breath at the given index."""
        start_idx = self.breath_segments.loc[breath_index, "Start"]
        mid_idx = self.breath_segments.loc[breath_index, "Mid"]
        end_idx = self.breath_segments.loc[breath_index, "End"]
        return start_idx, mid_idx, end_idx

    def get_segments_ts(self, breath_index):
        """Get the segmentation points of the breath at the given index."""
        start_idx, mid_idx, end_idx = self.get_segments(breath_index)
        in_haling_ts = self.raw_ts.get_ts_in_range(start_idx, mid_idx)
        ex_haling_ts = self.raw_ts.get_ts_in_range(mid_idx, end_idx)
        return in_haling_ts, ex_haling_ts

    def find_breath_by_index(self, index):
        return np.searchsorted(self.breath_segments["Start"], index, side='right')

    def find_breath_by_timestamp(self, timestamp):
        clicked_index = self.raw_ts.find_index_by_timestamp(timestamp)
        return self.find_breath_by_index(clicked_index)


class PerBreathParamTable:
    """Per breath parameters table handler."""

    def __init__(self, per_breath_ts):
        """Read data from a CSV file and segment each breath."""
        self.raw_ts = per_breath_ts.raw_ts
        self.per_breath_ts = per_breath_ts
        self.param_table = pd.DataFrame()
        self.param_table["Breath"] = np.arange(self.per_breath_ts.n_breaths, dtype=int) + 1
        self.param_table.set_index("Breath", inplace=True)
        for i in self.param_table.index:
            params = self.calc_params(i)
            for key, value in params.items():
                self.param_table.loc[i, key] = value

        print(f"Successfully calculated {self.param_table.shape[1]} parameters for {self.per_breath_ts.n_breaths} breaths")

    def calc_params(self, breath_index):
        """Get the volume of the breath at the given index."""
        start_idx, mid_idx, end_idx = self.per_breath_ts.get_segments(breath_index)
        in_timestamp = self.raw_ts.raw_df["Timestamp"][start_idx:mid_idx].to_numpy()
        ex_timestamp = self.raw_ts.raw_df["Timestamp"][mid_idx:end_idx].to_numpy()
        whole_timestamp = self.raw_ts.raw_df["Timestamp"][start_idx:end_idx].to_numpy()
        start_timestamp = self.raw_ts.raw_df["Timestamp"][start_idx]
        mid_timestamp = self.raw_ts.raw_df["Timestamp"][mid_idx]
        end_timestamp = self.raw_ts.raw_df["Timestamp"][end_idx]
        whole_pressure = self.raw_ts.raw_df["Pressure"][start_idx:end_idx].to_numpy()
        in_flow = self.raw_ts.raw_df["Flow"][start_idx:mid_idx].to_numpy()
        ex_flow = self.raw_ts.raw_df["Flow"][mid_idx:end_idx].to_numpy()
        whole_flow = self.raw_ts.raw_df["Flow"][start_idx:end_idx].to_numpy()

        params = {}
        params["Max_gap(ms)"] = np.max(np.diff(whole_timestamp))
        params["In_vol(ml)"] = np.trapz(in_flow, in_timestamp) / 100000
        params["Ex_vol(ml)"] = np.trapz(ex_flow, ex_timestamp) / 100000
        params["IE_vol_ratio"] = - params["In_vol(ml)"] / params["Ex_vol(ml)"]
        params["Duration(s)"] = (end_timestamp - start_timestamp) / 10000
        params["In_duration(s)"] = (mid_timestamp - start_timestamp) / 10000
        params["Ex_duration(s)"] = (end_timestamp - mid_timestamp) / 10000
        params["IE_duration_ratio"] = params["In_duration(s)"] / params["Ex_duration(s)"]
        params["P_peak"] = np.max(whole_pressure) / 100
        params["P_min"] = np.min(whole_pressure) / 100
        params["P_mean"] = np.mean(whole_pressure) / 100
        params["F_peak"] = np.max(whole_flow) / 100
        params["F_min"] = np.min(whole_flow) / 100
        params["F_mean"] = np.mean(whole_flow) / 100
        # params["Flow_index_mass_quantile_70"] = index_mass_quantile(whole_flow, [{'q': 0.7}])
        # params["Pressure_index_mass_quantile_70"] = index_mass_quantile(whole_pressure, [{'q': 0.7}])
        # params["Flow_energy_ratio_by_chunks_10_9"] = energy_ratio_by_chunks(whole_flow, [{'num_segments': 10, 'segment_focus': 9}])
        # params["Pressure_energy_ratio_by_chunks_10_9"] = energy_ratio_by_chunks(whole_pressure, [{'num_segments': 10, 'segment_focus': 9}])
        params["Flow_autocorrelation_3"] = autocorrelation(whole_flow, 3)
        params["Pressure_autocorrelation_3"] = autocorrelation(whole_pressure, 3)
        params["PEEP"] = whole_pressure[-1] / 100
        params["Dy_comp"] = - params["Ex_vol(ml)"] / (params["P_peak"] - 5)
        return params

    def get_params(self, breath_index, params=None):
        """Get the parameters of the breath at the given index."""
        if params is None:
            params = self.param_table.columns
        return self.param_table.loc[breath_index, params]

    def get_params_in_range(self, start_breath_index, end_breath_index=None, params=None):
        """Get the parameters of the breaths in the given range."""
        if end_breath_index is None:
            return self.get_params(start_breath_index, params)
        start_breath_index = np.clip(start_breath_index, 1, self.per_breath_ts.n_breaths)
        end_breath_index = np.clip(end_breath_index, 1, self.per_breath_ts.n_breaths)
        if params is None:
            params = self.param_table.columns
        return self.param_table.loc[start_breath_index:end_breath_index-1, params]

    def plot(self, start_breath_index=1, end_breath_index=None, params=None):
        """Plot the partial data in the given range."""
        if end_breath_index is None:
            end_breath_index = self.per_breath_ts.n_breaths
        start_breath_index = np.clip(start_breath_index, 1, self.per_breath_ts.n_breaths)
        end_breath_index = np.clip(end_breath_index, 1, self.per_breath_ts.n_breaths)
        if params is None:
            params = ["Duration(s)", "Dy_comp"]
        for param in params:
            plt.subplot(len(params), 1, params.index(param) + 1)
            x = self.param_table.index[start_breath_index:end_breath_index]
            y = self.param_table[param][start_breath_index:end_breath_index]
            y_5 = np.percentile(y, 5)
            y_95 = np.percentile(y, 95)
            y_min = y_5 - (y_95 - y_5) * 0.15
            y_max = y_95 + (y_95 - y_5) * 0.15
            plt.ylim(y_min, y_max)
            plt.plot(x, y, label=param)
            plt.legend()
        plt.tight_layout()
        plt.show()



class EVLPCase:

    def __init__(self, case_id,
                 donor_info, recipient_info,
                 ventilator_raw_ts, ventilator_per_breath_ts,
                 per_breath_param_table,
                 transplant_outcome):
        self.case_id = case_id
        self.donor_info = donor_info
        self.recipient_info = recipient_info
        self.ventilator_raw_ts = ventilator_raw_ts
        self.ventilator_per_breath_ts = ventilator_per_breath_ts
        self.per_breath_param_table = per_breath_param_table
        self.transplant_outcome = transplant_outcome

    def print_case_info(self):
        print(f"==================== Case {self.case_id} ====================")
        print("Donor Info:")
        print(self.donor_info)
        print()
        print("Recipient Info:")
        print(self.recipient_info)
        print()
        print("Transplant Outcome:")
        if self.transplant_outcome == NOT_TRANPLANTED:
            print("Not transplanted")
        else:
            print(self.transplant_outcome)
        print()
        print("Raw Ventilator Time Series:")
        print(self.ventilator_raw_ts.raw_df)
        print()
        print("Per Breath Ventilator Time Series:")
        print(self.ventilator_per_breath_ts.breath_segments)
        print()
        print("Per Breath Parameters Table:")
        print(self.per_breath_param_table.param_table)
        print()


class EVLPCases:

    def __init__(self, ventilator_raw_ts_folder,
                 donor_info_file_path, recipient_info_file_path,
                 transplant_outcome_file_path,
                 processed_case_save_folder,
                 use_cases=None, force_replace=False):
        self.donor_info = None if donor_info_file_path is None else pd.read_csv(donor_info_file_path, header=0, index_col=0)
        self.recipient_info = None if recipient_info_file_path is None else pd.read_csv(recipient_info_file_path, header=0, index_col=0)
        self.transplant_outcome = None if transplant_outcome_file_path is None else pd.read_csv(transplant_outcome_file_path, header=0, index_col=0)

        if not os.path.exists(processed_case_save_folder):
            os.makedirs(processed_case_save_folder)

        self.cases = {}
        for file_name in os.listdir(ventilator_raw_ts_folder):
            if file_name.endswith(".csv"):
                case_id = int(re.search(r"EVLP\d+", file_name).group()[4:])
                if use_cases is not None and case_id not in use_cases:
                    continue
                print(f"Processing case {case_id}")
                if not force_replace and os.path.exists(os.path.join(processed_case_save_folder, f"EVLP{case_id}.pkl")):
                    print(f"Case {case_id} already exists, loading from file")
                    self.cases[case_id] = pd.read_pickle(os.path.join(processed_case_save_folder, f"EVLP{case_id}.pkl"))
                else:
                    raw_ts = VentilatorRawTS(os.path.join(ventilator_raw_ts_folder, file_name))
                    per_breath_ts = VentilatorPerBreathTS(raw_ts)
                    per_breath_param_table = PerBreathParamTable(per_breath_ts)
                    donor_info = None if self.donor_info is None else self.donor_info.loc[case_id]
                    recipient_info = None if self.recipient_info is None else self.recipient_info.loc[case_id]
                    if self.transplant_outcome is None:
                        transplant_outcome = None
                    elif case_id not in self.transplant_outcome.index:
                        transplant_outcome = NOT_TRANPLANTED
                    else:
                        transplant_outcome = self.transplant_outcome.loc[case_id, "Extubation Time Hrs"]
                    self.cases[case_id] = EVLPCase(case_id, donor_info, recipient_info,
                                                   raw_ts, per_breath_ts, per_breath_param_table,
                                                   transplant_outcome)
                    pickle.dump(self.cases[case_id], open(os.path.join(processed_case_save_folder, f"EVLP{case_id}.pkl"), "wb"))

                print(f"Successfully processed case {case_id}")
                print()

    def get_case(self, case_id):
        return self.cases[case_id]

    def get_cases(self, case_ids=None):
        if case_ids is None:
            return self.cases
        else:
            return {case_id: self.cases[case_id] for case_id in case_ids}


if __name__ == "__main__":
    DATA_FOLDER = r"..\EVLP Data"
    VENTILATOR_RAW_TS_FOLDER = os.path.join(r"raw ventilator ts")
    PARAM_TABLE_FOLDER = os.path.join(r"parameter table")
    DONOR_INFO_FILE_PATH = os.path.join(r"EVLP#1-879_donor.csv")
    RECIPIENT_INFO_FILE_PATH = None
    TRANSPLANT_OUTCOME_FILE_PATH = os.path.join(r"EVLP#1-879_disposition.csv")
    EVLPCASE_OBJECTS_SAVE_FOLDER = os.path.join(r"EVLP case objects")

    evlp_cases = EVLPCases(VENTILATOR_RAW_TS_FOLDER,
                           DONOR_INFO_FILE_PATH, RECIPIENT_INFO_FILE_PATH,
                           TRANSPLANT_OUTCOME_FILE_PATH,
                           EVLPCASE_OBJECTS_SAVE_FOLDER,
                           force_replace=True)

    # # check if parameters table folder exists
    # if not os.path.exists(PARAM_TABLE_FOLDER):
    #     os.makedirs(PARAM_TABLE_FOLDER)
    #
    # for case in evlp_cases.cases.values():
    #     # save the parameters table to a csv file
    #     case.per_breath_param_table.param_table.to_csv(os.path.join(PARAM_TABLE_FOLDER, f"EVLP{case.case_id}.csv"))

    case = evlp_cases.get_case(818)
    case.print_case_info()

