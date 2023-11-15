from EVLPCases import *
import numpy as np
import pandas as pd


VENTILATOR_RAW_TS_FOLDER = r"C:\Users\weiyi\Workspace\UHN\ventilator data"
DONOR_INFO_FILE_PATH = r"C:\Users\weiyi\Workspace\UHN\EVLP Data\EVLP#1-879_donor.csv"
RECIPIENT_INFO_FILE_PATH = None
TRANSPLANT_OUTCOME_FILE_PATH = r"C:\Users\weiyi\Workspace\UHN\EVLP Data\EVLP#1-879_disposition.csv"
PROCESSED_CASE_SAVE_FOLDER = r"C:\Users\weiyi\Workspace\UHN\processed evlp cases"

cases = [551, 553, 554, 555, 557, 558, 560, 563, 564, 565, 568, 572, 573,
         574, 575, 577, 579, 592, 593, 595, 598, 600, 603, 610, 615, 616,
         617, 618, 619, 631, 682, 685, 686, 694, 698, 730, 731, 736,
         738, 753, 762, 782, 803, 817, 818]


evlp_cases = EVLPCases(VENTILATOR_RAW_TS_FOLDER,
                       DONOR_INFO_FILE_PATH, RECIPIENT_INFO_FILE_PATH,
                       TRANSPLANT_OUTCOME_FILE_PATH,
                       PROCESSED_CASE_SAVE_FOLDER, cases)


def get_assessment_periods(case):
    """Get the assessment periods of the case."""
    assessment_periods = []
    # a valid assessment period should have at least 20 breaths with duration between 5.5 and 6.5 seconds
    breath_idx = 1
    while breath_idx <= case.ventilator_per_breath_ts.n_breaths - 20:
        durations = case.per_breath_param_table.param_table.loc[breath_idx:breath_idx+19, "Duration(s)"]
        if 5.5 < durations.loc[breath_idx] < 6.5 and np.sum(np.abs(durations - 6) < 0.5) > 10:
            # assessment period ends at the last consecutive breath with duration between 5.5 and 6.5 seconds
            end_idx = breath_idx + 19
            while np.any(np.abs(case.per_breath_param_table.param_table.loc[end_idx+1:end_idx+20, "Duration(s)"] - 6) < 0.5)\
                    and end_idx < case.ventilator_per_breath_ts.n_breaths - 1:
                end_idx += 1
            assessment_periods.append((breath_idx, end_idx))
            breath_idx = end_idx + 1
        else:
            breath_idx += 1
    return assessment_periods


for case in evlp_cases.cases.values():
    assessment_periods = get_assessment_periods(case)
    print(case.case_id)
    print(assessment_periods)
    if len(assessment_periods) != 3:
        x = case.ventilator_raw_ts.raw_df["Timestamp"].to_numpy()
        y = case.ventilator_raw_ts.raw_df["Flow"].to_numpy()
        plt.plot(x, y)
        for start_idx, end_idx in assessment_periods:
            plt.axvspan(x[case.ventilator_per_breath_ts.breath_segments.loc[start_idx, "Start"]],
                        x[case.ventilator_per_breath_ts.breath_segments.loc[end_idx, "End"]],
                        color="grey", alpha=0.5)
        plt.tight_layout()
        plt.show()