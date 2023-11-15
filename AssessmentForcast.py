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
         617, 618, 619, 621, 631, 682, 685, 686, 694, 698, 730, 731, 736,
         738, 753, 762, 782, 803, 817, 818]


evlp_cases = EVLPCases(VENTILATOR_RAW_TS_FOLDER,
                       DONOR_INFO_FILE_PATH, RECIPIENT_INFO_FILE_PATH,
                       TRANSPLANT_OUTCOME_FILE_PATH,
                       PROCESSED_CASE_SAVE_FOLDER, cases)


def get_assessment_periods(case, tol=0.5, consecutive_breaths=20):
    """Get the assessment periods of the case."""
    assessment_periods = []
    # a valid assessment period should have at least 20 breaths with duration between 5.5 and 6.5 seconds
    breath_idx = 1
    while breath_idx <= case.ventilator_per_breath_ts.n_breaths - consecutive_breaths:
        durations = case.per_breath_param_table.param_table.loc[breath_idx:breath_idx+consecutive_breaths-1, "Duration(s)"]
        if abs(durations.loc[breath_idx] - 6) < tol and np.all(np.abs(durations - 6) < tol):
            # assessment period ends at the last consecutive breath with duration between 5.5 and 6.5 seconds
            end_idx = breath_idx + consecutive_breaths - 1
            while np.any(np.abs(case.per_breath_param_table.param_table.loc[end_idx+1:end_idx+consecutive_breaths, "Duration(s)"] - 6) < tol)\
                    and end_idx < case.ventilator_per_breath_ts.n_breaths - 1:
                end_idx += 1
            assessment_periods.append((breath_idx, end_idx))
            breath_idx = end_idx + 1
        else:
            breath_idx += 1
    return assessment_periods


assessment_periods_per_case = {}
for case in evlp_cases.cases.values():
    tol = 0.5
    consecutive_breaths = 20
    while True:
        assessment_periods = get_assessment_periods(case, tol, consecutive_breaths)
        print(case.case_id)
        print(assessment_periods)
        x = case.per_breath_param_table.param_table.index.to_numpy()
        y = case.per_breath_param_table.param_table["Duration(s)"].to_numpy()
        plt.plot(x, y)
        for start_idx, end_idx in assessment_periods:
            plt.axvspan(start_idx, end_idx, color="grey", alpha=0.5)
        plt.tight_layout()
        plt.show()
        option = input("Accept? (y/n/skip) ")
        if option == "y":
            break
        elif option == "skip":
            break
        else:
            tol = float(input("tol: "))
            consecutive_breaths = int(input("consecutive_breaths: "))

    assessment_periods_per_case[case.case_id] = assessment_periods


print(assessment_periods_per_case)
with open("assessment_periods_per_case.pkl", "wb") as f:
    pickle.dump(assessment_periods_per_case, f)