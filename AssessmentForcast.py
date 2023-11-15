from EVLPCases import *
import numpy as np
import pandas as pd
import pickle

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


def get_assessment_periods(case, duration_tol=0.75, consecutive_breaths=20, noise_tol=3):
    """Get the assessment periods of the case."""
    assessment_periods = []
    # a valid assessment period should have at least 20 breaths with duration between 5.5 and 6.5 seconds
    breath_idx = 1
    while breath_idx <= case.ventilator_per_breath_ts.n_breaths - consecutive_breaths:
        durations = case.per_breath_param_table.param_table.loc[breath_idx:breath_idx + consecutive_breaths - 1,
                    "Duration(s)"]
        if abs(durations.loc[breath_idx] - 6) < duration_tol and np.sum(
                np.abs(durations - 6) < duration_tol) > consecutive_breaths - noise_tol:
            # assessment period ends at the last consecutive breath with duration between 5.5 and 6.5 seconds
            end_idx = breath_idx + consecutive_breaths - 1
            while (np.abs(case.per_breath_param_table.param_table.loc[end_idx + 1, "Duration(s)"] - 6) < duration_tol \
                   or np.sum(np.abs(
                        case.per_breath_param_table.param_table.loc[end_idx + 1:end_idx + consecutive_breaths,
                        "Duration(s)"] - 6) < duration_tol) > noise_tol) \
                    and end_idx < case.ventilator_per_breath_ts.n_breaths - 1:
                end_idx += 1
            assessment_periods.append((breath_idx, end_idx))
            breath_idx = end_idx + 1
        else:
            breath_idx += 1
    return assessment_periods


# assessment_periods_per_case = {"Good": {}, "OK": {}, "Bad": {}}
with open("assessment_periods_per_case.pkl", "rb") as f:
    assessment_periods_per_case = pickle.load(f)

for case in evlp_cases.cases.values():
    if case.case_id in assessment_periods_per_case["Good"]:
        print(f"Case {case.case_id} already marked as Good.")
        continue
    elif case.case_id in assessment_periods_per_case["OK"]:
        print(f"Case {case.case_id} already marked as OK.")
        continue
    elif case.case_id in assessment_periods_per_case["Bad"]:
        print(f"Case {case.case_id} already marked as Bad.")
        continue
    duration_tol = 0.75
    consecutive_breaths = 20
    niose_tol = 3
    while True:
        assessment_periods = get_assessment_periods(case, duration_tol, consecutive_breaths)
        print(case.case_id)
        print(assessment_periods)
        x = case.per_breath_param_table.param_table.index.to_numpy()
        y = case.per_breath_param_table.param_table["Duration(s)"].to_numpy()
        plt.plot(x, y)
        for start_idx, end_idx in assessment_periods:
            plt.axvspan(start_idx, end_idx, color="grey", alpha=0.5)
        plt.tight_layout()
        plt.show()
        option = input("Options:\n\tG: Good\n\tO: OK\n\tB: Bad\n\tAP: Adjust Parameters\nInput: ")
        if option in ["G", "O", "B"]:
            if option == "G":
                assessment_periods_per_case["Good"][case.case_id] = assessment_periods
                print(f"Case {case.case_id} marked as Good.")
            elif option == "O":
                assessment_periods_per_case["OK"][case.case_id] = assessment_periods
                print(f"Case {case.case_id} marked as OK.")
            else:
                assessment_periods_per_case["Bad"][case.case_id] = assessment_periods
                print(f"Case {case.case_id} marked as Bad.")
            break
        else:
            duration_tol = float(input("duration_tol: "))
            consecutive_breaths = int(input("consecutive_breaths: "))
            niose_tol = int(input("niose_tol: "))

    assessment_periods_per_case[case.case_id] = assessment_periods

with open("assessment_periods_per_case.pkl", "wb") as f:
    pickle.dump(assessment_periods_per_case, f)
