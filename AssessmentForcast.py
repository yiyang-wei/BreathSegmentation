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


def supervise_assessment_periods(assessment_periods_per_case = None):
    if assessment_periods_per_case is None:
        assessment_periods_per_case = {"Good": {}, "OK": {}, "Bad": {}}
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
            assessment_periods = get_assessment_periods(case, duration_tol, consecutive_breaths, niose_tol)
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


# assessment_periods_per_case = {"Good": {}, "OK": {}, "Bad": {}}
with open("assessment_periods_per_case.pkl", "rb") as f:
    assessment_periods_per_case = pickle.load(f)

clean_cases = assessment_periods_per_case["Good"]
min_len0 = min([periods[0][1] - periods[0][0] for periods in clean_cases.values()])
min_len1 = min([periods[1][1] - periods[1][0] for periods in clean_cases.values()])
print(min_len0)
print(min_len1)

# create training data using the clean cases
cases_id = []
X = []
y = []
for case_id, assessment_periods in clean_cases.items():
    cases_id.append(case_id)
    case = evlp_cases.get_case(case_id)
    assessment_1_start_breath_idx = assessment_periods[0][0]
    assessment_2_start_breath_idx = assessment_periods[1][0]

    X.append(case.per_breath_param_table.get_params_in_range(assessment_1_start_breath_idx - 20, assessment_1_start_breath_idx + 40, params=["Dy_comp"]).to_numpy())
    # X.append(case.ventilator_per_breath_ts.get_breath_ts(assessment_1_start_breath_idx, assessment_1_start_breath_idx + 30).to_numpy())
    y.append(case.per_breath_param_table.get_params_in_range(assessment_2_start_breath_idx, assessment_2_start_breath_idx + 30, params=["Dy_comp"]).to_numpy())

X = np.array(X)
y = np.array(y)
print(X.shape)
print(y.shape)

# train with tsMixer
from tsMixer import *

# create training and validation datasets with leave-one-out
maes = []

# clean models folder to empty
import shutil
shutil.rmtree("./models")

# create models if not exist
if not os.path.exists("./models"):
    os.mkdir("./models")

good_preds = []
for i in range(X.shape[0]):
    train_X = np.concatenate((X[:i], X[i+1:]), axis=0)
    train_y = np.concatenate((y[:i], y[i+1:]), axis=0)
    val_X = X[i:i+1]
    val_y = y[i:i+1]

    # build datasets with the first column as batch
    train_dataset = tf.data.Dataset.from_tensor_slices((train_X, train_y)).batch(1)
    val_dataset = tf.data.Dataset.from_tensor_slices((val_X, val_y)).batch(1)

    model = build_model(
        input_shape=train_X.shape[1:],
        pred_len=30,
        norm_type='B',
        activation='relu',
        n_block=6,
        dropout=0,
        ff_dim=13,
        target_slice=None,
    )

    loss_fn = tf.keras.losses.MeanSquaredError()
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
    epochs = 200
    patience = 10
    min_delta = 0.001
    model_dir = f'./models/case{i}'

    # train
    model = train_model(
        model,
        train_dataset,
        val_dataset,
        loss_fn,
        optimizer,
        epochs,
        patience,
        min_delta,
        model_dir=model_dir
    )

    # evaluate
    model.evaluate(val_dataset)

    # evaluate with MAE
    pred = model.predict(val_dataset)
    maes.append(np.mean(np.abs(pred - val_y)))
    print(maes[-1])

    if maes[-1] < 10:
        good_preds.append(pred)

    # plot
    # plt.plot(pred[0], label="pred")
    # plt.plot(val_y[0], label="true")
    # plt.legend()
    # plt.show()

print(good_preds)

print(maes)
print(np.mean(maes))

# plot maes on std of y
stds = np.std(y, axis=1)
plt.scatter(stds, maes)
plt.xlabel("std of y")
plt.ylabel("mae")
plt.show()

# remove outliers
outlier_idx = np.where(np.array(maes) > np.std(maes) * 2 + np.mean(maes))[0]
print(outlier_idx)
maes = np.delete(maes, outlier_idx)
print(np.mean(maes))


