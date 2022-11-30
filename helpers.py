"""Some helper functions for project 1."""
import csv
import numpy as np


def load_csv_data(data_path, sub_sample=False):
    """Loads data and returns y (class labels), tX (features) and ids (event ids)"""
    y = np.genfromtxt(data_path, delimiter=",", skip_header=1, dtype=str, usecols=1)
    x = np.genfromtxt(data_path, delimiter=",", skip_header=1)
    ids = x[:, 0].astype(np.int)
    input_data = x[:, 2:]

    # convert class labels from strings to binary (-1,1)
    yb = np.ones(len(y))
    yb[np.where(y == "b")] = 0
    yb[np.where(y == "-1.0")] = 0

    # sub-sample
    if sub_sample:
        yb = yb[::50]
        input_data = input_data[::50]
        ids = ids[::50]

    return yb, input_data, ids


def create_csv_submission(ids, y_pred, name):
    """
    Creates an output file in .csv format for submission to Kaggle or AIcrowd
    Arguments: ids (event ids associated with each prediction)
               y_pred (predicted class labels)
               name (string name of .csv output file to be created)
    """
    with open(name, "w") as csvfile:
        fieldnames = ["Id", "Prediction"]
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for r1, r2 in zip(ids, y_pred):
            writer.writerow({"Id": int(r1), "Prediction": int((r2 - 0.5) * 2)})


def write_csv_file(x, y, idx, name):
    res = [[int(id)] + [n] + list(m) for m, n, id in zip(x, y, idx)]
    print(res[:5])
    with open(name, "w") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(res)


def clip(fit_data, upper=99, lower=0):
    upper_clip = np.percentile(fit_data, upper)
    lower_clip = np.percentile(fit_data, lower)
    return np.clip(fit_data, lower_clip, upper_clip)


def clip_outliers(X_train, X_test):
    """
    Clip outliers for each column with customisation based on the inspective result of their distribution
    """
    f = open("data/train.csv")
    column_names = f.readline().strip().split(",")[2:]
    f.close()
    ls_99 = [
        "DER_mass_MMC",
        "DER_mass_transverse_met_lep",
        "DER_mass_vis",
        "DER_pt_h",
        "DER_deltaeta_jet_jet",
        "DER_mass_jet_jet",
        "DER_pt_tot",
        "PRI_tau_pt",
        "PRI_met",
        "PRI_jet_leading_pt",
        "PRI_jet_all_pt",
    ]
    ls_999 = ["DER_sum_pt", "DER_pt_ratio_lep_tau", "PRI_met_sumet"]
    ls_199 = ["DER_prodeta_jet_jet"]
    for name in ls_99:
        col = X_train[:, column_names.index(name)]
        col = clip(col, 99, 0)
        X_train[:, column_names.index(name)] = col

        col = X_test[:, column_names.index(name)]
        col = clip(col, 99, 0)
        X_test[:, column_names.index(name)] = col

    for name in ls_999:
        col = X_train[:, column_names.index(name)]
        col = clip(col, 99.9, 0)
        X_train[:, column_names.index(name)] = col

        col = X_test[:, column_names.index(name)]
        col = clip(col, 99.9, 0)
        X_test[:, column_names.index(name)] = col

    for name in ls_199:
        col = X_train[:, column_names.index(name)]
        col = clip(col, 99, 1)
        X_train[:, column_names.index(name)] = col

        col = X_test[:, column_names.index(name)]
        col = clip(col, 99, 1)
        X_test[:, column_names.index(name)] = col
    # col = prep.transfrom(col, type = "clip", param = 1.0, upper = 99, lower = 0)
    return X_train, X_test
