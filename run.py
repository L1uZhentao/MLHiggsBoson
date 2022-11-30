import numpy as np
from implementations import *
from helpers import *
from preprocess import *
import csv

y_train, X_train, ids = load_csv_data("data/train.csv")
_, X_test, ids_test = load_csv_data("data/test.csv")

f = open("data/train.csv")
column_names = f.readline().strip().split(",")[2:]
f.close()

# replace empty value -999 by nan.
X_train[X_train == -999] = np.nan
X_test[X_test == -999] = np.nan


def jet_num_subset(X, y):
    """
    Based on jet_num, we split the dataset into 3 subsets, with jet_num = 0, 1, and 2 or more.
    """
    zero = X[:, 22] == 0
    one = X[:, 22] == 1
    others = X[:, 22] >= 2

    X_others, y_others = X[others], y[others]
    X_zero, y_zero = X[zero], y[zero]
    X_one, y_one = X[one], y[one]

    return X_zero, y_zero, X_one, y_one, X_others, y_others


# Split test and train datasets by jet numbs.
(
    X_train_zero,
    y_train_zero,
    X_train_one,
    y_train_one,
    X_train_other,
    y_train_other,
) = jet_num_subset(X_train, y_train)
(
    X_test_zero,
    ids_test_zero,
    X_test_one,
    ids_test_one,
    X_test_other,
    ids_test_other,
) = jet_num_subset(X_test, ids_test)


def clip_outliers_zero(X_train, X_test):
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
        col = clip(col, 99, 0)
        X_train[:, column_names.index(name)] = col
        col = X_test[:, column_names.index(name)]
        col = clip(col, 99, 0)
        X_test[:, column_names.index(name)] = col

    for name in ls_199:
        col = X_train[:, column_names.index(name)]
        col = clip(col, 99, 1)
        X_train[:, column_names.index(name)] = col
        col = X_test[:, column_names.index(name)]
        col = clip(col, 99, 1)
        X_test[:, column_names.index(name)] = col
    return X_train, X_test


def clip_outliers_one(X_train, X_test):
    """
    Clip outliers for each column with customisation based on the inspective result of their distribution
    """
    # replace empty value -999 by nan.
    X_train[np.isnan(X_train)] = -999
    X_test[np.isnan(X_test)] = -999
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
    return X_train, X_test


X_train_zero, X_test_zero = clip_outliers_zero(X_train_zero, X_test_zero)
X_train_one, X_test_one = clip_outliers_one(X_train_one, X_test_one)
X_train_other, X_test_other = clip_outliers_one(X_train_other, X_test_other)


def missing_value_proportion(X):
    return np.count_nonzero(np.isnan(X), axis=0) / len(X)


def all_missing_indices(X):
    return X == 1


def no_missing_indices(X):
    return X == 0


def partial_missing_indices(X):
    return (X > 0) & (X < 1)


def feature_engg_X(X_train_set, X_test_set):
    """
    Performs some of our  feature engineering, replaces missing values and adds the binary encoding.
    """
    # replace empty value -999 by nan.
    X_train_set[X_train_set == -999] = np.nan
    X_test_set[X_test_set == -999] = np.nan

    # Get the missing value proportion for each column
    X_train_set_missing_proportion = missing_value_proportion(X_train_set)

    # X_train_set
    X_partial_missing = partial_missing_indices(X_train_set_missing_proportion)

    # Store new column to indicate whether the value is missing or not
    new_col_X_train = np.where(np.isnan(X_train_set[:, X_partial_missing]), 0, 1)
    new_col_X_test = np.where(np.isnan(X_test_set[:, X_partial_missing]), 0, 1)

    # fill missing value with median
    X_train_set_median = np.nanmedian(X_train_set[:, X_partial_missing], axis=0)
    arr = X_train_set[:, X_partial_missing]
    arr[np.isnan(arr)] = X_train_set_median
    X_train_set[:, X_partial_missing] = arr

    arr = X_test_set[:, X_partial_missing]
    arr[np.isnan(arr)] = X_train_set_median
    X_test_set[:, X_partial_missing] = arr

    # drop the columns with all missing values for X_train and X_test
    X_zero_to_delete = all_missing_indices(X_train_set_missing_proportion)
    X_train_set = np.delete(X_train_set, X_zero_to_delete, axis=1)
    X_test_set = np.delete(X_test_set, X_zero_to_delete, axis=1)

    # Add the new column
    X_train_set = np.hstack([new_col_X_train, X_train_set])
    X_test_set = np.hstack([new_col_X_test, X_test_set])
    return X_train_set, X_test_set


X_train_zero, X_test_zero = feature_engg_X(X_train_zero, X_test_zero)
X_train_one, X_test_one = feature_engg_X(X_train_one, X_test_one)
X_train_other, X_test_other = feature_engg_X(X_train_other, X_test_other)


def add_interactive_term(X, ind1, ind2):
    return np.c_[X, X[:, ind1] * X[:, ind2]]


def add_to_set(X_train, X_test, pairs):
    X_train1 = X_train.copy()
    X_test1 = X_test.copy()
    for pair in pairs:
        X_train1 = add_interactive_term(X_train1, pair[0], pair[1])
        X_test1 = add_interactive_term(X_test1, pair[0], pair[1])
    return X_train1, X_test1


zero_interactive_pair = [
    (0, 2),
    (0, 3),
    (0, 5),
    (0, 7),
    (0, 8),
    (0, 10),
    (0, 13),
    (0, 18),
    (1, 2),
    (1, 3),
    (1, 5),
    (1, 7),
    (1, 8),
    (1, 9),
    (1, 10),
    (1, 13),
    (1, 16),
    (1, 18),
    (2, 3),
    (2, 5),
    (2, 7),
    (2, 8),
    (2, 9),
    (2, 10),
    (2, 11),
    (2, 16),
    (2, 18),
    (3, 5),
    (3, 8),
    (3, 9),
    (3, 13),
    (3, 16),
    (4, 6),
    (4, 9),
    (4, 10),
    (5, 7),
    (5, 8),
    (5, 9),
    (5, 10),
    (5, 13),
    (5, 16),
    (5, 18),
    (6, 9),
    (7, 8),
    (7, 10),
    (7, 13),
    (7, 15),
    (7, 16),
    (8, 9),
    (8, 10),
    (8, 16),
    (8, 18),
    (9, 13),
    (9, 16),
    (9, 18),
    (10, 13),
    (10, 15),
    (10, 16),
    (11, 14),
    (12, 15),
    (12, 17),
    (13, 14),
    (13, 15),
    (13, 16),
    (13, 17),
    (16, 18),
]
X_train_zero1, X_test_zero1 = add_to_set(
    X_train_zero, X_test_zero, zero_interactive_pair
)

one_interactive_pair = [
    (0, 2),
    (0, 3),
    (0, 4),
    (0, 5),
    (0, 9),
    (0, 10),
    (0, 16),
    (0, 18),
    (1, 2),
    (1, 3),
    (1, 4),
    (1, 5),
    (1, 6),
    (1, 8),
    (1, 9),
    (1, 10),
    (1, 13),
    (1, 16),
    (2, 3),
    (2, 4),
    (2, 5),
    (2, 6),
    (2, 7),
    (2, 9),
    (2, 10),
    (2, 13),
    (2, 16),
    (2, 17),
    (2, 20),
    (2, 23),
    (3, 4),
    (3, 5),
    (3, 6),
    (3, 7),
    (3, 8),
    (3, 9),
    (3, 10),
    (3, 13),
    (3, 16),
    (4, 5),
    (4, 6),
    (4, 7),
    (4, 8),
    (4, 9),
    (4, 18),
    (5, 6),
    (5, 7),
    (5, 8),
    (5, 9),
    (5, 10),
    (5, 13),
    (5, 16),
    (6, 10),
    (6, 12),
    (6, 16),
    (6, 18),
    (7, 8),
    (7, 9),
    (7, 10),
    (7, 11),
    (7, 16),
    (7, 20),
    (8, 9),
    (8, 10),
    (8, 13),
    (8, 16),
    (9, 10),
    (9, 13),
    (9, 16),
    (10, 11),
    (10, 13),
    (10, 16),
    (10, 17),
    (11, 14),
    (11, 20),
    (11, 21),
    (11, 23),
    (12, 17),
    (12, 18),
    (13, 16),
    (13, 21),
    (16, 22),
    (20, 23),
]

X_train_one1, X_test_one1 = add_to_set(X_train_one, X_test_one, one_interactive_pair)
other_interactive_pair = [
    (0, 2),
    (0, 3),
    (0, 4),
    (0, 5),
    (0, 7),
    (0, 8),
    (0, 9),
    (0, 12),
    (0, 20),
    (0, 27),
    (1, 2),
    (1, 3),
    (1, 4),
    (1, 5),
    (1, 6),
    (1, 8),
    (1, 11),
    (1, 12),
    (1, 14),
    (1, 20),
    (1, 24),
    (1, 27),
    (2, 3),
    (2, 4),
    (2, 5),
    (2, 7),
    (2, 8),
    (2, 9),
    (2, 12),
    (2, 13),
    (2, 20),
    (2, 24),
    (2, 28),
    (2, 29),
    (3, 4),
    (3, 8),
    (3, 10),
    (3, 11),
    (3, 12),
    (3, 13),
    (3, 14),
    (3, 17),
    (3, 20),
    (3, 27),
    (4, 5),
    (4, 6),
    (4, 7),
    (4, 8),
    (4, 12),
    (4, 14),
    (4, 19),
    (4, 22),
    (4, 24),
    (4, 29),
    (5, 6),
    (5, 7),
    (5, 10),
    (5, 13),
    (5, 14),
    (5, 17),
    (5, 19),
    (5, 23),
    (5, 24),
    (5, 27),
    (5, 30),
    (6, 7),
    (6, 9),
    (6, 10),
    (6, 12),
    (6, 14),
    (6, 17),
    (6, 22),
    (6, 23),
    (6, 25),
    (6, 27),
    (6, 30),
    (7, 9),
    (7, 14),
    (7, 19),
    (7, 20),
    (7, 25),
    (7, 27),
    (8, 10),
    (8, 11),
    (8, 12),
    (8, 13),
    (8, 14),
    (8, 17),
    (8, 19),
    (8, 20),
    (8, 21),
    (8, 22),
    (8, 30),
    (9, 12),
    (9, 14),
    (9, 19),
    (9, 20),
    (9, 21),
    (9, 22),
    (9, 23),
    (9, 26),
    (9, 30),
    (10, 11),
    (10, 14),
    (10, 17),
    (10, 24),
    (10, 30),
    (11, 12),
    (11, 14),
    (11, 15),
    (11, 17),
    (11, 18),
    (11, 20),
    (11, 21),
    (11, 22),
    (11, 28),
    (11, 30),
    (12, 13),
    (12, 14),
    (12, 20),
    (12, 25),
    (12, 27),
    (13, 14),
    (13, 17),
    (13, 22),
    (13, 23),
    (13, 26),
    (14, 17),
    (14, 20),
    (14, 22),
    (14, 23),
    (15, 17),
    (15, 18),
    (15, 25),
    (15, 28),
    (16, 25),
    (17, 20),
    (17, 30),
    (18, 19),
    (18, 25),
    (18, 27),
    (19, 26),
    (20, 21),
    (20, 24),
    (20, 27),
    (22, 23),
    (22, 26),
    (24, 27),
    (25, 28),
    (26, 27),
    (26, 28),
    (27, 30),
]

X_train_other1, X_test_other1 = add_to_set(
    X_train_other, X_test_other, other_interactive_pair
)


def accuracy_loss(y, prediction):
    """Computes 1-accuracy of prediction, and also the F1-score"""
    # y values are either -1 or 1.
    # So if prediction and y are both negative -> it's correct or if prediction and y both are positive -> it's correct
    accuracy = ((y < 0.5) == (prediction < 0.5)).sum() / y.shape[0]
    TPR = ((y < 0.5) & (prediction < 0.5)).sum()
    FNR = ((y < 0.5) & (prediction >= 0.5)).sum()
    FPR = ((y >= 0.5) & (prediction < 0.5)).sum()
    TNR = ((y >= 0.5) & (prediction >= 0.5)).sum()
    # Depending on what one labels as a "positive" and what as a "negative" a different one of the F1 scores is relevant here.
    F_1_score = 2 * TPR / (2 * TPR + FNR + FPR)
    alt_F_1_score = 2 * TNR / (2 * TNR + FNR + FPR)
    return 1 - accuracy, F_1_score, alt_F_1_score


import itertools  # In standard library so should be allowed?


def tune_hyperparamters(
    model, parameter_ranges, train_x, train_y, test_x, test_y, metric=compute_loss_mse
):
    """
    Given a model and a list of parameters to that model and their ranges,
    for each combination of parameters this model will train them on the training dataset and then evaluate them on the test set.
    It returns the best parameters and their corresponding loss.
    Expects the model to take inputs in the following day: y,X, and then the parameters specified in the input in the same order.
    """
    best_loss = None
    best_param = None
    best_w = None
    for param in itertools.product(*parameter_ranges):
        w_vec, train_loss = model(train_y, train_x, *param)
        validation_loss = metric(test_y, test_x @ w_vec)
        if model in [logistic_regression, reg_logistic_regression]:
            prediction = sigmoid(test_x @ w_vec)
            validation_loss = metric(test_y, prediction)

        if best_loss == None or validation_loss < best_loss:
            best_loss, best_param = (validation_loss, param)
            best_w = w_vec
    return best_param, best_loss, best_w


def tune_model(
    model,
    parameter_ranges,
    poly_degree,
    dimension,
    train_x,
    train_y,
    test_x,
    test_y,
    metric=compute_loss_mse,
):
    """
    model: The model to optimize
    parameter_ranges: The input parameters to the model and the ranges to search for those
    poly_degree: List of degrees to use for polynomial expansion
    dimension: Dimension to project data down to using PCA
    train_x,train_y,test_x,test_y: Train and test data
    metric: Metric to use to compare. E.g. compute_loss_mse. Should expect two inputs: y and prediction
    Will perform an extensive hyperparameter search and return the ones that give the lowest value in the metric.
    Expects the model to take inputs in the following day: y,X, and then the parameters specified in the input in the same order.
    Return: best_param, best_loss, degree, dimension
    The best parameters in the parameter_ranges, the loss achieved for those, the degree of the polynomial expansion for the best value and the dimension
    """
    best_loss = None
    best_param = None
    best_degree = None
    best_dimension = None
    best_w = None
    for degree in poly_degree:
        poly_mapped_train = build_poly(train_x, degree)
        poly_mapped_test = build_poly(test_x, degree)

        preproc = Preprocessor()
        preproc.fit(poly_mapped_train)
        poly_mapped_train = preproc.standardize(poly_mapped_train)
        poly_mapped_test = preproc.standardize(poly_mapped_test)

        for pca_dimension in dimension:
            if pca_dimension > poly_mapped_train.shape[1]:
                continue
            mapped_train, _ = preproc.PCA(poly_mapped_train, pca_dimension)
            mapped_test, _ = preproc.PCA(poly_mapped_test, pca_dimension)

            # Specifiy a constant inital w such that it is of the right size for our adjusted data
            cur_param_range = parameter_ranges.copy()
            for i in range(0, len(parameter_ranges)):
                if parameter_ranges[i] == "initial_w":
                    cur_param_range[i] = [np.zeros(mapped_train.shape[1])]
            # Tune all other parameters
            param, cur_loss, w = tune_hyperparamters(
                model,
                cur_param_range,
                mapped_train,
                train_y,
                mapped_test,
                test_y,
                metric,
            )
            if best_loss == None or cur_loss < best_loss:
                best_loss, best_param = (cur_loss, param)
                best_degree, best_dimension = degree, pca_dimension
                best_w = w
    return best_param, best_loss, best_degree, best_dimension, best_w


def predict(X_test, ids, w, X_train, dimension, poly_degree, logistic=False):
    """
    Performs the model prediction.
    Given test dataset, the corresponding IDs, weight vector w, train dataset, PCA target dimension, polynomial degree, and a Boolean to indicate if we should use the sigmoid function or not, this generates the predictions and returns them.
    """
    poly_mapped_test = build_poly(X_test, poly_degree)
    poly_mapped_train = build_poly(X_train, poly_degree)
    preproc = Preprocessor()
    preproc.fit(poly_mapped_train)
    poly_mapped_train = preproc.standardize(poly_mapped_train)
    poly_mapped_test = preproc.standardize(poly_mapped_test)
    mapped_test, _ = preproc.PCA(poly_mapped_test, dimension)
    prediction = mapped_test @ w
    if logistic:
        prediction = sigmoid(prediction)
    for index in range(0, prediction.shape[0]):
        if prediction[index] < 0.5:
            prediction[index] = 0
        else:
            prediction[index] = 1
    return prediction


def evaluate_least_squares(train_x, train_y, test_x, test_y, X_test):
    """
    Evaluates how well least squares works with various parameters
    """
    parameter_ranges = []
    poly_degree = [8, 14, 17]
    dimension = [1149, 1280, 1310]
    best_param, best_loss, best_degree, best_dimension, best_w = tune_model(
        least_squares,
        parameter_ranges,
        poly_degree,
        dimension,
        train_x,
        train_y,
        test_x,
        test_y,
        accuracy_loss,
    )
    accuracy = 1 - best_loss[0]
    f_1_score = best_loss[2]
    print(
        f"Optimum found, using degree {best_degree} and dimension {best_dimension} achieved accuracy of {accuracy} and F1 score of {f_1_score}"
    )
    prediction = predict(
        X_test, ids_test, best_w, train_x, best_dimension, best_degree, False
    )
    return prediction


def pred_each_jet_set(X_train, y_train, X_test, model_evaluate=evaluate_least_squares):
    """
    Given a single subset of data, trains the model and generates the predictions from that.
    """

    X_train, X_val, y_train, y_val = split_data(X_train, y_train, 0.8, 123)

    pred = model_evaluate(X_train, y_train, X_val, y_val, X_test)
    return pred


def pred_all_sets(
    X_train_list,
    y_train_list,
    X_test_list,
    id_list,
    model_to_evaluate=evaluate_least_squares,
    output_name="prediction.csv",
):
    """
    Arguments:
    X_train_list: A list of our X_train sets
    y_train_list: A list of our output values for the train sets
    X_test_list: A list of outputs to predict using our trained model
    id_list: A list of the output IDs for the tests.
    model_to_evaluate: One of the evaluate_* functions to use to evaluate the model
    output_name: Name of output file to write
    """
    labels = []
    for (X_train, y_train, X_test) in zip(X_train_list, y_train_list, X_test_list):
        labels.append(pred_each_jet_set(X_train, y_train, X_test, model_to_evaluate))
    label_all = np.hstack(labels)
    ids_test = np.hstack(id_list)
    create_csv_submission(ids_test, label_all, output_name)


pred_all_sets(
    [X_train_zero1, X_train_one1, X_train_other1],
    [y_train_zero, y_train_one, y_train_other],
    [X_test_zero1, X_test_one1, X_test_other1],
    [ids_test_zero, ids_test_one, ids_test_other],
    evaluate_least_squares,
    "predictions_final.csv",
)
