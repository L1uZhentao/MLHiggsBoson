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
    poly_degree = [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
    dimension = [
        50,
        75,
        100,
        115,
        120,
        125,
        130,
        135,
        175,
        200,
        250,
        300,
        305,
        310,
        315,
        320,
        325,
        330,
        335,
        340,
        345,
        350,
    ]
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
    [X_train_zero, X_train_one, X_train_other],
    [y_train_zero, y_train_one, y_train_other],
    [X_test_zero, X_test_one, X_test_other],
    [ids_test_zero, ids_test_one, ids_test_other],
    evaluate_least_squares,
    "predictions_no_cross_term.csv",
)
