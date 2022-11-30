import numpy as np


def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
    Batch iteration function used for Stochastic Gradient Descent.
    Input:
        y (labels)
        tx (feature matrix)
        batch_size (scale of a batch)
        shuffle (Shuffle the feature matrix and labels if True)
    """
    data_size = len(y)

    if shuffle:
        indices = np.random.permutation(np.arange(data_size))
        tx_shuffled = tx[indices]
        y_shuffled = y[indices]

    else:
        tx_shuffled = tx
        y_shuffled = y

    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min(data_size, (batch_num + 1) * batch_size)
        if start_index != end_index:
            yield tx_shuffled[start_index:end_index], y_shuffled[start_index:end_index]


def build_poly(x, degree):
    """
    Polynomial expansion of preprocessed feature matrix, up to (degree - 1).
    Input:
        x (preprocessed feature matrix)
        degree (the upper limit of polynomial expansion, upper limit not included)
    """
    poly = np.ones((len(x), 1))
    for degrees in range(1, degree):
        poly = np.c_[poly, np.power(x, degrees)]

    return poly


def compute_gradient(y, tx, w):
    """
    Compute the gradient of loss function.
    Input:
        y (labels)
        tx (feature matrix)
        w (weight)
    """
    e = y - tx.dot(w)
    grad = -1.0 / y.shape[0] * tx.T.dot(e)
    return grad, e


def sigmoid(t):
    """
    Sigmoid function used in logistic regression, np.exp overflow prevented, returns sigmoid output
    Input:
        t (list of variables in sigmoid function)
    """
    """
    output = []
    for i in range(len(t)):
        if t[i] >= 0:
            output.append(1.0 / (1.0 + np.exp(-t)))
        else:
            output.append(np.exp(t[i]) / (1 + np.exp(t[i])))
    output = np.asarray(output).reshape((-1, 1))
    return output
    """
    t = np.clip(t, -20, 20)
    return 1 / (1 + np.exp(-t))


def calculate_gradient_logistic(y, tx, w):
    """
    Calculate gradient of loss function in logistic regression
    Input:
        y (labels relabeled as 0 and 1)
        tx (feature matrix)
        w (weights)
    """
    # sigmoid_pred = sigmoid(tx.dot(w))
    # grad = tx.T.dot(sigmoid_pred - y)
    # return grad
    return (1.0 / y.shape[0]) * (tx.T @ (sigmoid(tx @ w) - y))


def learning_by_SGD_logistic(y, tx, w, gamma, batch_size=1):
    """
    Logistic regression by stochastic gradient descent, return weight and loss
    Input:
        y (labels relabeled as 0 and 1)
        tx (feature matrix)
        w (weights)
        gamma (Gamma parameter)
        batch_size (number of samples in batch)
    """
    for tx_batch, y_batch in batch_iter(y, tx, batch_size, num_batches=1):
        # loss = calculate_loss_logistic(y_batch, tx_batch, w)
        grad = calculate_gradient_logistic(y_batch, tx_batch, w)
        w = w - gamma * grad
    loss = calculate_loss_logistic(y, tx, w)
    return w, loss


def penalized_logistic_regression(y, tx, w, lambda_):
    """
    Calculate loss, gradient, hessian for regularized logistic regression
    Input:
        y (labels relabeled as 0 and 1)
        tx (feature matrix)
        w (weights)
        lambda_ (Lambda parameter)
    """
    # return loss, gradient, and hessian
    loss = np.squeeze(calculate_loss_logistic(y, tx, w) + lambda_ * (w.T.dot(w)))
    grad = calculate_gradient_logistic(y, tx, w) + 2 * lambda_ * w

    return float(loss), grad, None


def learning_by_penalized_logistic(y, tx, w, gamma, lambda_, batch_size=1):
    """
    Regularized logistic regression by stochastic gradient descent, return weight and loss
    Input:
        y (labels relabeled as 0 and 1)
        tx (feature matrix)
        w (weights)
        gamma (Gamma parameter)
        lambda_ (Lambda parameter)
        batch_size (number of samples in batch)
    """
    for tx_batch, y_batch in batch_iter(y, tx, batch_size, num_batches=1):
        loss, grad, _ = penalized_logistic_regression(y_batch, tx_batch, w, lambda_)
        w = w - gamma * grad

    return w, loss


def compute_mse(e):
    return 1 / 2 * np.mean(e**2)


def compute_loss_mse(y, tx, w):
    """
    Compute MSE loss using the error term, return the mse loss.
    Input:
        y (labels of the training data)
        tx (feature matrix of the training data)
        w (weight)
    """
    e = y - tx.dot(w)
    loss = compute_mse(e)
    return loss


def calculate_loss_logistic(y, tx, w):
    """
    Compute loss of logistic regression, return the loss.
    Input:
        y (labels of the training data)
        tx (feature matrix of the training data)
        w (weight)
    """
    # sigmoid_pred = sigmoid(tx.dot(w))
    # loss = -(y.T.dot(np.log(sigmoid_pred)) + (1 - y).T.dot(np.log(1 - sigmoid_pred)))
    # loss = np.squeeze(loss)
    probs = sigmoid(tx @ w)
    return -(1 / len(y)) * np.sum(y * np.log(probs) + (1 - y) * np.log(1 - probs))


"""
    Six Basic Function
"""


def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    """
    Train using mean squared error gradient descent
    Input:
        y: label of training data
        tx: training data feature
        initial_w: Initial weights
        max_iters: max iteration number
        gamma: the gamma parameter
    Return:
        the weights and loss of the last iteration
    """
    losses = []
    w = initial_w
    ws = [initial_w]
    for _ in range(max_iters):
        grad, _ = compute_gradient(y, tx, w)
        loss = compute_loss_mse(y, tx, w)
        w = w - grad * gamma
        ws.append(w)
        losses.append(loss)
    loss = compute_loss_mse(y, tx, w)
    losses.append(loss)
    return ws[-1], losses[-1]


def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma, batch_size=1):
    """
    Train using mean squared error SGD
    Input:
        y: label of training data
        tx: training data feature
        initial_w: Initial weights
        max_iters: max iteration number
        gamma: the gamma parameter
        batch_size: here use default batch size 1 for SGD
    Return:
        the weights and loss of the last iteration
    """

    losses = []
    w = initial_w
    ws = [initial_w]

    for _ in range(max_iters):
        for tx_batch, y_batch in batch_iter(y, tx, batch_size, num_batches=1):
            grad, _ = compute_gradient(y_batch, tx_batch, w)
            loss = compute_loss_mse(y_batch, tx_batch, w)
            w = w - grad * gamma
            ws.append(w)
            losses.append(loss)
    loss = compute_loss_mse(y, tx, w)
    losses.append(loss)

    return ws[-1], losses[-1]


def least_squares(y, tx):
    """
    Train using least_squares
    Input:
        y: label of training data
        tx: training data feature
    Return:
        the weights and loss
    """
    w = np.linalg.lstsq(tx, y, rcond=-1)[0]
    loss = compute_loss_mse(y, tx, w)
    return w, loss


def ridge_regression(y, tx, lambda_):
    """
    Train using ridge regression
    Input:
        y: label of training data
        tx: training data feature
        lambda_: the lambda hyperparameters
    Return:
        the weights and loss
    """
    aI = 2 * tx.shape[0] * lambda_ * np.identity(tx.shape[1])
    a = tx.T.dot(tx) + aI
    b = tx.T.dot(y)
    w = np.linalg.solve(a, b)
    loss = compute_loss_mse(y, tx, w)
    # if calculate root mse as loss
    # loss = np.sqrt(2 * compute_loss_mse(y, tx, w))

    return w, loss


def logistic_regression(y, tx, initial_w, max_iters, gamma, threshold=1e-8):
    """
    Train using logistic_regression
    Input:
        y: label of training data
        tx: training data feature
        initial_w: Initial weights
        max_iters: max iteration number
        gamma: the gamma parameter
        threshold: the threshold parameter
    Return:
        the weights and loss of the last iteration
    """

    losses = []
    # tx = np.c_[np.ones((tx.shape[0], 1)), tx]

    w = initial_w
    ws = [initial_w]

    for _ in range(max_iters):
        w, loss = learning_by_SGD_logistic(y, tx, w, gamma, batch_size=y.shape[0])
        losses.append(loss)
        ws.append(w)

    losses = calculate_loss_logistic(y, tx, ws[-1])

    return ws[-1], losses


def reg_logistic_regression(
    y, tx, lambda_, initial_w, max_iters, gamma, threshold=1e-8
):
    """
    Train using reg_logistic_regression
    Input:
        y: label of training data
        tx: training data feature
        lambda_: the lambda parameter
        initial_w: Initial weights
        max_iters: max iteration number
        gamma: the gamma parameter
        threshold: the threshold parameter
    Return:
        the weights and loss of the last iteration
    """

    losses = []
    w = initial_w
    ws = [initial_w]

    for _ in range(max_iters):
        w, loss = learning_by_penalized_logistic(
            y, tx, w, gamma, lambda_, batch_size=y.shape[0]
        )
        losses.append(loss)
        ws.append(w)

    loss = np.squeeze(calculate_loss_logistic(y, tx, w))

    return ws[-1], loss


def split_data(x, y, ratio, seed=1):
    """
    split the dataset based on the split ratio. If ratio is 0.8
    you will have 80% of your data set dedicated to training
    and the rest dedicated to testing. If ratio times the number of samples is not round
    you can use np.floor. Also check the documentation for np.random.permutation,
    it could be useful.

    Args:
        x: numpy array of shape (N,), N is the number of samples.
        y: numpy array of shape (N,).
        ratio: scalar in [0,1]
        seed: integer.

    Returns:
        x_tr: numpy array containing the train data.
        x_te: numpy array containing the test data.
        y_tr: numpy array containing the train labels.
        y_te: numpy array containing the test labels.

    >>> split_data(np.arange(13), np.arange(13), 0.8, 1)
    (array([ 2,  3,  4, 10,  1,  6,  0,  7, 12,  9]), array([ 8, 11,  5]), array([ 2,  3,  4, 10,  1,  6,  0,  7, 12,  9]), array([ 8, 11,  5]))
    """
    # set seed
    np.random.seed(seed)

    indexes = np.arange(x.shape[0])
    np.random.shuffle(indexes)
    cutoff = int(np.floor(ratio * x.shape[0]))
    train_indexes = indexes[:cutoff]
    test_indexes = indexes[cutoff:]
    return x[train_indexes], x[test_indexes], y[train_indexes], y[test_indexes]
