# -*- coding: utf-8 -*-
import numpy as np



"""Helper Functions For Machine Learning Methods"""


"""Compute Mean Square Error"""

def compute_mse(y, tx, w):
    """Calculate the loss."""
    e = y - tx @ w
    return e.T @ e / (2 * len(e))


"""Compute Gradient"""

def compute_gradient(y, tx, w):
    """Compute the gradient."""
    e = y - tx @ w
    return (-1 / len(e)) * tx.T @ e


"""Generate Minibatch Iterator"""

def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """Generate a minibatch iterator for a dataset."""
    data_size = len(y)
    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]

            
"""Apply Sigmoid"""

def sigmoid(t):
    """Apply sigmoid function on t."""
    return 1.0 / (1 + np.exp(-t))


"""Compute Loss For Logistic Regression"""

def calculate_loss_log(y, tx, w):
    """Compute the cost by negative log likelihood."""
    pred = sigmoid(tx @ w)
    loss = y.T @ (np.log(pred)) + (1 - y).T @ (np.log(1 - pred))
    return np.squeeze(- loss)


"""Compute Gradient For Logistic Regression"""

def calculate_gradient_log(y, tx, w):
    """Compute the gradient of loss."""
    pred = sigmoid(tx @ w)
    grad = tx.T @ (pred - y)
    return grad


"""Compute Hessian For Logistic Regression"""

def calculate_hessian_log(y, tx, w):
    """Return the hessian of the loss function."""
    pred = sigmoid(tx @ w)
    pred = np.diag(pred.T[0])
    s = np.multiply(pred, (1-pred))
    return tx.T @ s @ tx


"""Compute Penalized Loss, Gradient And Hessian"""

def penalized_logistic_regression(y, tx, w, lambda_):
    """Return the loss, gradient and hessian with a penalty term."""
    loss = calculate_loss_log(y, tx, w) + (lambda_ / 2) * np.squeeze(w.T @ w)
    grad = calculate_gradient_log(y, tx, w) + lambda_ * w
    hess = calculate_hessian_log(y, tx, w) + (lambda_ / 2) * np.diag(w.T[0])
    return loss, grad, hess



"""Machine Learning Methods"""


"""Gradient Descent"""

def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """Gradient descent algorithm."""
    w = initial_w
    for n_iter in range(max_iters):
        grad = compute_gradient(y, tx, w)
        w = w - gamma * grad
    loss = compute_mse(y, tx, w)
    return w, loss


"""Stochastic Gradient Descent"""

def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    """Stochastic gradient descent algorithm."""
    w = initial_w
    for n_iter in range(max_iters):
        for minibatch_y, minibatch_tx in batch_iter(y, tx, 1):
            grad = compute_gradient(minibatch_y, minibatch_tx, w)
            w = w - gamma * grad
    loss = compute_mse(y, tx, w)
    return w, loss


"""Least Square"""

def least_squares(y, tx):
    """Calculate the least squares solution."""
    w = np.linalg.solve(tx.T @ tx, tx.T @ y)
    loss = compute_mse(y, tx, w)
    return w, loss


"""Ridge Regression"""

def ridge_regression(y, tx, lambda_):
    """Implement ridge regression."""
    lambda_prime = lambda_ * 2 * len(y)
    txSquare = tx.T @ tx
    a = txSquare + lambda_prime * np.identity(txSquare.shape[0])
    b = tx.T @ y
    w = np.linalg.solve(a, b)
    loss = compute_mse(y, tx, w)
    return w, loss


"""Logistic Regression"""

def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """Implement logistic regression."""
    w = initial_w
    for iter in range(max_iter):
        grad = calculate_gradient_log(y, tx, w)
        w = w - gamma * grad
    loss = calculate_loss_log(y, tx, w)
    return w, loss


"""Regularized Logistic Regression"""

def reg_logistic_regression(y, tx, lambda_ , initial_w, max_iters, gamma):
    """Implement regularized logistic regression using the penalized logistic regression."""
    w = initial_w
    for iter in range(max_iter):
        _, grad, _ = penalized_logistic_regression(y, tx, w, lambda_)
        w = w - gamma * grad
    loss, _, _ = penalized_logistic_regression(y, tx, w, lambda_)
    return w, loss
