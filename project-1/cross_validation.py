import numpy as np
from implementations import *



"""Helper Functions For Cross-Validation And Predictions Computation"""



"""Implement Polynomial Basis Function For Vector"""

def build_poly_vect(x, degree):
    """Polynomial basis functions for input vector x, for j=0 up to j=degree."""
    return np.vander(x, degree + 1, True)


"""Implement Polynomial Basis Function For Matrix"""

def build_poly_mtrx(tx, degree):
    """Polynomial functions for input matrix tx, for j=0 up to j=degree."""
    t_input_data = build_poly_vect(tx[:,0], degree)

    for i in range(1,tx.shape[1]):
        t_input_data = np.hstack((t_input_data, build_poly_vect(tx[:,i], degree)[:,1:]))

    return t_input_data


"""Implement Trigonometric Basis Function For Matrix"""

def build_trig_mtrx(poly_data):
    """Trigonometric functions for input matrix poly_data, applying cosinus, sinus and tangent function."""
    nb_col = poly_data.shape[1]

    cos = np.cos(poly_data[:,1:nb_col])
    sin = np.sin(poly_data[:,1:nb_col])
    tan = np.tan(poly_data[:,1:nb_col])

    return np.hstack((poly_data,cos,sin,tan))


"""Indices Set Building"""

def build_k_indices(y, k_fold, seed):
    """Build k indices for k-fold."""
    num_row = y.shape[0]
    interval = int(num_row / k_fold)

    np.random.seed(seed)
    indices = np.random.permutation(num_row)

    k_indices = [indices[k * interval: (k + 1) * interval]
                 for k in range(k_fold)]

    return np.array(k_indices)


"""Cross-Validation"""

def cross_validation(y, x, k_indices, k, lambda_, degree):
    """Test error estimation by local cross-validation using ridge regression."""
    # Get the k'th subgroup of features and labels as the test set
    test_xs = np.array([np.take(x[:,i], k_indices[k]) for i in range(x.shape[1])]).T
    test_y  = np.take(y, k_indices[k])

    # Remove indice k from the indices list
    new_k_indices = np.delete(k_indices, k, axis=0)
    new_k_indices = np.ravel(new_k_indices)

    # Get the remaining k-1 subgroup of features and labels as the train set
    train_xs = np.array([np.take(x[:,i], new_k_indices) for i in range(x.shape[1])]).T
    train_y  = np.take(y, new_k_indices)

    # Form data with polynomial degree
    poly_test  = build_poly_mtrx(test_xs, degree)
    poly_train = build_poly_mtrx(train_xs, degree)

    # Calculate weight vector w using ridge regression:
    w_ridge, loss_tr = ridge_regression(train_y, poly_train, lambda_)

    # Calculate the loss for the test set
    loss_te = compute_mse(test_y, poly_test, w_ridge)

    return loss_te, w_ridge
