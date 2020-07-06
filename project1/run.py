import numpy as np
from implementations import *
from preprocessing import *
from cross_validation import *
from proj1_helpers import *



"""Main Function To Run And Generate Data Prediction"""



def main():
    # Load the data
    yb_tr, input_data_tr, ids_tr = load_csv_data("Data/train.csv")
    yb_te, input_data_te, ids_te = load_csv_data("Data/test.csv")

    # Perform preprocessing and data cleaning
    input_data_tr, input_data_te = preprocess(input_data_tr, input_data_te)


    """
    # Cross-validation steps

    # We define the parameters
    seed = 1
    k_fold = 4
    max_deg = 11
    degrees = np.linspace(2,max_deg,max_deg-1, dtype=int)
    lambdas = np.logspace(-9,0,10)

    # We split data in k parts
    k_indices = build_k_indices(yb_tr, k_fold, seed)

    # We perform cross-validation to tune lambda and degree parameters
    rmse_te = []
    for i, lambda_ in enumerate(lambdas):
        rmse_te_l = []
        for j,deg in enumerate(degrees):
            rmse_te_d = []
            for k in range(k_fold):
                print('ijk', i, j, k)
                loss_te, w = cross_validation(yb_tr, input_data_tr, k_indices, k, lambda_, deg)
                rmse_te_d.append(np.sqrt(2*loss_te))
            rmse_te_l.append(np.mean(rmse_te_d))
        rmse_te.append(rmse_te_l)
    rmse_te_array = np.asarray(rmse_te)

    # We retrieve the best lambda and degree found
    ind = np.unravel_index(np.argmin(rmse_te_array, axis=None), rmse_te_array.shape)
    best_lambda_ = lambdas[ind[0]]
    best_degree = degrees[ind[1]]

    # We expand the columns
    poly_test  = build_poly_mtrx(input_data_te, best_degree)
    poly_train = build_poly_mtrx(input_data_tr, best_degree)
    poly_test  = build_trig_mtrx(poly_test)
    poly_train = build_trig_mtrx(poly_train)

    # ridge regression
    w_ridge = ridge_regression(yb_tr, poly_train, best_lambda_)[0]
    """


    # Form data with polynomial degree and trigonometric functions
    poly_test  = build_poly_mtrx(input_data_te, degree=10)
    poly_train = build_poly_mtrx(input_data_tr, degree=10)
    poly_test  = build_trig_mtrx(poly_test)
    poly_train = build_trig_mtrx(poly_train)

    # Compute weight w using ridge regression
    w_ridge, loss_tr = ridge_regression(yb_tr, poly_train, lambda_=0.00000001)

    # Generate and output predictions in .csv format
    y_pred = predict_labels(w_ridge, poly_test)
    create_csv_submission(ids_te, y_pred, "Predictions_TeamMOV.csv")



if __name__ == '__main__':
    main()
