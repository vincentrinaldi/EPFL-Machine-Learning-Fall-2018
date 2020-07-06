import numpy as np
from implementations import *



"""Helper Functions For Data Preprocessing"""



"""Predict Missing Values"""

def train_pred_missing_col(data_clean, data_full, train, col_to_predict):
    """Predict each nan value of the data set using ridge regression"""
    # Separate the column having the nan values we want to predict from the other columns of the data set
    y = train[:,col_to_predict]
    x = np.delete(train,col_to_predict,axis=1)

    # Calculate the weight w using ridge regression
    w, _ = ridge_regression(y, x, lambda_=0.001)

    # Compute the new values of the column we are analyzing by multiplying the temporary new features data set with w
    y_pred = np.dot(np.delete(data_clean, col_to_predict,axis=1), w)
    
    # Get back the original values of the column and replace only the nan values by their corresponding computed value
    y_pred = np.asarray([y if  np.isnan(d) else d for (y,d) in zip(y_pred,data_full[:,col_to_predict])])
    data_clean[:,col_to_predict] = y_pred

    return data_clean


"""Standardize Data"""

def standardize(x):
    """Standardize the original data set."""
    mean_x = np.mean(x, axis=0)
    x = x - mean_x
    std_x = np.std(x, axis=0)
    x = x / std_x
    return x, mean_x, std_x


"""Preprocess Data"""

def preprocess(input_data_tr, input_data_te, nan_val = -999.):
    """Preprocess the original data set and perform data cleaning before performing regression."""
    # Replace erroneous values by nan ones
    input_data_tr[input_data_tr == nan_val] = np.nan
    input_data_te[input_data_te == nan_val] = np.nan

    # Find columns containing nans and set them to true whereas the others are set to false
    NanCol_tr = np.isnan(input_data_tr).sum(axis=0).astype(bool)
    NanCol_te = np.isnan(input_data_te).sum(axis=0).astype(bool)
    NanCols = np.logical_or(NanCol_te, NanCol_tr)

    # Stack arrays in sequence vertically to obtain a full dataset
    data_f = np.vstack((input_data_tr, input_data_te))

    # Clean columns containing at least a nan by filling them with zeros
    data_c = np.copy(data_f)
    for i,b in enumerate(NanCols):
        if b:
            data_c[:,i] = np.zeros(data_c.shape[0])

    # Predict and replace nan values from full dataset columns
    for i,b in enumerate(NanCols):
        if b:
            # Put back original values of current analyzed column containing nan values
            data_c[:,i] = data_f[:,i]
            # Remove rows that have a nan value for this column to form a training set
            train = data_c[~np.isnan(data_c).any(axis=1)]
            # Predict the nan values of the column we are operating on by training on our training set
            data_c = train_pred_missing_col(data_c, data_f, train, i)

    # Standardize the data set
    data_c = standardize(data_c)[0]
    
    # Split the preprocessed data into approximately 30:70 for training and testing sets
    input_data_tr = data_c[:250000]
    input_data_te = data_c[250000:]

    return input_data_tr, input_data_te
