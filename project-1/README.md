# Machine Learning Project 1

Omar Boujdaria, Mohamed Ndoye, Vincent Rinaldi

Run the command below to start the predictions
    `python run.py`

Make sure you have the data (to be downloaded from Kaggle) under the right path `Data/train.csv`,`Data/test.csv`.



## Description

### `run.py`

This script generates the model that output the best prediction score submitted on Kaggle after :

- Preprocessing the data
- Generating the model
- Generating and writing predictions


### `implementation.py`

This script contains the required machine learning algorithms for this project:
- Least Squares
- Gradient Descent
- Stochastic Gradient Descent
- Ridge Regression
- Logistic Regression
- Regularized Logistic Regression

As well as some helper functions for the machine learning algorithms :
- compute_mse
- compute_gradient
- batch_iter
- sigmoid
- calculate_loss_log
- calculate_gradient_log
- calculate_hessian_log


### `preprocessing.py`

Contains all the functions to clean, split, standardize and predict the missing values in the datasets
- train_pred_missing_col
- standardize
- preprocess


### `cross_validation.py`

Contains the cross_validation function to perform cross validation on the training set in order to find the best hyper-parameters and compare models.
- build_poly_vect
- build_poly_mtrx
- build_trig_mtrx
- buid_k_indices
- cross_validation


### `proj1_helpers.py`

Contains helpers functions to load the dataset, predict labels and create submission as csv file.
- load_csv_data
- predict_labels
- create_csv_submission
