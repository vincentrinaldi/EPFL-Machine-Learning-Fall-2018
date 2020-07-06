# Machine Learning Project 2
# Recommender Systems

Omar Boujdaria, omar.boujdaria@epfl.ch
Mohamed Ndoye, mohamed.ndoye@epfl.ch
Vincent Rinaldi, vincent.rinaldi@epfl.ch

CrowdAI username: Vincent Rinaldi
Submission ID: 23513

External packages to install: numpy, pandas, pyspark, csv, math, time, scipy, matplotlib.

Run the command below to start the predictions
    `python run.py`

Make sure you have the data (to be downloaded from CrowdAI) under the right path `Data/data_train.csv`,`Data/sample_submission.csv`.

Competition link : https://www.crowdai.org/challenges/epfl-ml-recommender-system

## Description

### `run.py`

This script generates the model that output the best prediction score submitted on CrowdAI.

### `baseline_model.py`

This script reads the data, implement the baseline model and create the submission csv file.

### `als_baseline_model.py`

This script reads the data, trains the model and predicts over a part of the sample submission using Alternating Least Squares. The other part is predicted with the baseline model. Finally, the predictions are combined and the csv file is created.

### `svd_regularization.py`

Run and generate data prediction using SVD Regularization.

### `matrix_factorization.py`

This script implements the Matrix-Factorization with Stochastic Gradient Descent as it was presented in lab 10.

### `cross_val_features.py`

Cross-validate the number-of-features k, where k iterates in [10 20 30 40 50].


### `cross_val_lrate.py`

Cross-validate the learning rate l, where l iterates in [.001 .002 .003 .004 .005].


### `cross_val_regularizer.py`

Cross-validate the regularizer term lambda, where lambda iterates in [.001 .005 .01 .015 .02].


### `helpers.py`

Contains helper functions for baseline model

- mean_
- u_id_biases_
- i_id_biases_
- predict


### `svd_regularization_helper.py`

Contains helper functions for SVD Regularization Method

- init_train
- init_full
- init_UV
- partial_UV_pred_rating
- full_UV_pred_rating
- calc_ratings_rmse


### `matrix_factorization_helper.py`

Contains helper functions for Matrix Factorization Method

- read_preprocess_data
- get_data
- preprocess_elem
- compute_error
- create_csv_submission
