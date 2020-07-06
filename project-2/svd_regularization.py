import pandas as pd
import numpy as np

from svd_regularization_helper import *

import csv
import math
import time



"""Run And Generate Data Prediction Using SVD Regularization"""


def svd_regularization():
    
    # Hyperparameters
    k = 50             # Size of each feature vectors
    lrate = 0.035      # Learning rate
    regularizer = 0.01 # Regularization parameter

    max_epochs = 50    # Maximum number of iterations for feature training
    min_improv = 0.001 # Stop the feature training prematurely if improvement in RMSE is smaller than this value

    start = time.time()

    # Get train dataset and the one to predict
    dftrain = init_train()
    dffull, nb_users_full, nb_items_full = init_full()

    # Initialize feature matrix U and V
    train_ratings = dftrain.values
    U, V = init_UV(nb_users_full, nb_items_full, train_ratings, dffull, k)
    print ("Time after feature matrix initilization :", time.time() - start)

    # Train matrix U and V
    U, V = full_UV_pred_rating(U, V, train_ratings, k, lrate, regularizer, max_epochs, min_improv)
    print ("Time after training matrix U and V :", time.time() - start)

    # Calculate RMSE between the actual train set ratings and its predicted ones
    _, rmse = calc_ratings_rmse(U, V, train_ratings)
    print("RMSE with train set =", rmse, "for : k =", k, "; lrate =", lrate, "; regularizer =", regularizer)

    # Retrieve the predicted ratings for the set to predict
    dfsub = dffull.copy()
    final_ratings = dffull.values
    pred_ratings, _ = calc_ratings_rmse(U, V, final_ratings)
    dfsub.loc[:,'Rating'] = pred_ratings

    # Reorganize dataframe to the correct output format
    dfsub['Id'] = dfsub[['User', 'Item']].apply(lambda x: '_'.join(x), axis=1)
    dfsub['Prediction'] = dfsub['Rating']
    dfsub.drop('User', axis=1, inplace=True)
    dfsub.drop('Item', axis=1, inplace=True)
    dfsub.drop('Rating', axis=1, inplace=True)
    dfsub = dfsub[['Id', 'Prediction']]

    # Create the output file in csv format containing the predicted ratings for the given set
    user_movie_pairs = dfsub['Id'].tolist()
    computed_predictions = dfsub['Prediction'].tolist()
    with open('Regularized_SVD_Predictions.csv', 'w') as csvfile:
        fieldnames = ['Id', 'Prediction']
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames, lineterminator = '\n')
        writer.writeheader()
        for r1, r2 in zip(user_movie_pairs, computed_predictions):
            writer.writerow({'Id':r1,'Prediction':np.clip(round(r2), 1, 5)})		
    print ("Time after finishing rating predictions :", time.time() - start)
