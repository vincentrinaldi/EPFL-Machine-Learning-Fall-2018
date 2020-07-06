import numpy as np
import scipy.sparse as sp

from matrix_factorization_helper import *

import time
import math
import csv



"""Run And Generate Data Prediction Using Matrix Factorization"""


def matrix_factorization():

    # Get data

    datam = get_data(r'Data')

    num_users = 10000
    num_items = 1000
    num_features = 10


    # Initialize latent vectors

    user_features = np.random.rand(num_users, num_features)
    item_features = np.random.rand(num_items, num_features)

    item_sum = datam.sum(axis=0)    #sum of ratings to a movie for each movie
    item_nnz = datam.getnnz(axis=0) #nb of ratings received by a movie for each movie

    user_sum = datam.sum(axis=1)    #sum of ratings by a user for each user
    user_nnz = datam.getnnz(axis=1) #nb of movies rated by a user for each user

    for i in range(num_items):
        if item_nnz[i] == 0:
            continue
        else:
            item_features[i, 0] = item_sum[0, i]/item_nnz[i]
    
    for i in range(num_users):
        if user_nnz[i] == 0:
            continue
        else:
            user_features[i, 0] = user_sum[i, 0]/user_nnz[i]

    a,b = datam.nonzero()
    nz_data = list(zip(a,b))


    # Start SGD computation

    start = time.time()

    max_epochs = 20
    lrate = 0.002
    regularizer = 0.02

    for it in range(max_epochs):
        for d, n in nz_data:
            item_info = item_features[n, :]
            user_info = user_features[d, :]
            err = datam[d, n] - user_info.T.dot(item_info)

            item_features[n, :] += lrate * (err * user_info - regularizer * item_info)
            user_features[d, :] += lrate * (err * item_info - regularizer * user_info)

        rmse = compute_error(datam, user_features, item_features, nz_data)
        print("iter: {}, RMSE on data set: {}.".format(it+1, rmse))

    print ("time after full UV computation :", time.time() - start)


    # Output the submission file with the predicted ratings

    final_predictions = sp.lil_matrix((num_users, num_items))
    for user in range(num_users):
        for item in range(num_items):
            user_info = user_features[user,:]
            item_info = item_features[item, :]
            final_predictions[user, item] = user_info.T.dot(item_info)
        
    create_csv_submission(final_predictions)
