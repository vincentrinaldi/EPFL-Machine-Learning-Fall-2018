import pandas as pd
import numpy as np

import csv
import math
import time



"""Helper Functions For SVD Regularization Method"""


"""
Get the training set of user-movie pairs

"""

def init_train():
    # Retrieve the set from the csv file and transform it into a dataframe
    dftrain = pd.read_csv('Data/data_train.csv')

    # Modify the dataframe organisation
	
    dftrain['User'] = dftrain['Id'].apply(lambda x : x.split('_')[0])
    dftrain['Item'] = dftrain['Id'].apply(lambda x : x.split('_')[1])
    dftrain['Rating'] = dftrain['Prediction']

    dftrain.drop('Id', axis=1, inplace=True)
    dftrain.drop('Prediction', axis=1, inplace=True)
    dftrain = dftrain[['User', 'Item', 'Rating']]

    # Get the number of different existing users and movies from this set
    nb_ratings_train = len(dftrain)
    nb_users_train = len(set(dftrain['User']))
    nb_items_train = len(set(dftrain['Item']))
    print("There are", nb_ratings_train, "ratings in total with", nb_users_train, "users and", nb_items_train, "items in the train set.")

    return dftrain


"""
Get the set of user-movie pairs for which we want to predict their ratings

"""

def init_full():
    # Retrieve the set from the csv file and transform it into a dataframe
    dffull = pd.read_csv('Data/sample_submission.csv')

    # Modify the dataframe organisation
	
    dffull['User'] = dffull['Id'].apply(lambda x : x.split('_')[0])
    dffull['Item'] = dffull['Id'].apply(lambda x : x.split('_')[1])
    dffull['Rating'] = dffull['Prediction']

    dffull.drop('Id', axis=1, inplace=True)
    dffull.drop('Prediction', axis=1, inplace=True)
    dffull = dffull[['User', 'Item', 'Rating']]

    # Get the number of different existing users and movies from this set
    nb_ratings_full = len(dffull)
    nb_users_full = len(set(dffull['User']))
    nb_items_full = len(set(dffull['Item']))
    print("There are", nb_ratings_full, "ratings in total with", nb_users_full, "users and", nb_items_full, "items among the ratings to predict.")

    return dffull, nb_users_full, nb_items_full


"""
Create and initialize feature matrix U and V using Panda Dataframes

nb_users_full : Number of different users in the dataset to predict
nb_items_full : Number of different movies in the dataset to predict
train_ratings : Numpy array containing every 3-tuples user-movie-rating of the train set
dffull        : Dataframe of the dataset to predict
k             : Number of features
"""

def init_UV(nb_users_full, nb_items_full, train_ratings, dffull, k):
    # Get average rating from train set
    avg_ratings = float(np.mean(train_ratings[:,2]))
        
    # Set initial values of every matrix U and V elements to square root of average/rank
    initval = math.sqrt(avg_ratings/k)
    
    # Set the lists of factors and values to build the dataframes representing the matrix
	
    arr_factor_user = []
    for user in range (nb_users_full):
        for i in range (k):
            curr_char = "f" + str(i + 1)
            arr_factor_user.append(curr_char)
    
    arr_value_user = []
    for user in range (nb_users_full):
        for i in range (k):
            arr_value_user.append(initval)
            
    arr_factor_item = []
    for item in range (nb_items_full):
        for i in range (k):
            curr_char = "f" + str(i + 1)
            arr_factor_item.append(curr_char)
    
    arr_value_item = []
    for item in range (nb_items_full):
        for i in range (k):
            arr_value_item.append(initval)
    
    # Initialize and orderer matrix U
    
    dfmatrixU = dffull.copy()
    dfmatrixU.drop('Item', axis=1, inplace=True)
    dfmatrixU.drop('Rating', axis=1, inplace=True)

    dfmatrixU['Sort'] = dfmatrixU['User'].str.extract('(\d+)', expand=False).astype(int)
    dfmatrixU.sort_values('Sort', inplace=True, ascending=True)
    dfmatrixU = dfmatrixU.drop('Sort', axis=1)
    dfmatrixU = dfmatrixU.drop_duplicates(["User"])

    dfmatrixU = pd.concat([dfmatrixU] * k, ignore_index=True)
    dfmatrixU['Sort'] = dfmatrixU['User'].str.extract('(\d+)', expand=False).astype(int)
    dfmatrixU.sort_values('Sort', inplace=True, ascending=True)
    dfmatrixU = dfmatrixU.drop('Sort', axis=1)

    dfmatrixU = dfmatrixU.assign(Factor = pd.Series(arr_factor_user).values)
    dfmatrixU = dfmatrixU.assign(Value = pd.Series(arr_value_user).values)

    dfmatrixU = dfmatrixU.pivot_table(index='User', columns='Factor', values='Value')
    dfmatrixU = dfmatrixU.reindex(sorted(dfmatrixU.index, key=lambda x: int(x[1:])), axis=0)
    dfmatrixU = dfmatrixU.reindex(sorted(dfmatrixU.columns, key=lambda x: int(x[1:])), axis=1)
    
    # Initialize and orderer matrix V
    
    dfmatrixV = dffull.copy()
    dfmatrixV.drop('User', axis=1, inplace=True)
    dfmatrixV.drop('Rating', axis=1, inplace=True)

    dfmatrixV['Sort'] = dfmatrixV['Item'].str.extract('(\d+)', expand=False).astype(int)
    dfmatrixV.sort_values('Sort', inplace=True, ascending=True)
    dfmatrixV = dfmatrixV.drop('Sort', axis=1)
    dfmatrixV = dfmatrixV.drop_duplicates(["Item"])

    dfmatrixV = pd.concat([dfmatrixV] * k, ignore_index=True)
    dfmatrixV['Sort'] = dfmatrixV['Item'].str.extract('(\d+)', expand=False).astype(int)
    dfmatrixV.sort_values('Sort', inplace=True, ascending=True)
    dfmatrixV = dfmatrixV.drop('Sort', axis=1)
    
    dfmatrixV = dfmatrixV.assign(Factor = pd.Series(arr_factor_item).values)
    dfmatrixV = dfmatrixV.assign(Value = pd.Series(arr_value_item).values)

    dfmatrixV = dfmatrixV.pivot_table(index='Item', columns='Factor', values='Value')
    dfmatrixV = dfmatrixV.reindex(sorted(dfmatrixV.index, key=lambda x: int(x[1:])), axis=0)
    dfmatrixV = dfmatrixV.reindex(sorted(dfmatrixV.columns, key=lambda x: int(x[1:])), axis=1)
    
    return dfmatrixU, dfmatrixV

            
"""
Update the currently feature column of both matrix U and V using Stochastic Gradient Descent technique and return the updated matrix U and V and the computed RMSE

U             : User feature matrix
V             : Movie feature matrix
train_ratings : Numpy array containing every 3-tuples user-movie-rating of the train set
col           : Feature column currently analyzed
lrate         : Learning rate
regularizer   : Regularization parameter
"""

def partial_UV_pred_rating(U, V, train_ratings, col, lrate, regularizer):
    # Set the name of the feature column and initialize the aggregate error value
    curr_char = "f" + str(col + 1)
    agg_err = 0.0
    nb = 0
        
    for i in range(train_ratings.shape[0]):            
        # Perform dot product to compute the estimated rating for the currently analyzed user-movie pair
        product = np.dot(U.loc[(train_ratings[i,0])], V.loc[(train_ratings[i,1])])
        if product > 5:
            product = 5
        elif product < 1:
            product = 1

        # Compute the error between the estimated rating and the actual rating and update the aggregate error value
        err = train_ratings[i,2] - product
        agg_err += err**2
        nb += 1

        # Save the corresponding actual feature value according to the currently analyzed user-movie pair for both matrix U and V
        curr_U = U.loc[(train_ratings[i,0], curr_char)]
        curr_V = V.loc[(train_ratings[i,1], curr_char)]

        # Update the corresponding feature value according to the currently analyzed user-movie pair for both matrix U and V
        U.loc[(train_ratings[i,0], curr_char)] += lrate * (err * curr_V - regularizer * curr_U)
        V.loc[(train_ratings[i,1], curr_char)] += lrate * (err * curr_U - regularizer * curr_V)
        
    return math.sqrt(agg_err/nb), U, V


"""
Train each column of matrix U and V

U             : User feature matrix
V             : Movie feature matrix
train_ratings : Numpy array containing every 3-tuples user-movie-rating of the train set
k             : Number of features
lrate         : Learning rate
regularizer   : Regularization parameter
max_epochs    : Maximum number of iterations for feature training
min_improv    : Stop the feature training prematurely if improvement in RMSE is smaller than this value
"""

def full_UV_pred_rating(U, V, train_ratings, k, lrate, regularizer, max_epochs, min_improv):        
    # Set the initial train error
    old_train_err = 1000000.0
       
    for col in range(k):
        
        print ("For k =", col + 1)
        
        for epoch in range(max_epochs):
            # Train the currently analyzed column
            train_err, U, V = partial_UV_pred_rating(U, V, train_ratings, col, lrate, regularizer)
            print ("For epoch =", epoch + 1, ": train error =", train_err)
            
            # If train error improvement is too small we immediately start computing the next column of matrix U and V
            if abs(old_train_err - train_err) < min_improv:
                break
            
            old_train_err = train_err
    
    return U, V


"""
Calculate the predicted ratings of the user-movie pairs from the given set and compute the resulting RMSE

U           : User feature matrix
V           : Movie feature matrix
set_ratings : Numpy array containing every 3-tuples user-movie-rating of a dataset
"""

def calc_ratings_rmse(U, V, set_ratings):    
    # Get matrix U and V with their indices
    npU = U.reset_index().values
    npV = V.reset_index().values

    # Retrieve the array containing the user ids in the same order as in the given set
    idx_set_users = set_ratings[:,0].copy()
    idx_set_users[:] = [s[1:] for s in idx_set_users]
    idx_set_users = idx_set_users.astype(int)
    idx_set_users = np.asarray(idx_set_users - 1)
    users_factors = npU[idx_set_users,1:]
	
    # Retrieve the array containing the movie ids in the same order as in the given set
    idx_set_items = set_ratings[:,1].copy()
    idx_set_items[:] = [s[1:] for s in idx_set_items]
    idx_set_items = idx_set_items.astype(int)
    idx_set_items = np.asarray(idx_set_items - 1)
    items_factors = npV[idx_set_items,1:]

    # Calculate the rating predictions using a dot product between the two corresponding user and movie feature vectors from U and V respectively
    pred_ratings = (users_factors * items_factors).sum(axis=1)
    pred_ratings = np.where(pred_ratings > 5, 5, pred_ratings)
    pred_ratings = np.where(pred_ratings < 1, 1, pred_ratings)

    # Compute and return RMSE with the predicted ratings
    npErr = set_ratings[:,2] - pred_ratings
    agg_npErr = np.mean(np.square(npErr))

    return pred_ratings, math.sqrt(agg_npErr)
