import numpy as np
import scipy.sparse as sp

import time
import math
import csv

import matplotlib.pyplot as plt


""" 
Process a whole dataset to return it as a Numpy array for easier computation

path : path to the dataset file
"""

def read_preprocess_data(path):
    with open(path, "r") as f:
        data = f.read().splitlines()

    data = [process_elem(elem) for elem in data[1:]]
    return np.array(data)


""" 
Process elements from a dataset to return a Numpy array

elem : row from the dataset to analyse and convert to an array
"""

def process_elem(elem):    
    u_i, r = elem.split(',')
    user, item = u_i.split("_")
    user = user.replace("r", "")
    item = item.replace("c", "")
    return [int(user), int(item), float(r)]
	

""" 
Compute the MSE between predictions and nonzero train elements

data          : complete dataset
user_features : latent vectors of each user
item_features : latent vectors of each movie
nz            : indices of every user-movie pairs that have a non-zero associated rating 
"""

def compute_error(data, user_features, item_features, nz):
    mse = 0
    pred = np.dot(user_features, item_features.T)

    for row, col in nz:
        mse += np.square((data[row, col] - pred[row, col]))

    return np.sqrt(mse/len(nz))

	
""" 
Initialize every user and item latent vectors

num_users    : the total number of users
num_items 	 : the total number of items
num_features : the set number of features in each latent vector
train        : train set resulting from the split of the complete dataset
test         : test set resulting from the split of the complete dataset
"""

def init_latent_vectors(num_users, num_items, num_features, train, test):
    # Initialize latent vectors

    user_features = np.random.rand(num_users, num_features)
    item_features = np.random.rand(num_items, num_features)

    item_sum = train.sum(axis=0)
    item_nnz = train.getnnz(axis=0)

    user_sum = train.sum(axis=1)
    user_nnz = train.getnnz(axis=1)

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

    a,b = train.nonzero()
    nz_train=list(zip(a,b))

    a,b = test.nonzero()
    nz_test=list(zip(a,b))
    
    return user_features, item_features, nz_train, nz_test


""" 
Create K different folds for Cross-Validation Purpose

data    : complete dataset
k_fold 	: the number of folds to create
seed    : variable for random shuffling
"""

def build_k_indices(data, k_fold, seed):
    # Build k indices for k-fold
    
    num_row = data.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval]
                 for k in range(k_fold)]
    return np.array(k_indices)


""" 
Convert Numpy arrays to Scipy matrices

train_split : train set resulting from the split of the whole dataset
test_split 	: test set resulting from the split of the whole dataset
num_users   : total number of user
num_items   : total number of items
"""

def convert_to_scipy(train_split, test_split, num_users, num_items):  
    train = sp.lil_matrix((num_users, num_items))
    for e in train_split:
        u = int(e[0])-1
        i = int(e[1])-1
        r = e[2]
        train[u,i] = r

    test = sp.lil_matrix((num_users, num_items))
    for e in test_split:
        u = int(e[0])-1
        i = int(e[1])-1
        r = e[2]
        test[u,i] = r

    return train, test

	
	
"""Cross-validation to find optimal learning rate"""
	
	
def cross_val_lrate():

    # CODE FOR K-FOLD CROSS-VALIDATION #


    # Get data

    folder_path = r'Data'
    data = read_preprocess_data(folder_path+"\data_train.csv")

    num_users = 10000
    num_items = 1000
    num_features = 25
    ks = [10, 20, 30, 40, 50]


    # Start SGD computation

    start = time.time()

    lrate = 0.0025
    lrates = [0.001, 0.002, 0.003, 0.004, 0.005]
    regularizer = 0.01
    regularizers = [0.001, 0.005, 0.01, 0.015, 0.02]
    #lambda_user = 0.1
    #lambda_item = 0.7
    max_epochs = 25

    seed = 1
    k_fold = 5
    k_indices = build_k_indices(data, k_fold, seed)
    rmse_tr = []
    rmse_te = []

    np.random.seed(988)

    for indx, lrate_val in enumerate(lrates):
        rmse_train_each_fold = []
        rmse_test_each_fold = []
    
        print("For lrate :", lrate_val)
    
        for k in range(k_fold):
        
            print("For fold :", k)
        
            test_split = data[k_indices[k]]
            train_split = data[np.delete(k_indices, k, 0).reshape(-1)]
        
            train, test = convert_to_scipy(train_split, test_split, num_users, num_items)
        
            lrate_val = lrates[indx]
        
            user_features, item_features, nz_train, nz_test = init_latent_vectors(num_users, num_items, num_features, train, test)
            for it in range(max_epochs):
            
                #shuffle the training rating indices
                np.random.shuffle(nz_train)
            
                #decrease step size
                lrate_val /= 1.2

                for d, n in nz_train:
                
                    #update latent vectors
                    item_info = item_features[n, :]
                    user_info = user_features[d, :]
                    err = train[d, n] - user_info.T.dot(item_info)

                    # calculate the gradient and update
                    item_features[n, :] += lrate_val * (err * user_info - regularizer * item_info)
                    user_features[d, :] += lrate_val * (err * item_info - regularizer * user_info)   

                train_rmse = compute_error(train, user_features, item_features, nz_train)
                test_rmse = compute_error(test, user_features, item_features, nz_test)
                print("iter: {}, RMSE on train set: {}, RMSE on test set: {}".format(it+1, train_rmse, test_rmse))
        
            rmse_train_each_fold.append(train_rmse)
            rmse_test_each_fold.append(test_rmse)
        
        rmse_tr.append(np.mean(rmse_train_each_fold))
        rmse_te.append(np.mean(rmse_test_each_fold))
    
    print ("time after Cross-Val computation :", time.time() - start)

    plt.semilogx(lrates, rmse_tr, marker=".", color='b', label='train error')
    plt.semilogx(lrates, rmse_te, marker=".", color='r', label='test error')
    plt.xlabel("Learning Rate")
    plt.ylabel("RMSE")
    plt.title("Effect of Learning Rate")
    plt.legend(loc=2)
    plt.grid(True)
    plt.savefig("Effect_learning_rate")
