import numpy as np
import scipy.sparse as sp

import time
import math
import csv



"""Helper Functions For Matrix Factorization Method"""


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
Convert a Numpy array into a Scipy sparse matrix

folder_path : path to the dataset file
"""

def get_data(folder_path):
    data = read_preprocess_data(folder_path+"\data_train.csv")
    
    data_m = sp.lil_matrix((10000, 1000))
    for e in data :
        u = int(e[0])-1
        i = int(e[1])-1
        r = e[2]
        data_m[u,i] = r
    
    return data_m
	

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
Create a csv file from the dataset of the predicted ratings under the name 'MF_SGD.csv'

predictions    : dataset with predicted ratings after training on the given train set
test_data_path : datapath of the initial dataset for which we have to estimate the ratings
"""

def create_csv_submission(predictions, test_data_path = r'Data\sample_submission.csv'):

    def process_elem_csv(elem):
        u_i, r = elem.split(',')
        user, item = u_i.split("_")
        user = user.replace("r", "")
        item = item.replace("c", "")
        return [int(user)-1, int(item)-1, u_i]

    with open(test_data_path, "r") as f_in:
        test_data = f_in.read().splitlines()
        fieldnames = test_data[0].split(",")
        test_data = test_data[1:]

    with open('MF_SGD.csv', 'w') as f_out:
        writer = csv.DictWriter(f_out, delimiter=",", fieldnames=fieldnames, lineterminator = '\n')
        writer.writeheader()
        for line in test_data:
            user, item, user_item_id = process_elem_csv(line)
            prediction = np.clip(round(predictions[user, item]),1,5)
            prediction = prediction.astype(int)
            writer.writerow({fieldnames[0]: user_item_id,fieldnames[1]: prediction})
