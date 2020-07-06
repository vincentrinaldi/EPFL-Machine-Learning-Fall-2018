import numpy as np
import pandas as pd
from helpers import *
from pyspark import SparkConf
from pyspark.context import SparkContext
from pyspark.mllib.recommendation import ALS, MatrixFactorizationModel, Rating

sc = SparkContext.getOrCreate(SparkConf().setMaster("local[*]"))

def prediction_als_bm():

    data_train_path = "Data/data_train.csv"
    data_subm_path = "Data/sample_submission.csv"

    # Load and parse the data
    data = sc.textFile(data_train_path)
    data_subm = sc.textFile(data_subm_path)
    header = data.first()
    data = data.filter(lambda line: line != header).map(lambda l: l.split('_')).map(lambda l: [l[0], l[1].split(',')]).map(lambda l: (l[0], l[1][0], int(l[1][1])))
    data_subm = data_subm.filter(lambda line: line != header).map(lambda l: l.split('_')).map(lambda l: [l[0], l[1].split(',')]).map(lambda l: (l[0], l[1][0], int(l[1][1])))

    # Construct the set of users and items of the data train set and data test set (on submission)
    users = set(data.map(lambda l: l[0]).collect())
    items = set(data.map(lambda l: l[1]).collect())
    users_s = set(data_subm.map(lambda l: l[0]).collect())
    items_s = set(data_subm.map(lambda l: l[1]).collect())

    users_union = users.union(users_s)
    items_union = items.union(items_s)
    # We only need to care on new items since there are no new users in the sample.
    new_items = items_union.difference(items)

    # Take the pairs we want to predict using als
    data_subm_als = data_subm.filter(lambda l: l[1] not in new_items)
    # Take the pairs we want to predict using the baseline model
    data_subm_bm = data_subm.filter(lambda l: l[1] in new_items)

    # Convert the user and item ids to integer
    d = data.map(lambda l: (int(l[0][1:]), int(l[1][1:]), l[2]))
    d_subm = data_subm_als.map(lambda l: (int(l[0][1:]), int(l[1][1:]), l[2]))

    # Create the Rating objects for the library
    ratings = d.map(lambda l: Rating(l[0], l[1], l[2]))
    ratings_subm = d_subm.map(lambda l: Rating(l[0], l[1], l[2]))
    # Build the recommendation model using Alternating Least Squares
    numIterations = 20
    rank = 20              # Rank
    l = 10**-5             # Regularization parameter
    model = ALS.train(ratings, rank, numIterations, lambda_=l)                              # trained on data_train
    testdata = ratings_subm.map(lambda p: (p[0], p[1]))
    predictions_als = model.predictAll(testdata).map(lambda r: ((r[0], r[1]), r[2]))        # predict the pairs on submission

    # Convert back the ids to string
    final_predictions_als = predictions_als.map(lambda l: ('r'+str(l[0][0]), 'c'+str(l[0][1]), l[1])).collect()

    # Predicting the pairs (new data) with the baseline model
    mean = mean_(data)
    u_id_biases_rdd = u_id_biases_(data, mean)
    default_u_bias = np.mean(u_id_biases_rdd.map(lambda x: x[1]).collect())
    u_id_biases = u_id_biases_rdd.collectAsMap()
    i_id_biases_rdd = i_id_biases_(data, mean, u_id_biases)
    default_i_bias = np.mean(i_id_biases_rdd.map(lambda x: x[1]).collect())
    i_id_biases = i_id_biases_rdd.collectAsMap()

    final_predictions_bm = data_subm_bm.map(lambda x: (x[0], x[1], predict(x[0], x[1], mean, u_id_biases, i_id_biases, default_u_bias, default_i_bias)[2])).collect()

    # Combine the predictions (stack the predictions)
    final_predictions = final_predictions_als + final_predictions_bm

    # Reordering the predictions to have the same order as in the submission file in order to create the csv submission
    dic_users = {(k1+'_'+k2) : np.clip(round(v),1,5) for (k1,k2,v) in final_predictions}
    pred = pd.Series(dic_users)
    m = pd.DataFrame(final_predictions)
    subm = pd.read_csv(data_subm_path)
    subm.set_index('Id', drop=True, inplace=True)
    m['Id'] = m[0] + '_' + m[1]
    m.set_index('Id', drop=True, inplace=True)
    subm['Prediction']  = pred

    subm.to_csv('submission_bm_als.csv')
