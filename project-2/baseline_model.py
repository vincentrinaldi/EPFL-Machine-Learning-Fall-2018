import numpy as np
import csv
from helpers import *
from pyspark import SparkConf
from pyspark.context import SparkContext

sc = SparkContext.getOrCreate(SparkConf().setMaster("local[*]"))

def prediction_bm():

    data_train_path = "Data/data_train.csv"
    data_subm_path = "Data/sample_submission.csv"

    # Load and parse the data
    data_ = sc.textFile(data_train_path)
    data_subm_ = sc.textFile(data_subm_path)
    header = data_.first()
    data_ = data_.filter(lambda line: line != header)
    data_subm_ = data_subm_.filter(lambda line: line != header)
    data = data_.map(lambda l: l.split('_')).map(lambda l: [l[0], l[1].split(',')]).map(lambda l: (l[0], l[1][0], int(l[1][1])))
    data_subm = data_subm_.map(lambda l: l.split('_')).map(lambda l: [l[0], l[1].split(',')]).map(lambda l: (l[0], l[1][0], int(l[1][1])))

    # Computing the main parameters
    mean = mean_(data)

    u_id_biases_rdd = u_id_biases_(data, mean)
    default_u_bias = np.mean(u_id_biases_rdd.map(lambda x: x[1]).collect())
    u_id_biases = u_id_biases_rdd.collectAsMap()

    i_id_biases_rdd = i_id_biases_(data, mean, u_id_biases)
    default_i_bias = np.mean(i_id_biases_rdd.map(lambda x: x[1]).collect())
    i_id_biases = i_id_biases_rdd.collectAsMap()

    # Compute predictions on submission file
    p_subm = data_subm.map(lambda x: (x[0], x[1]))
    predictions_subm = p_subm.map(lambda x: predict(x[0], x[1], mean, u_id_biases, i_id_biases, default_u_bias, default_i_bias)).map(lambda x: x[2]).collect()
    # Construct the id pairs: user-id_movie-id
    pairs = data_subm_.map(lambda l: l.split(',')).map(lambda l: l[0]).collect()

    # Creation of the csv file with predicted values
    with open("submission_bm.csv", 'w') as csvfile:
        fieldnames = ['Id', 'Prediction']
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for r1, r2 in zip(pairs, predictions_subm):
            writer.writerow({'Id':r1,'Prediction':np.clip(round(r2), 1, 5)})
