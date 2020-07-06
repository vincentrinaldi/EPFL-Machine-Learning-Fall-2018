# -*- coding: utf-8 -*-
import numpy as np



"""Helper Functions For Machine Learning Methods"""


"""Compute the mean of the ratings"""

def mean_(data):
    return data.map(lambda x: x[2]).mean()

"""Compute the bias of each user"""

def u_id_biases_(data, m):
    a = data.map(lambda x: (x[0], x[2]-m)).aggregateByKey((0,0),
                                                       lambda acc, val: (acc[0]+val, acc[1]+1),
                                                       lambda ac1, ac2: (ac1[0]+ac2[0], ac1[1]+ac2[1]))
    u_id_biases = a.map(lambda x: (x[0], x[1][0]/x[1][1]))
    return u_id_biases


"""Compute the bias of each item"""

def i_id_biases_(data, m, u_id_biases):
    a = data.map(lambda x: (x[1], x[2] - m - u_id_biases[x[0]])).aggregateByKey((0,0),
                                                       lambda acc, val: (acc[0]+val, acc[1]+1),
                                                       lambda ac1, ac2: (ac1[0]+ac2[0], ac1[1]+ac2[1]))
    i_id_biases = a.map(lambda x: (x[0], x[1][0]/x[1][1]))
    return i_id_biases


"""Given a pair of user and item, predict the rating"""

def predict(u_id, i_id, mean, u_b, i_b, default_u_bias, default_i_bias):
    if u_id in u_b.keys():
        user_bias = u_b[u_id]
    else:
        user_bias = default_u_bias
    if i_id in i_b.keys():
        item_bias = i_b[i_id]
    else:
        item_bias = default_i_bias

    return (u_id, i_id, mean + user_bias + item_bias)
