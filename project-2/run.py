import pandas as pd
import numpy as np

from matrix_factorization import *
from svd_regularization import *

from cross_val_features import *
from cross_val_lrate import *
from cross_val_regularizer import *

from baseline_model import *
from als_baseline_model import *

import csv
import math
import time



"""Main Function To Run Our Different Implemented """


def main():

	# Call the SVD Regularization technique 
	# This model generated our best score but is not efficient in terms of computation time
	svd_regularization()
	
	# Call the Matrix Factorization technique using SGD
	# This technique is the one on which we performed the most detailed tuning and analysis
    #matrix_factorization()
	
	# Functions for K-Folds Cross-Validation
	# Those cross-validations have been made to tune each hyperparameter for the SGD Matrix Factorization technique
	#cross_val_features()
	#cross_val_lrate()
	#cross_val_regularizer()
	
	# Call the Baseline technique or the Matrix Factorization technique using ALS
	# These are alternative models that we also tried to implement to check if they perform better than the others
	#prediction_bm()
	#prediction_als_bm()

	
if __name__ == '__main__':
    main()
