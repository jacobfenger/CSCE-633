###################################
# Main file for CS 633
# Jacob Fenger
#################################

import matplotlib.pyplot as plt
import csv
import numpy as np
import inspection
import KNN

def main():
    train_ftrs, train_outcome = inspection.read_data('train.csv')
    test_ftrs, test_outcome = inspection.read_data('test.csv')

    # Dichotomize outcomes to be 0 or 1
    train_outcome = inspection.dichotomize_outcome(train_outcome)
    test_outcome = inspection.dichotomize_outcome(test_outcome)

    # Normalize features to be between 0 and 1
    train_ftrs = inspection.normalize_features(train_ftrs, 0, 1)
    test_ftrs = inspection.normalize_features(test_ftrs, 0, 1)

    #x = KNN.run_knn(3, train_outcome, train_ftrs, test_ftrs[10])

    k = KNN.find_best_K(train_ftrs, train_outcome)
    print k
    #error_count = KNN.cross_validation(train_ftrs, train_outcome, 5)
    #print 'outcome', error_count

    #KNN.test_classifier(train_ftrs, train_outcome, test_ftrs, test_outcome, 9)



if (__name__) == '__main__':
    main()
