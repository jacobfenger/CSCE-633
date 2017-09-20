import math
import copy
import numpy as np
from sklearn.model_selection import KFold

# Compute difference between 2 points
def euclidean_distance(a, b):
    d = 0
    for i in range(len(a)):
        d += (a[i] - b[i]) ** 2
    return math.sqrt(d)

def find_neighbors(K, train, point):
    distances = []

    for i in range(len(train)):
        d = euclidean_distance(train[i], point)
        distances.append((d, i)) # Create a tuple of the distance and sample #

    # Only return the top K distances
    return sorted(distances)[:K]

def run_knn(K, truth, train, point):

    neighbors = find_neighbors(K, train, point)

    yes_votes = 0
    no_votes = 0
    for point in neighbors:
        if truth[point[1]] == 1:
            yes_votes += 1
        else:
            no_votes += 1

    if yes_votes >= no_votes:
        return 1
    else:
        return 0

def cross_validation(feature_set, truth_set, K):

    kf = KFold(n_splits=10)
    set_accuracy = []
    # Split the set indices between training and testing sets
    for train_index, test_index in kf.split(feature_set):
        errors = 0

        X_train, x_test = feature_set[train_index], feature_set[test_index]
        y_train, y_test = truth_set[train_index], truth_set[train_index]

        for i in range(len(x_test)):
            classification = run_knn(K, y_train, X_train, x_test[i])

            if classification != y_test[i]:
                errors += 1

        set_accuracy.append(errors)

    print("K VALUE:", K, "---- ERRORS:", sum(set_accuracy)/len(set_accuracy))
    return sum(set_accuracy)/len(set_accuracy)

def find_best_K(feature_set, truth_set):

    K = [ i for i in range(3, 53)]

    best_K = 1
    error_count = 10000000000
    for k in K:

        x = cross_validation(feature_set, truth_set, k)

        if x < error_count:
            error_count = x
            best_K = k

    print "BEST K", best_K, " --- Errors: ", error_count
    return best_K

def test_classifier(train_features, train_truth, test_features, test_truth, K):

    error_count = 0
    for p in range(len(test_features)):
        classification = run_knn(K, train_truth, train_features, test_features[p])

        if classification != test_truth[p]:
            error_count += 1

    print "ERROR_COUNT FOR TEST SET: ", error_count
