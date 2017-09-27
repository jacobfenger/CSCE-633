import matplotlib.pyplot as plt
from numpy.linalg import inv
import csv
import numpy as np
import inspection

def strip_useless_data(features, outcomes):

    new_features = []
    new_outcomes = []

    for i in range(len(outcomes)):
        if outcomes[i] != 0:
            new_features.append(features[i])
            new_outcomes.append(outcomes[i])
    return new_features, new_outcomes


# The log graph addresses skewed data and we can view a more normal distribution
# of the outcomes with the log graph. It decreases the variability of the data.
def plot_outcome_histogram(outcomes):
    plt.hist(outcomes, color='lime')
    plt.gcf().set_facecolor('white')
    plt.title('Histogram of Burned Area ')
    plt.ylabel('Bin Count')
    plt.xlabel('Burned Area')
    plt.show()

    plt.hist(np.log(outcomes))
    plt.gcf().set_facecolor('white')
    plt.title('Histogram of Logarithimically Scaled Burned Area')
    plt.ylabel('Bin Count')
    plt.xlabel('Burned Area (Log Scaled)')
    plt.show()

def compute_ols(X, Y):
        X_t = np.transpose(X)
        w_1 = inv(np.dot(X_t, X))
        w_2 = np.dot(X_t, Y)

        # Compute optimal weight vector
        w = np.dot(w_1, w_2)
        return w

# Computes RSS with testin data and computed weight vector
def compute_RSS(w, X, Y):
    RSS = np.dot(np.transpose((Y - np.dot(X, w))), (Y - np.dot(X, w)))
    print "RSS:", RSS

    return np.dot(X, w)

def main():
    train_ftrs, train_outcome = inspection.read_data('train.csv')
    test_ftrs, test_outcome = inspection.read_data('test.csv')

    train_ftrs, train_outcome = strip_useless_data(train_ftrs, train_outcome)
    test_ftrs, test_outcome = strip_useless_data(test_ftrs, test_outcome)

    # Append 1s to the first column for the feature data
    train_ftrs = np.insert(train_ftrs, 0, 1, axis=1)
    test_ftrs = np.insert(test_ftrs, 0, 1, axis=1)

    train_ftrs = np.square(train_ftrs)
    test_ftrs = np.square(test_ftrs)

    #plot_outcome_histogram(train_outcome)

    weight_vector = compute_ols(train_ftrs, train_outcome)
    predicted = compute_RSS(weight_vector, test_ftrs, test_outcome)

    print np.corrcoef(test_outcome, predicted)[0, 1]

if (__name__) == '__main__':
    main()
