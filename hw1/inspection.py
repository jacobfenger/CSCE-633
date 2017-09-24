import csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing

def read_data(input_file):
    features = []
    outcome = []

    with open(input_file, 'rb') as csvfile:
        reader = csv.reader(csvfile)
        descriptions = reader.next()
        for row in reader:

            row = map(float, row)

            features.append(row[:12])
            outcome.append(row[12])

    return np.asarray(features), np.asarray(outcome)

def dichotomize_outcome(outcomes):
    for i in range(len(outcomes)):
        if outcomes[i] > 0:
            outcomes[i] = 1

    return outcomes

def normalize_features(features, min_range, max_range):

    # Used scikitlearn normalize
    min_max_scaler = preprocessing.MinMaxScaler()
    return min_max_scaler.fit_transform(features)

    # feature_min = features.min(axis=0)
    # feature_max = features.max(axis=0)
    #
    # return (features - feature_min) / (feature_max - feature_min)

# outcome corresponds to area
def explore_inputs(features, outcome):

    # We are not graphing the outcomes of 0 area affected
    outcome[outcome == 0] = np.nan

    #Scatterplot of the month of year vs. the burned area
    mnth = plt.hist(features[:,2], 12, histtype='bar', color="lime")
    plt.gcf().set_facecolor('white')
    plt.title("Distribution of the Months of Year in the Feature Set")
    plt.ylabel("Count")
    plt.xlabel("Month of Year")
    #plt.xlim([1, 13])
    #plt.ylim([0, max(outcome)+(max(outcome)/10)])

    plt.show()

    temp = plt.scatter(features[:,8], outcome, color="red")
    plt.gcf().set_facecolor('white')
    # m, b = np.polyfit(features[:, 8], outcome, 1)
    # plt.plot(outcome, m*outcome + b, '-')
    plt.title("Temperature vs. Burned Area (Logarithimically Scaled)")
    plt.ylabel("Area Affected")
    plt.xlabel("Temperature in Celsius")
    plt.ylim([0, 1800])
    plt.xlim([0, max(features[:,8])+2])
    plt.yscale('log')
    plt.show()

    temp = plt.scatter(features[:,9], outcome, color='blue')
    plt.gcf().set_facecolor('white')
    # m, b = np.polyfit(features[:, 9], outcome, 1)
    # plt.plot(outcome, m*outcome + b, '-')
    plt.title("Relative Humidity vs. Burned Area (Logarithmically Scaled)")
    plt.ylabel("Area Affected")
    plt.xlabel("Relative Humidity in Percent")
    plt.xlim([0, max(features[:,9])])
    plt.ylim([0, 1700])
    plt.yscale('log')
    plt.show()

    plt.scatter(features[:,7], features[:,6], color='blue')
    plt.gcf().set_facecolor('white')
    plt.title("Comparison of the Initial Spread Index (ICI) and Drought Code Levels (DC)")
    plt.ylabel('DC Levels')
    plt.xlabel('Initial Spread Index (ICI)')
    plt.ylim([0, 900])
    plt.xlim([0, max(features[:,7]) + 1])
    plt.show()

def main():
    train_ftrs, train_outcome = read_data('train.csv')
    #explore_inputs(train_ftrs, train_outcome)


if (__name__) == '__main__':
    main()
