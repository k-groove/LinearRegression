
from pylab import scatter, show, title, xlabel, ylabel, plot, contour
from numpy import loadtxt, zeros, ones, array, linspace, logspace

import csv

# with open('housing_training.csv') as trainingFile:
#     readTraining = csv.reader(trainingFile, delimiter=',')
data = loadtxt('housing_training.csv', delimiter=',')

scatter(data[:, 0], data[:, 1], marker='o', c='b')
title('Profits distribution')
xlabel('Population of City in 10,000s')
ylabel('Profit in $10,000s')
show()

with open('housing_test.csv') as testFile:
    readTest = csv.reader(testFile, delimiter=',')
    for testRow in readTest:
        print(testRow[0], testRow[1])


def feature_normalize(X):
    mean_r = []
    std_r = []

    X_norm = X

    n_c = X.shape[1]
    for i in range(n_c):
        m = mean(X[:, i])
        s = std(X[:, i])
        mean_r.append(m)
        std_r.append(s)
        X_norm[:, i] = (X_norm[:, i] - m) / s

    return X_norm, mean_r, std_r


def compute_cost(X, y, theta):
    # Number of training samples
    m = y.size

    predictions = X.dot(theta)

    sqErrors = (predictions - y)

    J = (1.0 / (2 * m)) * sqErrors.T.dot(sqErrors)

    return J



# training data for 1.
# test data for 2,3,4
