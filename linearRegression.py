from statistics import mean
import numpy
import matplotlib.pyplot as plt
import random
from matplotlib import style
from tabulate import tabulate
from pylab import *

style.use('ggplot')

# Training data
training_data = loadtxt(open("housing_training.csv", "r"), delimiter=",")
training_data_transpose = training_data.transpose()
x_training = training_data_transpose[0:13].sum(axis=0)
y_training = training_data_transpose[13]

# Test data
test_data = loadtxt(open("housing_test.csv", "r"), delimiter=",")
test_data_transpose = test_data.transpose()
x_test = test_data_transpose[0:13].sum(axis=0)
y_test = test_data_transpose[13]

# Table of the coefficients in the linear regression model
print(tabulate(mat(x_training), headers="keys", tablefmt="plain"))


def getslope(x, y):
    # y = mx + b
    slope = (((mean(x) * mean(y)) - mean(x * y)) / ((mean(x) * mean(x)) - mean(x * x)))
    return slope


def getintercept(x, y, slope):
    intercept = mean(y) - slope * mean(x)
    return intercept


slope_training = getslope(x_training, y_training)
intercept_training = getintercept(x_training, y_training, slope_training)


def regress_line(x):
    regression_line = []
    for i in x:
        regression_line.append((slope_training * i) + intercept_training)
    return regression_line


line_test = regress_line(x_test)

predict_x = numpy.array(x_training)
predict_y = numpy.array([(slope_training * j) + intercept_training for j in predict_x])

plt.figure()
plt.scatter(x_test, y_test, color='green', label=str(len(x_test)) + ' Test Data Points')
plt.plot(x_test, line_test, color='blue', label='Regression Line')
plt.legend()
plt.show()
print(sum(y_test))
print(sum(x_test))

# x is a vector
# X is a matrix
# y is last column in housing_test.csv
# b is the optimal coefficients of the model

# RSS is not correct
rss = sum((y_test - (test_data_transpose/206 * x_test/206))) ** 2
print(rss)