from statistics import mean
import numpy
import matplotlib.pyplot as plt
import random
from matplotlib import style
from tabulate import tabulate
from pylab import *

style.use('ggplot')

# Training data array
training_data = loadtxt(open("housing_training.csv", "r"), delimiter=",")

# Remove empty rows
r = 0
for i in training_data:
    if len(i) == 0:
        training_data.remove(i)
    r += 1

# Test data array
test_data = loadtxt(open("housing_test.csv", "r"), delimiter=",")

# Remove empty rows
for i in training_data:
    if len(i) == 0:
        training_data.remove(i)
    r += 1

# train the model
x = []
# z is used as a temporary array
z = []
for i in training_data:
    z = [1]
    for j in i[:13]:
        z.append(j)
    x.append(z)

y = []
for i in training_data:
    y.append(i[13])

# numpy array with type float
x = np.array(x, dtype=float)
y = np.array(y, dtype=float)
xt = x.transpose()
x_xt = np.dot(xt, x)
x_inverse = inv(x_xt)
xt_y = np.dot(xt, y)
coefficients = np.dot(xt_y, x_inverse)
print(coefficients)


# Returns the slope
def getslope(x, y):
    # y = mx + b
    slope = (((mean(x) * mean(y)) - mean(x * y)) / ((mean(x) * mean(x)) - mean(x * x)))
    return slope


# Returns the intercept
def getintercept(x, y, slope):
    intercept = mean(y) - slope * mean(x)
    return intercept


slope_training = getslope(xt, y)
intercept_training = getintercept(x, y, slope_training)


def regress_line(x):
    regression_line = []
    for i in x:
        regression_line.append((slope_training * i) + intercept_training)
    return regression_line


testx = test_data[:13]
testxsum = testx.sum(axis=0) / 206
testy = test_data[13]
line_test = regress_line(testxsum)
print(line_test)
plt.figure()
plt.scatter(testxsum, testy, color='green', label='Test Data Points')
plt.plot(testxsum, line_test, color='blue', label='Regression Line')
plt.legend()
plt.show()

# Residual Sum of Squares
# rss = y' * y - b' * x' * y
yt = y.transpose()
bt = coefficients.transpose()
#  yt * y
y_yt = np.dot(yt, y)
# xt * y
y_xt = np.dot(xt, y)
# bt * (y*xt)
b_y_xt = np.dot(bt, y_xt)
rss = y_yt - b_y_xt
print(rss)
