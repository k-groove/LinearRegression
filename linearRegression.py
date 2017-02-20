# Barry Oliver
# 2/19/2017

import matplotlib.pyplot as plt1
import matplotlib.pyplot as plt2
from pylab import *

style.use('ggplot')

# Training data array
training_data = loadtxt(open("housing_training.csv", "r"), delimiter=",")

# Remove empty rows
for i in training_data:
    if len(i) == 0:
        training_data.remove(i)

# Test data array
test_data = loadtxt(open("housing_test.csv", "r"), delimiter=",")

# Remove empty rows
for i in training_data:
    if len(i) == 0:
        training_data.remove(i)

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
print("Coefficients")
print(coefficients)
print("-"*60)

# Predict y values for test_data
# xvalues from test data
x_test = []
for i in test_data:
    x_test.append(i[:len(i) - 1])

x_test = np.array(x_test, dtype=float)
# y values from test data
y_test = []
for i in test_data:
    y_test.append(i[len(i) - 1])
y_test = np.array(y_test, dtype=float)


# multiply x value per row by coefficient and add to get y
def predict_y(z, xvalues):
    pred_y = []
    y_intercept_coef = z[0]
    for i in xvalues:
        temp_y = y_intercept_coef
        index = 0
        for j in z[1:]:
            temp_calc = j * i[index]
            temp_y += temp_calc
            index += 1
        pred_y.append(temp_y)
    return pred_y


y_prediction = predict_y(coefficients, x_test)
index = 0

x3 = []
y3 = []
index = 0
while index <= 50:
    x3.append(index)
    y3.append(index)
    index += 1

# Plot showing prediction vs ground truth
plt1.figure()
plt1.scatter(y_prediction, y_test, color='green', label='Test Data Points')
plt1.plot(x3, y3, "b")
plt1.ylabel("Test data")
plt1.xlabel("Predicted values")
plt.legend()
plt1.show()

# diff of prediction vs ground truth
ind = []
diffs = []
count = 0
for i in y_test:
    ind.append(count)
    diffs.append(y_prediction[count] - i)
    count += 1

plt.figure()
plt2.scatter(ind, diffs, color='orange', label='Difference')
plt2.ylabel("Difference")
plt2.xlabel("Sample")
plt2.show()

# Residual Sum of Squares
yt = y.transpose()
bt = coefficients.transpose()
#  y_transpose * y
y_yt = np.dot(yt, y)
# x_transpose * y
y_xt = np.dot(xt, y)
# b_transpose * (y*x_transpose)
b_y_xt = np.dot(bt, y_xt)
rss = y_yt - b_y_xt
print("Residual Sum of Squares")
print(rss)
