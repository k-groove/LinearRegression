import csv
import numpy
import os
from tabulate import tabulate

# Delete table.html if it already exists
try:
    os.remove('table.html')
except OSError:
    pass


def getcolumnsums(matrix):
    columns = matrix.sum(axis=0)
    # number of rows is 300
    return columns / 300


with open("housing_training.csv") as data:
    b = getcolumnsums(numpy.loadtxt(open("housing_training.csv", "r"), delimiter=","))
    X = numpy.loadtxt(open("housing_test.csv", "r"), delimiter=",")
    y = X * b
    print(X)
    # print(b)

    # y =Xb + e
    # with open('table.html', 'a') as out:
    #     out.write("<style>td{border: 1px solid black;padding: 5px;}</style>")
    #     out.write(tabulate(trainingdata, headers="keys", tablefmt="html"))
