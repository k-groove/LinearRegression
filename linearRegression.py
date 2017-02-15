import csv
import operator
import os
from tabulate import tabulate

# Delete table.html if it already exists
try:
    os.remove('table.html')
except OSError:
    pass

sumX = 0

with open('housing_training.csv', 'r') as training:
    sumY = 0
    data2 = csv.reader(training, delimiter=',')
    rowcount = sum(1 for row in data2)
    for row in data2:
        sumY += float(row[13])
        x = row[0:13]
        print("")

# print(rowcount)
with open("housing_training.csv") as data:
    datareader = csv.reader(data, delimiter=',')
    with open('table.html', 'a') as out:
        out.write("<style>td{border: 1px solid black;padding: 5px;}</style>")
        out.write(tabulate(datareader, headers="keys", tablefmt="html"))
