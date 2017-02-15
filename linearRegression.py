import csv
import operator
from tabulate import tabulate

with open("housing_training.csv") as data:
    datareader = csv.reader(data, delimiter=',')
    with open('table.html', 'a') as out:
        out.write(tabulate(datareader, headers="keys", tablefmt="html"))
    # print(tabulate(datareader, headers="keys", tablefmt="html"))
