import csv
import numpy as np

raw_data = np.loadtxt('fruit.csv', dtype=str, delimiter=';')
reader = csv.reader(raw_data, delimiter=',')
reader.next()
fruits = [[],[],[]]
for row in reader:
    fruits[int(row[0])-1].append(map(lambda d: float(d), row[1:]))
fruits = [np.array(fruit) for fruit in fruits]

counts = [len(fruit) for fruit in fruits]
avgs = [sum(fruits[i][0:])/counts[i] for i in range(len(fruits))]

def var(fr_ind):
    var_mat = np.zeros((2,2))
    for i in range(counts[fr_ind]):
        diff = (fruits[fr_ind][i] - avgs[fr_ind]).reshape((1,2))
        var_mat += diff.transpose().dot(diff)
    return var_mat

sigma = sum([var(i) for i in range(len(fruits))])
