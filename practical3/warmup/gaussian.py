import csv
import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as linalg

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

sigma_inv = linalg.inv(sum([counts[i]*var(i)
                            for i in range(len(fruits))])/sum(counts))

fruit_labels = ['apples','oranges','lemons']
plt_colors = ['#990000','#ff6600','#ccff33']
for i in range(len(fruits)):
    plt.scatter(fruits[i][:,0], fruits[i][:,1], c=plt_colors[i],
                label=fruit_labels[i])

x = np.arange(5,11,1)
for i,j in [(0,1),(0,2),(1,2)]:
    mu1 = avgs[i].reshape((1,2))
    mu2 = avgs[j].reshape((1,2))
    diff_left = (mu1.dot(sigma_inv)).dot(mu1.transpose())
    diff_right = (mu2.dot(sigma_inv)).dot(mu2.transpose())
    diff = diff_left[0,0] - diff_right[0,0]
    coeff_mat = 2*(mu1-mu2).dot(sigma_inv)
    y = (diff - coeff_mat[0,0]*x)/coeff_mat[0,1]
    plt.plot(x,y, label=fruit_labels[i] + '/' + fruit_labels[j])
plt.legend(loc='lower right')
plt.show()
