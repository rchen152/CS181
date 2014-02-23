import StringIO
import csv
import numpy as np
import matplotlib.pyplot as plt


basis_fns  = [lambda x: 1, lambda x: x,lambda x : x**2, lambda x: x**3, lambda x : x**4,lambda x: x**5]
#, lambda x: x**6,lambda x : x**7, lambda x: x**8, lambda x : x**9]


'''basis_fns  = [lambda x: 1, lambda x: np.sin(.1*x),lambda x : np.sin(.20*x), lambda x: np.sin(.30*x), lambda x : np.sin(.40*x)]'''
BETA = .05
ALPHA = 1.
basis_dim = len(basis_fns)
init_mean = np.zeros((basis_dim))
init_var = ALPHA * np.identity(basis_dim)


#formatting data
raw_data = np.loadtxt('motorcycle.csv', dtype=str, delimiter=';')
reader = csv.reader(raw_data, delimiter=',')
init = []
for row in reader:
    init.append(row)
arr = init[1:]
flt_data = map(lambda x: map(float,x),arr)
num_pts = len(flt_data)
data = np.array(flt_data)

#calculate Phi
fst_fn = basis_fns[0]

trans = [map(fst_fn, arr)]

for fn in (basis_fns[1:]):
    trans = np.concatenate((trans,[map(fn, data[:,0])]),axis = 0)
phi = np.transpose(trans)

#calculate updated mean and variance for coefficients
coeffs_updated_var = np.linalg.inv(np.linalg.inv(init_var) + BETA * (np.dot(trans,phi)))
mean_factor = np.dot(np.linalg.inv(coeffs_updated_var), init_mean) + BETA * (np.dot(trans,data[:,1]))
coeffs_updated_mean = np.dot(coeffs_updated_var,mean_factor)

#calculate the predictive distribution
def predictive_dist(x):
    pred_list = map(lambda t: t(x), basis_fns)
    pred_vect = np.array(pred_list)
    var = (np.dot(np.dot(np.transpose(pred_vect),coeffs_updated_var),pred_vect))+ 1/BETA
    mean = np.dot(np.transpose(coeffs_updated_mean),pred_vect)
    return [mean,var]

x = np.linspace(0, 60,1000)
y = np.zeros((len(x)))
w = np.zeros((len(x)))
z = np.zeros((len(x)))
for i in range (len(x)):
    pred = predictive_dist(x[i])
    y[i] = pred[0]
    w[i] = pred[0] + pred[1]
    z[i] = pred[0] - pred[1]

plt.plot(x,y)
plt.plot(x,w)
plt.plot(x,z)

plt.plot(data[:,0],data[:,1],'ro')
plt.show()
