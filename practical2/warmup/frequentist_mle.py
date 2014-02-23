import StringIO
import csv
import numpy as np
import matplotlib.pyplot as plt

raw_data = np.loadtxt('motorcycle.csv', dtype=str, delimiter=';')
reader = csv.reader(raw_data, delimiter=',')
data = []

for row in reader:
    data.append(row)
arr = data[1:]
flt_data = map(lambda x: map(float,x),arr)
num_data = len(flt_data)
data_array = np.array(flt_data)

basis_fns  = [lambda x: 1, lambda x: x,lambda x : x**2, lambda x: x**3, lambda x : x**4,lambda x: x**5, lambda x: x**6,lambda x : x**7, lambda x: x**8, lambda x : x**9]


'''basis_fns  = [lambda x: 1, lambda x: np.sin(.1*x),lambda x : np.sin(.20*x), lambda x: np.sin(.30*x), lambda x : np.sin(.40*x)]'''

fst_fn = basis_fns[0]

trans = [map(fst_fn, arr)]

for fn in (basis_fns[1:]):
    trans = np.concatenate((trans,[map(fn, data_array[:,0])]),axis = 0)
phi = np.transpose(trans)

pseudo_inv = np.linalg.pinv(phi)
coeffs = np.dot(pseudo_inv, data_array[:,1])
print coeffs

var_vect = data_array[:,1] - np.dot(phi, np.transpose(coeffs))
mle_variance = np.sum(var_vect * var_vect,axis = 0)/num_data
mle_std = np.sqrt(mle_variance)
print mle_std

def reg(x):
    out = 0.
    for i in range(len(basis_fns)):
        out += coeffs[i]*basis_fns[i](x)
    return out

x = np.linspace(0, 60,1000)
y = reg(x)
w = map(lambda t: t + mle_std,y)
z = map(lambda t: t - mle_std,y)

plt.plot(x,y)
plt.plot(x,w)
plt.plot(x,z)

plt.plot(data_array[:,0],data_array[:,1],'ro')
plt.show()

