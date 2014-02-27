import StringIO
import csv
import numpy as np
import matplotlib.pyplot as plt

raw_data = np.loadtxt('motorcycle.csv', dtype=str, delimiter=';')
reader = csv.reader(raw_data, delimiter=',')
init = []

for row in reader:
    init.append(row)
arr = init[1:]
flt_data = map(lambda x: map(float,x),arr)
num_pts = len(flt_data)
data = np.array(flt_data)

'''basis_fns  = [lambda x: 1, lambda x: x,lambda x : x**2, lambda x: x**3, lambda x : x**4]'''


basis_fns  = [lambda x: 1, lambda x: np.sin(.1*x),lambda x : np.sin(.20*x), lambda x: np.sin(.30*x), lambda x : np.sin(.40*x)]

fst_fn = basis_fns[0]

trans = [map(fst_fn, arr)]

for fn in (basis_fns[1:]):
    trans = np.concatenate((trans,[map(fn, data[:,0])]),axis = 0)
phi = np.transpose(trans)

pseudo_inv = np.linalg.pinv(phi)
coeffs = np.dot(pseudo_inv, data[:,1])
print coeffs

var_vect = data[:,1] - np.dot(phi, np.transpose(coeffs))
mle_variance = np.sum(var_vect * var_vect,axis = 0)/num_pts
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

plt.plot(data[:,0],data[:,1],'ro')
plt.title('Frequentist Fourier Basis, No Regularization')
plt.xlabel('time since impact (ms)')
plt.ylabel('gforce')

plt.savefig('freq_fourier_noreg.png')

