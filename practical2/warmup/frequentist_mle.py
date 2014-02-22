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

basis_fns  = [lambda x: 1, lambda x : x, lambda x: x**2, lambda x: x**3, lambda x : x**4]

fst_fn = basis_fns[0]

trans = [map(fst_fn, arr)]

for fn in (basis_fns[1:]):
    trans = np.concatenate((trans,[map(fn, data_array[:,0])]),axis = 0)
phi = np.transpose(trans)

pseudo_inv = np.linalg.pinv(phi)
coeffs = np.dot(pseudo_inv, data_array[:,1])
print coeffs

def reg(x):
    out = 0.
    for i in range(len(basis_fns)):
        out += coeffs[i]*basis_fns[i](x)
    return out
print coeffs[0]

x = np.linspace(0, 60)
y = reg(x)

plt.plot(x,y)

plt.plot(data_array[:,0],data_array[:,1],'ro')
plt.show()

