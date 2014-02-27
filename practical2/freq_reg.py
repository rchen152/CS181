import StringIO
import csv
import numpy as np
import matplotlib.pyplot as plt

def regress(basis_fns, train_data, test_data, LAMBDA = 0):

    fst_fn = basis_fns[0]

    trans = [map(fst_fn, train_data)]

    num_pts = len(train_data[:,0])

    for fn in (basis_fns[1:]):
        trans = np.concatenate((trans,[map(fn, train_data[:,0])]),axis = 0)
    phi = np.transpose(trans)

    temp_pseudo = LAMBDA * np.identity(len(basis_fns))+np.dot(trans, phi)
    pseudo_inv = np.dot(np.linalg.inv(temp_pseudo),trans)

    coeffs = np.dot(pseudo_inv, train_data[:,1])
    print coeffs

    var_vect = train_data[:,1] - np.dot(phi, np.transpose(coeffs))
    mle_variance = np.sum(var_vect * var_vect,axis = 0)/num_pts
    mle_std = np.sqrt(mle_variance)
    print mle_std

    def reg(x):
        out = 0.
        for i in range(len(basis_fns)):
            out += coeffs[i]*basis_fns[i](x)
        return out

    return reg(test_data)
'''x = np.linspace(0, 60,1000)
y = reg(x)
w = map(lambda t: t + mle_std,y)
z = map(lambda t: t - mle_std,y)

plt.plot(x,y)
plt.plot(x,w)
plt.plot(x,z)

plt.plot(data[:,0],data[:,1],'ro')
plt.title('Frequentist Polynomial Basis, LAMBDA = 10')
plt.xlabel('time since impact (ms)')
plt.ylabel('gforce')

plt.savefig('freq_polys.png')'''
