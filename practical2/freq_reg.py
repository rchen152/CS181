import StringIO
import csv
import numpy as np
import matplotlib.pyplot as plt

def coeffs(basis_fns, train_data, LAMBDA = 0):
    fst_fn = basis_fns[0]
    trans = [map(fst_fn, train_data)]
    num_pts = len(train_data[:,0])
    for fn in (basis_fns[1:]):
        trans = np.concatenate((trans,[map(fn, train_data[:,0])]),axis = 0)
    phi = np.transpose(trans)
    temp_pseudo = LAMBDA * np.identity(len(basis_fns))+np.dot(trans, phi)
    pseudo_inv = np.dot(np.linalg.inv(temp_pseudo),trans)
    coeffs = np.dot(pseudo_inv, train_data[:,1])
    return coeffs

def product(x, coeffs, basis_fns):
    out = 0.
    for i in range(len(basis_fns)):
        out += coeffs[i]*basis_fns[i](x)
    return out

def regress(basis_fns, train_data, test_data, LAMBDA = 0):
    cs = coeffs(basis_fns, train_data, LAMBDA)
    return [product(test_data[i,0], cs, basis_fns)
            for i in range(test_data.shape[0])]
