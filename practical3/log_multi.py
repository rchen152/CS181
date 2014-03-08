import StringIO
import csv
import numpy as np
import matplotlib.pyplot as plt
import math

NUM_CLASS = 3

raw_data = np.loadtxt('fruit.csv', dtype=str, delimiter=';')
reader = csv.reader(raw_data, delimiter=',')
init = []

for row in reader:
    init.append(row)
arr = init[1:]
flt_data = map(lambda x: map(float,x),arr)
num_pts = len(flt_data)
data = np.array(flt_data)
len_data = len(data)

inputs = data[:,1:]
outputs = map(float, data[:,0])


#fix these basis functions
basis_fns = [lambda x: 1, lambda x: x[0],lambda x : x[1]]
basis_len = len(basis_fns)

def softmax(vec):
    sum = np.sum(math.e**vec)
    return (math.e**vec)/sum

def make_y(w_mat,phi_mat,sigma):
    product = np.dot(w_mat.transpose(),phi_mat.transpose())
    temp_mat = np.zeros(((NUM_CLASS,len_data)))
    for i in range(len_data):
        temp_mat[:,i] = sigma(product[:,i]).transpose()
    return temp_mat

def hessian(y_mat, phi_mat):
    diag_mat = np.diag(np.sum(y_mat,axis = 1))
    key_mat =  diag_mat - np.dot(y_mat,y_mat.transpose())
    kronecker_mat = np.zeros(((len_data, NUM_CLASS * basis_len,NUM_CLASS * basis_len)))
    for i in range (len_data):
        outer = np.outer(phi_mat[i],phi_mat[i])
        kronecker_mat[i] = np.kron(key_mat, outer)
    return np.sum(kronecker_mat,axis = 0)

def gradient(y_mat, out,phi_mat):
    output_mat = np.zeros(( (NUM_CLASS, len_data) ))
    nat_arr = np.arange(NUM_CLASS)+1
    for i in range(len_data):
        output_mat[:,i] = (out[i] == nat_arr)
    diff = y_mat - output_mat
    grad = np.zeros(( (NUM_CLASS,basis_len) ))
    for i in range(NUM_CLASS):
        grad[i] = np.sum(diff[i] * phi_mat.transpose(),axis = 1)
    return grad

def get_new_w(w_old, y_mat, out, phi_mat):
    hess = hessian(y_mat,phi_mat)
    grad = gradient(y_mat,out,phi_mat)
    old_w_long = np.concatenate(w_old,axis = 1)
    grad_long = np.concatenate(grad,axis = 1)
    w_new = old_w_long - np.dot(np.linalg.inv(hess),grad_long)
    split = np.split(w_new,NUM_CLASS)
    bracket_split = map(lambda x: [x],split)
    return np.concatenate(bracket_split, axis = 0)

def single_iteration(w_old, inputs, out,basis,sigma):
    phi_mat = np.zeros(( (len_data, basis_len) ))
    for i in range(len_data):
        for j in range(basis_len):
            phi_mat[i,j] = basis[j](inputs[i])
    y_mat = make_y(w_old,phi_mat,sigma)
    return get_new_w(w_old, y_mat, out, phi_mat)

def find_min_w(basis, inputs, out,sigma,epsilon):
    w_mat_init = np.random.random_sample(size = (basis_len , NUM_CLASS))
    current_w = single_iteration(w_mat_init,inputs, out,basis,sigma) 
    past_w = w_mat_init
    while( np.sum(np.sum((current_w - past_w)*(current_w - past_w),axis = 0),axis=0)> epsilon):
        past_w = current_w
        current_w = single_iteration(past_w,inputs, out,basis,sigma)
    return current_w


print find_min_w(basis_fns,inputs,outputs,softmax,100)
