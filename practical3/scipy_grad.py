import StringIO
import csv
import numpy as np
import matplotlib.pyplot as plt
import math
import scipy.optimize as opt

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
out_int = map(int, data[:,0])
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

def make_t(out):
    output_mat = np.zeros(( (NUM_CLASS, len_data) ))
    nat_arr = np.arange(NUM_CLASS)+1
    for i in range(len_data):
        output_mat[:,i] = (out[i] == nat_arr)    
    return output_mat

def hessian(y_mat, phi_mat):
    diag_mat = np.diag(np.sum(y_mat,axis = 1))
    key_mat =  diag_mat - np.dot(y_mat,y_mat.transpose())
    kronecker_mat = np.zeros(((len_data, NUM_CLASS * basis_len,NUM_CLASS * basis_len)))
    for i in range (len_data):
        outer = np.outer(phi_mat[i],phi_mat[i])
        kronecker_mat[i] = np.kron(key_mat, outer)
    return np.sum(kronecker_mat,axis = 0)

'''def gradient(y_mat, out,phi_mat):
    output_mat = make_t(out)
    diff = y_mat - output_mat
    grad = np.zeros(( (basis_len,NUM_CLASS) ))
    for i in range(NUM_CLASS):
        grad[:,i] = np.sum(diff[i] * phi_mat.transpose(),axis = 1)
    return grad'''

def gradient(y_mat, out,phi_mat):
    output_mat = make_t(out)
    diff = y_mat - output_mat
    grad = np.zeros(( (basis_len,NUM_CLASS) ))
    for i in range(basis_len):
        for j in range(NUM_CLASS):
            grad[i][j] = sum(diff[j][n]*phi_mat[n][i] for n in range(len_data))
    return grad

def grad(w_mat,out,inputs, basis, sigma):
    phi_mat = make_phi(inputs,basis)
    y_mat = make_y(w_mat,phi_mat,sigma)
    return gradient(y_mat,out,phi_mat)    

def grad_w(w):
    w_mat = w.reshape(basis_len,NUM_CLASS) 
    grad_mat = grad(w_mat,outputs,inputs,basis_fns,softmax)
    return np.concatenate(grad_mat,axis = 1)

def get_new_w(w_old, y_mat,out,phi_mat,small_const):
    grad = gradient(y_mat,out,phi_mat)
    grad_norm = math.sqrt (np.sum(np.sum(grad*grad,axis = 0),axis = 0))
    return w_old - small_const * grad/grad_norm

def get_new_w_hessian(w_old, y_mat, out, phi_mat):
    hess = hessian(y_mat,phi_mat)
    grad = gradient(y_mat,out,phi_mat)
    old_w_long = np.concatenate(w_old,axis = 1)
    grad_long = np.concatenate(grad,axis = 1)
    w_new = old_w_long - np.dot(np.linalg.inv(hess),grad_long)
    split = np.split(w_new,NUM_CLASS)
    bracket_split = map(lambda x: [x],split)
    return np.concatenate(bracket_split, axis = 0)

def make_phi(inputs,basis):
    phi_mat = np.zeros(( (len_data, basis_len) ))
    for i in range(len_data):
        for j in range(basis_len):
            phi_mat[i,j] = basis[j](inputs[i])
    return phi_mat

def cost_fn(y_mat,t_mat):
    cost = 0.
    for i in range(NUM_CLASS):
        for j in range(len_data):
            cost = cost - t_mat[i][j]*math.log(y_mat[i][j])
    return cost

def cost(w_mat,out,inputs,basis,sigma):
    phi_mat = make_phi(inputs,basis)
    y_mat = make_y(w_mat,phi_mat,sigma)
    t_mat = make_t(out)
    return cost_fn(y_mat,t_mat)

def cost_w(w):
    w_mat = w.reshape(basis_len,NUM_CLASS)
    return cost(w_mat,outputs,inputs,basis_fns,softmax)

def find_min_w(basis, inputs, out,sigma,small_const):
    t_mat = make_t(out)
    past_w_mat = np.random.random_sample(size = (basis_len , NUM_CLASS))
    phi_mat = make_phi(inputs,basis)
    past_y_mat = make_y(past_w_mat,phi_mat,sigma)
    current_w_mat = get_new_w(past_w_mat,past_y_mat,out,phi_mat,small_const)
    current_y_mat = make_y(current_w_mat,phi_mat,sigma)
#    print cost_fn(past_y_mat,t_mat)
#    print cost_fn(current_y_mat,t_mat)
    while(cost_fn(current_y_mat,t_mat) < cost_fn(past_y_mat, t_mat)):
        past_w_mat = current_w_mat
        past_y_mat = current_y_mat
        current_w_mat = get_new_w(past_w_mat, past_y_mat, out, phi_mat, small_const)
        current_y_mat = make_y(current_w_mat, phi_mat, sigma)
#        print cost_fn(past_y_mat,t_mat)
#        print cost_fn(current_y_mat,t_mat)

    return current_w_mat

#init_w_mat = np.random.random_sample(size = (basis_len , NUM_CLASS))
#init_w_mat = np.arange(basis_len*NUM_CLASS).reshape(basis_len,NUM_CLASS)
init_w_mat = np.random.random_sample(basis_len * NUM_CLASS)
best_w = opt.fmin_ncg(cost_w,init_w_mat,grad_w).reshape(basis_len,NUM_CLASS)
print best_w
phi_mat = make_phi(inputs,basis_fns)
best_y = make_y(best_w, phi_mat, softmax)
classes = np.argmax(best_y,axis = 0)+1
print classes

x = np.linspace(4,11,100)
i = 0
j = 1
def calc_line(i,j,vec):
    return map(lambda x: -(best_w[1][i] - best_w[1][j])/(best_w[2][i] - best_w[2][j]) *x -(best_w[0][i] - best_w[0][j])/(best_w[2][i]-best_w[2][j]), vec)

r = calc_line(0,1,x)
s = calc_line(1,2,x)
t = calc_line(2,0,x)

plt.plot(x,r)
plt.plot(x,s)
plt.plot(x,t)

fruit_labels = ['apples','oranges','lemons']
plt_colors = ['#990000','#ff6600','#ccff33']
for i in range(len_data):
    plt.scatter(inputs[i][0],inputs[i][1],c = plt_colors[out_int[i]-1],label = fruit_labels[out_int[i]-1])

plt.title('Logistic Regression')
plt.xlabel('Width (cm)')
plt.ylabel('Hieght (cm)')
plt.savefig('log_scipy.png')

