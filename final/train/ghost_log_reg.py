import StringIO
import csv
import numpy as np
import matplotlib.pyplot as plt
import math
import scipy as sp

NUM_CLASS = 5
EPSILON = .01

# load data from fruit.csv
raw_data = np.loadtxt('data/dumb_collection_ghost.csv', dtype=str, delimiter=';')
reader = csv.reader(raw_data, delimiter=' ')
init = []
for row in reader:
    init.append(row)
flt_data = map(lambda x: map(float,x),init)
num_pts = len(flt_data)
data = np.array(flt_data)
len_data = len(data)
out_int = map(int, data[:,1])
outputs = map(float, data[:,1])
inputs = sp.delete(data,1,1)

# choose basis functions
basis_fns = [lambda x: 1, lambda x: x[0],lambda x : x[1], lambda x : x[2], lambda x : x[3], lambda x : x[4], lambda x : x[5], lambda x : x[6], lambda x : x[7], lambda x : x[8], lambda x : x[9], lambda x : x[10], lambda x : x[11], lambda x : x[12], lambda x : x[13]]
basis_len = len(basis_fns)

# helper functions

def softmax(vec):
    sum = np.sum(math.e**vec)
    return (math.e**vec)/sum

def make_t(out):
    output_mat = np.zeros(( (NUM_CLASS, len_data) ))
    nat_arr = np.arange(NUM_CLASS)+1
    for i in range(len_data):
        output_mat[:,i] = (out[i] == nat_arr)    
    return output_mat
    
def make_phi(inputs,basis):
    phi_mat = np.zeros(( (len_data, basis_len) ))
    for i in range(len_data):
        for j in range(basis_len):
            phi_mat[i,j] = basis[j](inputs[i])
    return phi_mat

def make_y(w_mat,phi_mat,sigma):
    product = np.dot(w_mat.transpose(),phi_mat.transpose())
    temp_mat = np.zeros(((NUM_CLASS,len_data)))
    for i in range(len_data):
        temp_mat[:,i] = sigma(product[:,i]).transpose()
    return temp_mat

def gradient(y_mat, out,phi_mat):
    output_mat = make_t(out)
    diff = y_mat - output_mat
    grad = np.zeros(( (basis_len,NUM_CLASS) ))
    for i in range(basis_len):
        for j in range(NUM_CLASS):
            grad[i][j] = sum(diff[j][n]*phi_mat[n][i] for n in range(len_data))
    return grad
     
def get_new_w(w_old, y_mat,out,phi_mat,small_const):
    grad = gradient(y_mat,out,phi_mat)
    grad_norm = math.sqrt (np.sum(np.sum(grad*grad,axis = 0),axis = 0))
    return w_old - small_const * grad/grad_norm

def cost_fn(y_mat,t_mat):
    cost = 0.
    for i in range(NUM_CLASS):
        for j in range(len_data):
            cost = cost - t_mat[i][j]*math.log(y_mat[i][j])
    return cost

def find_min_w(basis, inputs, out,sigma,small_const):
    t_mat = make_t(out)
    past_w_mat = np.random.random_sample(size = (basis_len , NUM_CLASS))
    phi_mat = make_phi(inputs,basis)
    past_y_mat = make_y(past_w_mat,phi_mat,sigma)
    current_w_mat = get_new_w(past_w_mat,past_y_mat,out,phi_mat,small_const)
    current_y_mat = make_y(current_w_mat,phi_mat,sigma)
    while(cost_fn(current_y_mat,t_mat) < cost_fn(past_y_mat, t_mat)):
        past_w_mat = current_w_mat
        past_y_mat = current_y_mat
        current_w_mat = get_new_w(past_w_mat, past_y_mat, out, phi_mat,
                                  small_const)
        current_y_mat = make_y(current_w_mat, phi_mat, sigma)
    return current_w_mat

# print weights and classifications
best_w = find_min_w(basis_fns,inputs,outputs,softmax,EPSILON)
print best_w
phi_mat = make_phi(inputs,basis_fns)
best_y = make_y(best_w, phi_mat, softmax)
classes = np.argmax(best_y,axis = 0)
print classes

num_correct = 0
for i in range(len_data):
  if classes[i] == out_int[i] or (classes[i] == 4 and out_int[i] == 5):
    num_correct += 1
print (float(num_correct) / len_data)

'''# helper function to compute a line
def calc_line(i,j,vec):
    return map(lambda x: -(best_w[1][i] - best_w[1][j])/(best_w[2][i] -
                                                         best_w[2][j]) *x -
               (best_w[0][i] - best_w[0][j])/(best_w[2][i]-best_w[2][j]), vec)

# plot decision boundaries
x = np.linspace(4,11,100)
for (i,j) in [(0,1),(0,2),(1,2)]:
    plt.plot(x, calc_line(i,j,x))

# plot fruit.csv data
fruit_labels = ['apples','oranges','lemons']
plt_colors = ['#990000','#ff6600','#ccff33']
for i in range(len_data):
    plt.scatter(inputs[i][0],inputs[i][1],c = plt_colors[out_int[i]-1],label =
                fruit_labels[out_int[i]-1])

# prettify and show plot
plt.title('Logistic Regression')
plt.xlabel('Width (cm)')
plt.ylabel('Hieght (cm)')
plt.show()'''
