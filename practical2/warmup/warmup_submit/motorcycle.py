import csv
import matplotlib.pyplot as plt
import numpy as np
import reg

# plot predictions overlaid on the original data from motorcycle.csv
def plot(x,y,data,title):
    plt.plot(x,y)
    plt.plot(data[:,0], data[:,1], 'ro')
    plt.title(title)
    plt.xlabel('time since impact (ms)')
    plt.ylabel('gforce')
    plt.show()
    plt.clf()

# define basis functions
poly_basis = [lambda x: 1, lambda x: x,lambda x : x**2, lambda x: x**3,
              lambda x : x**4]
sin_basis = [lambda x: 1, lambda x: np.sin(.1*x),lambda x : np.sin(.20*x),
             lambda x: np.sin(.30*x), lambda x : np.sin(.40*x)]

# format data
raw_data = np.loadtxt('motorcycle.csv', dtype=str, delimiter=';')
reader = csv.reader(raw_data, delimiter=',')
init = []
for row in reader:
    init.append(row)
arr = init[1:]
flt_data = map(lambda x: map(float,x),arr)
num_pts = len(flt_data)
data = np.array(flt_data)

# show a bunch of plots
x = np.linspace(0,60,1000)

y = reg.freq_reg(poly_basis, data, x)
plot(x,y,data,'Frequentist Polynomial Basis, No Regularization')

y = reg.freq_reg(sin_basis, data, x)
plot(x,y,data,'Frequentist Fourier Basis, No Regularization')

y = reg.freq_reg(poly_basis, data, x, LAMBDA = 10)
plot(x,y,data,'Frequentist Polynomial Basis, LAMBDA = 10')

y = reg.freq_reg(sin_basis, data, x, LAMBDA = 10)
plot(x,y,data,'Frequentist Fourier Basis, LAMBDA = 10')

y = reg.bayes_reg(poly_basis, data, x)
plot(x,y,data,'Bayesian Polynomial Basis, BETA = .05, ALPHA = 1')

y = reg.bayes_reg(sin_basis, data, x)
plot(x,y,data,'Bayesian Fourier Basis, BETA = .05, ALPHA = 1')
