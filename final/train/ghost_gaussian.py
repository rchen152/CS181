import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as linalg
from scipy.stats import multivariate_normal as norm

# load training data
data = np.loadtxt(open('../data/train_ghost.csv'), delimiter = ' ')
good_feats = [3,4,5,6,7,13,14,15]
ghosts = [data[data[:,1] == i][:,good_feats] for i in [0,1,2,3,5]]
num_feats = len(good_feats)

# count number and average dimension of each type of ghost
counts = [len(ghost) for ghost in ghosts]
avgs = [sum(ghosts[i])/counts[i] for i in range(len(ghosts))]

# function to compute variance matrix
def var(gh_ind):
    var_mat = np.zeros((num_feats,num_feats))
    for i in range(counts[gh_ind]):
        diff = (ghosts[gh_ind][i] - avgs[gh_ind]).reshape((1,num_feats))
        var_mat += diff.transpose().dot(diff)
    return var_mat

sigma_inv = linalg.inv(sum([counts[i]*var(i)
                            for i in range(len(ghosts))])/sum(counts))
test = np.loadtxt(open('../data/validate_ghost.csv'), delimiter = ' ')
test_feats = test[:,good_feats]
for i in range(len(test)):
    for j in range(len(ghosts)):
        print norm.pdf(test_feats[i], mean=avgs[j], cov=sigma_iv)
    break
