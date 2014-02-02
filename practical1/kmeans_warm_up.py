import numpy as np
import util
import random

#matrix as list of column vector

def kmeans(m, k):
    mat = m.astype(float)
    num_pts = mat.shape[0]
    dim = mat.shape[1]
    resp = np.zeros((num_pts,1))
    for i in range(num_pts):
    	num = random.randrange(k)
	resp[i][0]=num
    mu = np.zeros((k,dim))
    result = np.zeros((num_pts,k))
    while(True):
        for i in range(k):
    	    mu[i] = np.mean(mat[resp[:,0]==i,:],axis=0)
        print np.multiply(mat,mat)
        fst = np.ones((num_pts,k))*np.sum(mat * mat, axis = 1).reshape(num_pts,1)
	product = np.dot(mat, np.transpose(mu))
	snd = np.ones((num_pts,k))*np.sum(mu*mu, axis = 1)
	result = fst - 2*product + snd
	temp_resp = np.argmin(result,axis = 1).reshape(num_pts,1)
	print sum(np.min(result,axis = 1))
        if(temp_resp == resp).all():
	     break
	resp = temp_resp
    return sum(np.min(result,axis = 1))
