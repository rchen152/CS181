import numpy as np
import util
import random
import cPickle
import math
from PIL import Image
import os

# kmeans implementation initialized with kmeans++
def kmeans_plus(m, k):
    mat = m.astype(float)
    num_pts = mat.shape[0]
    dim = mat.shape[1]
    mu = np.zeros((k,dim))
    mu[0] = mat[random.randrange(num_pts)]
    diff = mat - np.ones((num_pts, dim))*mu[0]
    dist = np.sum(diff*diff,axis = 1).reshape(num_pts,1)
    normalized = (np.transpose(dist/np.sum(dist,axis = 0)))[0]
    for i in range(1,k):
        rand_vec = np.random.multinomial(1,normalized,size=1)
        mu[i]=mat[np.argmax(rand_vec)]
        temp_diff = mat - np.ones((num_pts, dim))*mu[i]
        temp_dist = np.sum(temp_diff*temp_diff,axis = 1).reshape(num_pts,1)
        dist = np.min(np.concatenate((dist,temp_dist), axis = 1), axis =1).reshape(num_pts,1)
        normalized = (np.transpose(dist/np.sum(dist,axis = 0)))[0]
        
    resp = np.zeros((num_pts,1))
    result = np.zeros((num_pts,k))
    while(True):    
        fst = np.ones((num_pts,k))*np.sum(mat*mat, axis = 1).reshape(num_pts,1)
	product = np.dot(mat, np.transpose(mu))
	snd = np.ones((num_pts,k))*np.sum(mu*mu, axis = 1)
	result = fst - 2*product + snd
	temp_resp = np.argmin(result,axis = 1).reshape(num_pts,1)
	if(temp_resp == resp).all():
	     break
        resp = temp_resp
        for i in range(k):
    	    mu[i] = np.mean(mat[resp[:,0]==i,:],axis=0)
    error = math.sqrt(sum(np.min(result,axis = 1))/num_pts)
    return error
