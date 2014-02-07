import numpy as np
import matplotlib.pyplot as plt
import util
import random
import cPickle
import math
from PIL import Image
import os

# converts a cifar 32x32 image array to a pixel array, then saves as a png
def get_image(cifar_array, name):
    image_array = np.zeros((32,32,3), np.uint8)
    for i in range(32):
        for j in range(32):
            rgb = np.zeros(3, np.uint8)
            for k in range(3):
                rgb[k] = cifar_array[(i*32) + (j%32) + (k*1024)]
            image_array[i,j] = rgb
    img = Image.fromarray(image_array)
    img.save(name)

# kmeans implementation
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
    #keys = np.zeros((k, 1))
    steps = 0
    score = []
    while(True):
        steps = steps + 1
        for i in range(k):
    	    mu[i] = np.mean(mat[resp[:,0]==i,:],axis=0)
        fst = np.ones((num_pts,k))*np.sum(mat*mat, axis = 1).reshape(num_pts,1)
	product = np.dot(mat, np.transpose(mu))
	snd = np.ones((num_pts,k))*np.sum(mu*mu, axis = 1)
	result = fst - 2*product + snd
	temp_resp = np.argmin(result,axis = 1).reshape(num_pts,1)
	score.append(math.sqrt(sum(np.min(result,axis = 1))/num_pts))
        if(temp_resp == resp).all():
	     break
        resp = temp_resp
    #keys = np.argmin(result, axis = 0)
    error = math.sqrt(sum(np.min(result,axis = 1))/num_pts)
    result = np.zeros((2))
    result[0] = error
    result[1] = steps
    return score

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
    #keys = np.zeros((k, 1))
    steps = 0
    while(True):
        steps = steps + 1
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
    #keys = np.argmin(result, axis = 0)
    error = math.sqrt(sum(np.min(result,axis = 1))/num_pts)
    result = np.zeros((2))
    result[0] = error
    result[1] = steps
    return result

fo = open('cifar-10-batches-py/data_batch_1', 'rb')
dict = cPickle.load(fo)
fo.close()


score = kmeans(dict['data'], 10)
x = np.zeros((len(score)))
for i in range(len(score)):
    x[i] = i
    print i

# Create the plot
plt.plot(x,score,'ro')

# Save the figure in a separate file
plt.savefig('kmeans_1_run.png')
