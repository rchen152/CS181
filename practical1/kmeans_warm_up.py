import numpy as np
import util
import random
import cPickle
import math
from PIL import Image

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

#matrix as list of column vector
def get_distances(k, num_pts, mat, mu):
    fst = np.ones((num_pts,k))*np.sum(mat * mat, axis = 1).reshape(num_pts,1)
    product = np.dot(mat, np.transpose(mu))
    snd = np.ones((num_pts,k))*np.sum(mu*mu, axis = 1)
    result = fst - 2*product + snd
    return result

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
    keys = np.zeros((k, 1))        
    while(True):
        for i in range(k):
    	    mu[i] = np.mean(mat[resp[:,0]==i,:],axis=0)
        fst = np.ones((num_pts,k))*np.sum(mat*mat, axis = 1).reshape(num_pts,1)
	product = np.dot(mat, np.transpose(mu))
	snd = np.ones((num_pts,k))*np.sum(mu*mu, axis = 1)
	result = fst - 2*product + snd
	temp_resp = np.argmin(result,axis = 1).reshape(num_pts,1)
	print math.sqrt(sum(np.min(result,axis = 1))/num_pts)
        if(temp_resp == resp).all():
	     break
        resp = temp_resp
    keys = np.argmin(result, axis = 0)
    error = math.sqrt(sum(np.min(result,axis = 1))/num_pts)
    for i in range(k):
        get_image(mu[i], str(i) + ".png")
    return error

fo = open('cifar-10-batches-py/data_batch_1', 'rb')
dict = cPickle.load(fo)
fo.close()
print kmeans(dict['data'], 5)
