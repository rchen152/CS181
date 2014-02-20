import numpy as np
import cPickle
from PIL import Image
import os
import kmeans_warm_up

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

# performs kmeans clustering on a batch of cifar images
def cluster_images(data_batch, k, smart_init = False):
    fo = open(data_batch, 'rb')
    dict = cPickle.load(fo)
    fo.close()

    m = dict['data']
    kmeans_out = kmeans_warm_up.kmeans(m, k, smart_init)
    error = kmeans_out['error']
    mu = kmeans_out['mu']
    resp = kmeans_out['resp']
    result = kmeans_out['dist']

    # creates cluster directories and saves images
    for i in range(k):
        if not os.path.exists(str(i)):
            os.makedirs(str(i))
        get_image(mu[i], str(i) + "/0.png")
    num_pts = m.shape[0]
    for i in range(num_pts):
        get_image(m[i], str(resp[i][0]) + "/" + str(np.min(result[i])) + ".png")
    return error
