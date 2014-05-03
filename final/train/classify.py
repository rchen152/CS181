import cPickle as pickle
import numpy as np
from scipy.cluster.vq import kmeans, vq

fPCA = open('../train/pca.pickle', 'r')
pca = pickle.load(fPCA)
fPCA.close()

fKMeans = open('../train/kmeans_centers.pickle', 'r')
centers = pickle.load(fKMeans)
fKMeans.close()

good_data = pca.transform(np.loadtxt('../data/good_cap.csv'))
good_pt = good_data[0].reshape((1,good_data.shape[1]))
cats,_ = vq(good_pt,centers)
good_index = cats[0]

'''fTree = open('tree.pickle', 'r')
tree = pickle.load(fTree)
fTree.close()
'''
def cap_classify(data):
    data = pca.transform(data)
    cats,_ = vq(data,centers)
    return [1 if cat==good_index else 0 for cat in cats]
