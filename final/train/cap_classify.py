import cPickle as pickle
import numpy as np
from scipy.cluster.vq import kmeans, vq
import matplotlib.pyplot as plt

fPCA = open('pca.pickle', 'r')
pca = pickle.load(fPCA)
fPCA.close()

fKMeans = open('kmeans_centers.pickle', 'r')
centers = pickle.load(fKMeans)
fKMeans.close()

def classify(data):
    cats,_ = vq(data,centers)
    return [1 if cat==0 else 0 for cat in cats]

good_data = pca.transform(np.loadtxt('../data/good_cap.csv'))
print 'All good capsules:'
print classify(good_data)

mixed_data = pca.transform(np.loadtxt('../data/validate_cap.csv'))
print '\nMix of capsules:'
print classify(mixed_data)
