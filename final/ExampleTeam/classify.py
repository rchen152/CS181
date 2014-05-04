import cPickle as pickle
import numpy as np
from scipy.cluster.vq import kmeans, vq

# TODO change the path to the pickled decision tree
fTree = open('train/pickled_tree_101_200.p', 'r')
tree = pickle.load(fTree)
fTree.close()
GOOD_FEATS = [0,1,2,3,4,10,11,12]

fPCA = open('train/pca.pkl', 'r')
pca = pickle.load(fPCA)
fPCA.close()

fKMeans = open('train/kmeans_centers.pkl', 'r')
centers = pickle.load(fKMeans)
fKMeans.close()

good_data = pca.transform(np.loadtxt('data/good_cap.csv'))
good_pt = good_data[0].reshape((1,good_data.shape[1]))
cats,_ = vq(good_pt,centers)
good_index = cats[0]

def capClassify(data):
    data = pca.transform(data)
    cats,_ = vq(data,centers)
    return [1 if cat==good_index else 0 for cat in cats]

def ghostClassify(data):
    return [int(pred) for pred in tree.predict([d[GOOD_FEATS] for d in data])]
