import cPickle as pickle
from scipy.cluster.vq import kmeans, vq

fPCA = open('pca.pickle', 'r')
pca = pickle.load(fPCA)
fPCA.close()

fKMeans = open('kmeans_centers.pickle', 'r')
centers = pickle.load(fKMeans)
fKMeans.close()

fTree = open('pickled_tree.p', 'r')
tree = pickle.load(fTree)
fTree.close()

def cap_classify(data):
    cats,_ = vq(data,centers)
    return [1 if cat==2 else 0 for cat in cats]
