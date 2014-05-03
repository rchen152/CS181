import cPickle as pickle
from scipy.cluster.vq import kmeans, vq

fPCA = open('../train/pca.pickle', 'r')
pca = pickle.load(fPCA)
fPCA.close()

fKMeans = open('../train/kmeans_centers.pickle', 'r')
centers = pickle.load(fKMeans)
fKMeans.close()

'''fTree = open('tree.pickle', 'r')
tree = pickle.load(fTree)
fTree.close()
'''
def cap_classify(data):
    cats,_ = vq(data,centers)
    return [1 if cat==2 else 0 for cat in cats]
