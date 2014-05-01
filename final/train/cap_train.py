import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from scipy.cluster.vq import kmeans, vq
import cPickle as pickle

MIXED_CAPS = '../data/master_cap.csv'
GOOD_CAPS = '../data/good_cap.csv'

train_data = np.loadtxt(MIXED_CAPS)
pca = PCA(n_components = 2)
pca.fit(train_data)
all_data = pca.transform(train_data)
good_data = pca.transform(np.loadtxt(GOOD_CAPS))
centers,_ = kmeans(all_data,3)

def do_pickle():
    fPCA = open('pca.pickle', 'w')
    pickle.dump(pca, fPCA)
    fPCA.close()

    fKMeans = open('kmeans_centers.pickle', 'w')
    pickle.dump(centers, fKMeans)
    fKMeans.close()

def plot_pca():
    x1 = all_data[:,0]
    y1 = all_data[:,1]
    x2 = good_data[:,0]
    y2 = good_data[:,1]
    plt.plot(x1,y1,'ro')
    plt.plot(x2,y2,'go')
    plt.show()    

def plot_kmeans():
    ids,_ = vq(all_data,centers)
    clusters = [[],[],[]]
    for i in range(len(ids)):
        clusters[ids[i]].append(all_data[i])

    clusters = [np.array(clusters[i]) for i in range(len(clusters))]
    dataplot = [(clusters[i][:,0], clusters[i][:,1])
                for i in range(len(clusters))]
    plt.plot(dataplot[0][0], dataplot[0][1], 'ro')
    plt.plot(dataplot[1][0], dataplot[1][1], 'go')
    plt.plot(dataplot[2][0], dataplot[2][1], 'bo')
    plt.show()

