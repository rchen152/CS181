import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from scipy.cluster.vq import kmeans, vq
import cPickle as pickle

train_data = np.loadtxt('../data/train_cap.csv')
pca = PCA(n_components = 2)
pca.fit(train_data)
all_data = pca.transform(train_data)
centers,_ = kmeans(all_data,3)

def pickle():
    fPCA = open('pca.pickle', 'w')
    pickle.dump(pca, fPCA)
    fPCA.close()

    fKMeans = open('kmeans_centers.pickle', 'w')
    pickle.dump(centers, fKMeans)
    fKMeans.close()

def plot_pca():
    

def plot_kmeans():
'''ids,_ = vq(all_data,centers)
clusters = [[],[],[]]
for i in range(len(ids)):
    clusters[ids[i]].append(all_data[i])

clusters = [np.array(clusters[i]) for i in range(len(clusters))]
dataplot = [(clusters[i][:,0], clusters[i][:,1]) for i in range(len(clusters))]
plt.plot(dataplot[0][0], dataplot[0][1], 'ro')
plt.plot(dataplot[1][0], dataplot[1][1], 'go')
plt.plot(dataplot[2][0], dataplot[2][1], 'bo')
plt.show()

good_data = pca.transform(np.loadtxt(open('../data/good_cap.csv')))
ids,_ = vq(good_data,centers)
for i in ids:
    print i'''

#x1 = all_data[:,0]
#y1 = all_data[:,1]
#x2 = good_data[:,0]
#y2 = good_data[:,1]
#plt.plot(x1,y1,'ro')
#plt.plot(x2,y2,'go')
#plt.show()
