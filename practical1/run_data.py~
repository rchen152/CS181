from kmeans_warm_up import kmeans

def unpickle(file):
    import cPickle
    fo = open(file, 'rb')
    dict = cPickle.load(fo)
    fo.close()
    return dict['data']

kmeans(unpickle('/home/aaron/Downloads/cifar-10-batches-py/data_batch_1'),5)
