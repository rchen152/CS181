from warm_up_plus import kmeans_plus

def unpickle(file):
    import cPickle
    fo = open(file, 'rb')
    dict = cPickle.load(fo)
    fo.close()
    return dict['data']

kmeans_plus(unpickle('/home/aaron/Downloads/cifar-10-batches-py/data_batch_1'),5)
