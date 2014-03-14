import classification_starter as classify
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
from sklearn import tree
from sklearn.externals.six import StringIO
import pickle
import pydot
import util

NUM_MALEWARE = 15

def structure(tree):
    c = Counter()
    for node in tree.iter():
        if node.tag == 'processes':
            continue
        elif node.tag == 'process':
            c['num_processes'] += 1
        elif node.tag == 'thread':
            c['num_threads'] += 1
        elif node.tag == 'all_section':
            continue
        else:
            c['num_syscalls'] += 1
    return c

def syscalls(tree):
    bad_tags = ['processes', 'process', 'thread', 'all_section']
    calls = {}
    for node in tree.iter():
        if node.tag in bad_tags:
            continue
        if node.tag not in calls:
            calls[node.tag] = 1.
        else:
            calls[node.tag] += 1.
    return calls

def example_structure_plot():
    mat,key,cats,_ = classify.extract_feats([structure], 'train')
    for i in range(mat.shape[0]):
        color = 'red'
        if cats[i] == 8:
            color = 'black'
        plt.scatter([mat[i,key['num_processes']]], [cats[i]], c=color)
    plt.show()

'''mat,key,cats,_   = classify.extract_feats([syscalls], 'train')
mat = np.asarray(mat.todense())
matrix_train = open('matrix_train', 'wb')
pickle.dump((mat,key,cats),matrix_train)

test_mat,_,_,ids = classify.extract_feats([syscalls], direc='test',
                                          global_feat_dict = key)
test_mat = np.asarray(test_mat.todense())
matrix_test = open('matrix_test', 'wb')
pickle.dump((test_mat,ids),matrix_test)'''
