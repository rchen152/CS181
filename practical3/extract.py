import classification_starter as classify
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
from sklearn import tree
import StringIO
from sklearn.externals.six import StringIO
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
test_mat,_,_,ids = classify.extract_feats([syscalls], direc='test',
                                          global_feat_dict = key)
test_mat = np.asarray(test_mat.todense())
clf = tree.DecisionTreeClassifier()
clf = clf.fit(mat,cats)
util.write_predictions(clf.predict(test_mat),ids,'syscall_count_by_type-1.csv')

dot_data = StringIO.StringIO() 
tree.export_graphviz(clf, out_file=dot_data) 
graph = pydot.graph_from_dot_data(dot_data.getvalue()) 
graph.write_pgn("tree_some.pgn")
'''
