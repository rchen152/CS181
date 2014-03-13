import classification_starter as classify
from collections import Counter
import math
import matplotlib.pyplot as plt
import numpy as np
from sklearn import tree
import StringIO
from sklearn.externals.six import StringIO
import pydot
import util

NUM_MALWARE = 15

def syscalls(tree):
    calls = {}
    for sec in tree.iter('all_section'):
        for call in sec.iter():
            if call.tag == 'all_section':
                continue
            elif call.tag not in calls:
                calls[call.tag] = 1.
            else:
                calls[call.tag] += 1.
    return calls

def syscall_count_by_type():
    mat,key,cats,_ = classify.extract_feats([syscalls], 'train')
    mat = np.asarray(mat.todense())
    test_mat,_,_,ids = classify.extract_feats([syscalls], direc='test',
                                              global_feat_dict = key)
    test_mat = np.asarray(test_mat.todense())
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(mat,cats)
    util.write_predictions(clf.predict(test_mat),ids,
                           'syscall_count_by_type-3.csv')

def syscall_means_and_vars():
    mat,key,cats,ids = classify.extract_feats([syscalls], 'train')
    prop_mat = np.zeros((NUM_MALWARE, mat.shape[1]))
    cat_counts = np.zeros((NUM_MALWARE))
    for i in range(mat.shape[0]):
        prop_mat[cats[i]] += np.asarray(mat[i].todense())[0]
        cat_counts[cats[i]] += 1
    for i in range(NUM_MALWARE):
        prop_mat[i] /= cat_counts[i]
    var_mat = np.zeros((NUM_MALWARE, mat.shape[1]))
    for i in range(mat.shape[0]):
        diff = np.asarray(mat[i].todense())[0] - prop_mat[cats[i]]
        var_mat[cats[i]] += diff * diff
    for i in range(NUM_MALWARE):
        var_mat[i] /= (cat_counts[i] - 1)
    return (prop_mat, var_mat)

def print_means_and_vars(prop_mat, var_mat):
    for i in range(prop_mat.shape[0]):
        out = ''
        for j in range(prop_mat.shape[1]):
            out += str(prop_mat[i,j]) + '\t' + str(var_mat[i,j]) + '\t\t'
        print out[:-2]

def filter(prop_mat, var_mat, m = 1., n = 1.):
    keep = np.zeros(prop_mat.shape)
    for i in range(prop_mat.shape[0]):
        for j in range(prop_mat.shape[1]):
            mean_diff = math.fabs(prop_mat[i,j] - prop_mat[8,j])
            sd_sum = (m*math.sqrt(var_mat[i,j])) + (n*math.sqrt(var_mat[8,j]))
            if mean_diff > sd_sum:
                keep[i,j] = 1
    return keep

prop_mat, var_mat = syscall_means_and_vars()
keep = filter(prop_mat, var_mat)
keep_cols = keep.sum(axis = 0)
for i in range(len(keep_cols)):
    if keep_cols[i] > 0:
        print i
