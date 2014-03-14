import classification_starter as classify
from collections import Counter
import math
import matplotlib.pyplot as plt
import numpy as np
import pickle
import pydot
import random
from sklearn import tree
import StringIO
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
    mat,key,cats = pickle.load(open('matrix_train', 'rb'))
    test_mat,ids = pickle.load(open('matrix_test', 'rb'))
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(mat,cats)
    util.write_predictions(clf.predict(test_mat),ids,
                           'syscall_count_by_type-3.csv')

def syscall_means_and_vars():
    mat,key,cats = pickle.load(open('matrix_train', 'rb'))
    prop_mat = np.zeros((NUM_MALWARE, mat.shape[1]))
    cat_counts = np.zeros((NUM_MALWARE))
    for i in range(mat.shape[0]):
        prop_mat[cats[i]] += mat[i]
        cat_counts[cats[i]] += 1
    for i in range(NUM_MALWARE):
        prop_mat[i] /= cat_counts[i]
    var_mat = np.zeros((NUM_MALWARE, mat.shape[1]))
    for i in range(mat.shape[0]):
        diff = mat[i] - prop_mat[cats[i]]
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

def split_data(mat,cat,split=7):
    mats = [[] for i in range(split)]
    cats = [[] for i in range(split)]
    for i in range(mat.shape[0]):
        ind = random.randint(0,split-1)
        mats[ind].append(mat[i])
        cats[ind].append(cat[i])
    mats = [np.array(mat) for mat in mats]
    cats = [np.array(cat) for cat in cats]
    return (mats,cats)

def join_data(mats,cats):
    mat = []
    cat = []
    for i in range(len(mats)):
        for j in range(mats[i].shape[0]):
            mat.append(mats[i][j])
            cat.append(cats[i][j])
    return (np.array(mat),np.array(cat))

def test_depth(mat,cat,depth,split=7):
    mats,cats = split_data(mat,cat,split)
    correct = 0.
    for i in range(split):
        train_mats = [mats[j] for j in range(split) if not i == j]
        train_cats = [cats[j] for j in range(split) if not i == j]
        train_mat, train_cat = join_data(train_mats, train_cats)
        test_mat, test_cat = mats[i], cats[i]
        clf = tree.DecisionTreeClassifier(max_depth = depth)
        clf = clf.fit(train_mat,train_cat)
        pred_cat = clf.predict(test_mat)
        for i in range(len(test_cat)):
            if test_cat[i] == pred_cat[i]:
                correct += 1.
    return correct / len(cat)

def plot_cross_validation():
    mat,_,cat = pickle.load(open('matrix_train', 'rb'))
    xs = range(1,50)
    ys = []
    for d in xs:
        score = test_depth(mat, cat, depth = d, split = 7)
        print score
        ys.append(score)
    plt.scatter(xs,ys)
    plt.show()

def pickle_syscalls():
    mat,key,cats,_   = classify.extract_feats([syscalls], 'train')
    mat = np.asarray(mat.todense())
    matrix_train = open('matrix_train', 'wb')
    pickle.dump((mat,key,cats),matrix_train)

    test_mat,_,_,ids = classify.extract_feats([syscalls], direc='test',
                                              global_feat_dict = key)
    test_mat = np.asarray(test_mat.todense())
    matrix_test = open('matrix_test', 'wb')
    pickle.dump((test_mat,ids),matrix_test)
