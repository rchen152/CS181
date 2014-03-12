import classification_starter as classify
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
from sklearn import tree
import util

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

mat,key,cats,_   = classify.extract_feats([syscalls], 'train')
mat = np.asarray(mat)
test_mat,_,_,ids = classify.extract_feats([syscalls], direc='test',
                                          global_feat_dict = key)
test_mat = np.asarray(test_mat)
preds = []
clf = tree.DecisionTreeClassifier()
clf = clf.fit(mat,cats)
for i in range(test_mat.shape[0]):
    preds.append(clf.predict(test_mat[i]))
util.write_predictions(preds,ids,'syscall_count_by_type-1.csv')

'''counts = np.asarray(mat.sum(axis=0))[0]
prop_mat = np.zeros((16, mat.shape[1]))
prop_mat[prop_mat.shape[0]-1] = counts
cat_counts = np.zeros((16))
cat_counts[15] = mat.shape[0]
for i in range(mat.shape[0]):
    prop_mat[cats[i]] += mat[i]
    cat_counts[cats[i]] += 1
for i in range(prop_mat.shape[0]):
    prop_mat[i] /= cat_counts[i]
    out = ''
    for x in prop_mat[i]:
        out += str(x) + '\t'
    print out[:-1]

calltype_counts = np.zeros((prop_mat.shape[1]))
theones = np.ones((prop_mat.shape[1]))
sq_sum = np.zeros((prop_mat.shape[1]))
for i in range(mat.shape[0]):
    if cats[i] == 8:
        calltype_counts += theones
        diff = np.asarray(mat[i] - prop_mat[8])[0]
        sq_sum += diff * diff
sd = sq_sum / (calltype_counts - 1)
sd_out = ''
for i in range(prop_mat.shape[1]):
    sd_out += str(sd[i]) + '\t'
print sd_out[:-1]'''
