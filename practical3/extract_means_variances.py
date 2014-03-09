import classification_starter as classify
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
import math

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

def get_stats(key_name):
    mat,key,cats,ids = classify.extract_feats([structure], 'train')
    mat_len = mat.shape[0]
    sums =  np.zeros((NUM_MALEWARE))
    counts =  np.zeros((NUM_MALEWARE))
    means = np.zeros((NUM_MALEWARE))
    var = np.zeros((NUM_MALEWARE))
    for i in range(mat_len):
        sums[cats[i]] += mat[i,key[key_name]]
        counts[cats[i]] += 1
        for i in range(NUM_MALEWARE):
            means[i] = sums[i]/counts[i]
    for i in range(mat_len):
        var[cats[i]] += (mat[i,key[key_name]] - means[cats[i]])**2

    for i in range(NUM_MALEWARE):
        var[i] = var[i]/counts[i]
    std = np.zeros((NUM_MALEWARE))
    std = map(math.sqrt,var)
    return (means,var,std)

print get_stats('num_processes')



'''mat,key,cats,ids = classify.extract_feats([syscalls], 'train')
counts = np.asarray(mat.sum(axis=0))[0]
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
    print out[:-1]'''
