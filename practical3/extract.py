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

def ms(time):
    mins = time[:2]
    secs = time[3:5]
    millisecs = time[6:]
    return (int(mins) * 60000) + (int(secs) * 1000) + int(millisecs)

def structure(tree):
    c = Counter()
    c['num_processes'] = 0.
    c['num_threads'] = 0.
    c['num_syscalls'] = 0.
    c['avg_process_file_size'] = 0.
    c['avg_process_runtime'] = 0.
    c['max_num_threads_per_process'] = -1
    c['min_num_threads_per_process'] = 1000000
    c['max_num_syscalls_per_process'] = -1
    c['min_num_syscalls_per_process'] = 1000000
    c['max_num_syscalls_per_thread'] = -1
    c['min_num_syscalls_per_thread'] = 1000000

    thread_list = []
    calls_per_process_list = []
    calls_per_thread_list = []

    for p in tree.iter('process'):
        size = int(p.attrib['filesize'])
        if size < 0:
            size = 0
        c['avg_process_file_size'] += size
        time = ms(p.attrib['terminationtime']) - ms(p.attrib['starttime'])
        if time < 0:
            time = 0
        c['avg_process_runtime'] += time

        num_threads = len(p)
        num_syscalls = 0
        for sec in p.iter('all_section'):
            num_calls = 0
            for _ in sec.iter():
                num_calls += 1
            if c['max_num_syscalls_per_thread'] < num_calls:
                c['max_num_syscalls_per_thread'] = num_calls
            if c['min_num_syscalls_per_thread'] > num_calls:
                c['min_num_syscalls_per_thread'] = num_calls
            num_syscalls += num_calls
            calls_per_thread_list.append(num_calls)
        if c['max_num_threads_per_process'] < num_threads:
            c['max_num_threads_per_process'] = num_threads
        if c['min_num_threads_per_process'] > num_threads:
            c['min_num_threads_per_process'] = num_threads

        if c['max_num_syscalls_per_process'] < num_syscalls:
            c['max_num_syscalls_per_process'] = num_syscalls
        if c['min_num_syscalls_per_process'] > num_syscalls:
            c['min_num_syscalls_per_process'] = num_syscalls

        c['num_processes'] += 1.
        c['num_threads'] += num_threads
        c['num_syscalls'] += num_syscalls

        thread_list.append(num_threads)
        calls_per_process_list.append(num_syscalls)

    c['avg_process_file_size'] /= c['num_processes']
    c['avg_process_runtime'] /= c['num_processes']
    c['avg_num_threads_per_process'] = c['num_threads']/c['num_processes']
    c['avg_num_syscalls_per_process'] = c['num_syscalls']/c['num_processes']
    c['avg_num_syscalls_per_thread'] = c['num_syscalls']/c['num_threads']

    c['sd_num_threads_per_process'] = 0.
    c['sd_num_syscalls_per_process'] = 0.
    c['sd_num_syscalls_per_thread'] = 0.

    for t in thread_list:
        diff = t - c['avg_num_threads_per_process']
        c['sd_num_threads_per_process'] += diff * diff
    for sc in calls_per_process_list:
        diff = sc - c['avg_num_syscalls_per_process']
        c['sd_num_syscalls_per_process'] += diff * diff
    for sc in calls_per_thread_list:
        diff = sc - c['avg_num_syscalls_per_thread']
        c['sd_num_syscalls_per_thread'] += diff * diff

    c['sd_num_threads_per_process'] /= c['num_processes']
    c['sd_num_syscalls_per_process'] /= c['num_processes']
    c['sd_num_syscalls_per_thread'] /= c['num_threads']

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

def print_syscall_counts_by_type():
    mat,key,cats,ids = classify.extract_feats([syscalls], 'train')
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
    print out[:-1]

mat,key,cats,_ = classify.extract_feats([structure,syscalls], 'train')
mat = np.asarray(mat.todense())
test_mat,_,_,ids = classify.extract_feats([syscalls], direc='test',
                                          global_feat_dict = key)
test_mat = np.asarray(test_mat.todense())
clf = tree.DecisionTreeClassifier()
clf = clf.fit(mat,cats)

util.write_predictions(clf.predict(test_mat),ids,'everything-1.csv')'''

