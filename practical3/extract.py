import classification_starter as classify
from collections import Counter
import matplotlib.pyplot as plt

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

mat,key,cats,_ = classify.extract_feats([structure], 'train')
for i in range(mat.shape[0]):
    color = 'red'
    if cats[i] == 8:
        color = 'black'
    plt.scatter([mat[i,key['num_processes']]], [cats[i]], c=color)
plt.show()
