import csv
import extract
import numpy as np
import pickle
from sklearn import linear_model as linmod
from sklearn import tree
import util

PROPS = [3.69,1.62,1.20,1.03,1.33,1.26,1.72,1.33,52.14,0.68,17.56,1.04,12.18,
         1.91,1.30]

log_raw = np.loadtxt('predictions/logistic-4.csv', dtype=str, delimiter=';')
log_reader = csv.reader(log_raw, delimiter=',')
log_reader.next()
log_preds = {}
for row in log_reader:
    log_preds[row[0]] = int(row[1])

tree_raw = np.loadtxt('predictions/syscall_count_by_type-1.csv', dtype=str,
                        delimiter=';')
tree_reader = csv.reader(tree_raw, delimiter=',')
tree_reader.next()
preds = []
ids = []
for row in tree_reader:
    f_id = row[0]
    tree_pred = int(row[1])
    if PROPS[tree_pred] < PROPS[log_preds[f_id]]:
        preds.append(tree_pred)
    else:
        preds.append(log_preds[f_id])
    ids.append(f_id)

util.write_predictions(preds, ids, 'predictions/combined-2.csv')

'''mat,_,cat = pickle.load(open('matrix_train', 'rb'))
mats,cats = extract.split_data(mat,cat,7)
correct = 0.
for i in range(7):
    train_mats = [mats[j] for j in range(7) if not i == j]
    train_cats = [cats[j] for j in range(7) if not i == j]
    train_mat, train_cat = extract.join_data(train_mats, train_cats)
    test_mat, test_cat = mats[i], cats[i]

    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(train_mat, train_cat)
    tree_preds = clf.predict(test_mat)

    logreg = linmod.LogisticRegression()
    logreg = logreg.fit(train_mat, train_cat)
    log_preds = logreg.predict(test_mat)

    for i in range(len(tree_preds)):
        pred = log_preds[i]
        if PROPS[tree_preds[i]] <= PROPS[log_preds[i]]:
            pred = tree_preds[i]
        if pred == test_cat[i]:
            correct += 1.
print correct / len(cat)'''
