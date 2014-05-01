from sklearn import tree
import numpy as np
import scipy as sp
import csv
import pickle

TRAIN_NAME = "../data/master_valid_ghost.csv"
TEST_NAME = "../data/master_ghost.csv"

train_matrix = np.loadtxt(TRAIN_NAME)
test_matrix = np.loadtxt(TEST_NAME)

cats = train_matrix[:,1]
mat = train_matrix[:,3:]
test = test_matrix[:,3:]

clf = tree.DecisionTreeClassifier()
clf = clf.fit(mat,cats)

#pickle.dump(clf,open("pickled_tree_101_200.p","w"))
preds = clf.predict(test)
print preds
correct_preds = test_matrix[:,1]
print correct_preds
lst = [preds[i] for i in range(len(preds)) if correct_preds[i]==preds[i]]

for i in range(6):
    print len(filter(lambda x: x ==float(i),filter(correct_preds[i]==preds[i],preds)))
for i in range(6):
    print len(filter(lambda x: x ==float(i),correct_preds))
print len(lst)
print sum(lst)
pred2 = clf.predict(mat)
