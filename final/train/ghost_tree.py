from sklearn import tree
import numpy as np
import scipy as sp
import csv

TRAIN_NAME = "../data/train_ghost.csv"
TEST_NAME = "../data/validate_ghost.csv"

raw_data = np.loadtxt(TRAIN_NAME, dtype=str, delimiter=';')
reader = csv.reader(raw_data, delimiter=' ')
train_lst = []
for row in reader:
    train_lst.append(row)
train_matrix = np.array(train_lst)

raw_data = np.loadtxt(TEST_NAME, dtype=str, delimiter=';')
reader = csv.reader(raw_data, delimiter=' ')
test_lst = []
for row in reader:
    test_lst.append(row)
test_matrix = np.array(test_lst)

cats = train_matrix[:1000,1]
mat = train_matrix[:1000,:3]
test = test_matrix[:,:3]

clf = tree.DecisionTreeClassifier()
clf = clf.fit(mat,cats)
preds = clf.predict(test)
print preds
correct_preds = test_matrix[:,1]
print correct_preds
lst = [correct_preds[i]==preds[i] for i in range(len(preds))]
for i in range(6):
    print len(filter(lambda x: x ==str(float(i)),correct_preds))
print len(lst)
print sum(lst)
pred2 = clf.predict(mat)
