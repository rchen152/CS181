from sklearn import tree
import numpy as np

TRAIN_NAME = "../data/train_ghost.csv"
TEST_NAME = "../data/validate_ghost.csv"

train_matrix = np.loadtxt(TRAIN_NAME)
test_matrix = np.loadtxt(TEST_NAME)

good_feats = [3,4,5,6,7,13,14,15]

cats = train_matrix[:,1]
mat = train_matrix[:,good_feats]
test = test_matrix[:,good_feats]

clf = tree.DecisionTreeClassifier()
clf = clf.fit(mat,cats)
preds = clf.predict(test)
correct_preds = test_matrix[:,1]

correct = 0
total = len(preds)
for i in range(total):
    if preds[i] == correct_preds[i]:
        correct += 1
print (correct/total)
