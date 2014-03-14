import pickle
from sklearn import tree
import util

matrix_train = open('matrix_train', 'rb')
mat,key,cats = pickle.load(matrix_train)

matrix_test = open('matrix_test', 'rb')
test_mat,ids = pickle.load(matrix_test)
clfr = tree.DecisionTreeClassifier()
clfr = clfr.fit(mat,cats)
imp = clfr.feature_importances_
index =0
index_imp =[]
for i in range(len(imp)):
    index_imp.append((imp[i],i))
index_imp.sort(key=lambda tup:tup[0])

inds = map(lambda x: x[1],index_imp)
l_aar = inds[-45:]
l_reb = [7,10,11,13,17,18,20,25,26,32,33,40,52,56,66,72,73,90,93,96,98,100]


best_list_union = list(set(l_aar +l_reb))
best_list_int= list(set(l_aar).intersection(set(l_reb)))
print len(best_list_union)
print len(l_aar)
print len(l_reb)
print len(best_list_int)


reduced_train = mat[:,best_list_union]
reduced_test = test_mat[:,best_list_union]

clf = tree.DecisionTreeClassifier()
clf = clf.fit(reduced_train,cats)
util.write_predictions(clf.predict(reduced_test),ids,'top_list-union-2.csv')
