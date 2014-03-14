import pickle
import sklearn.linear_model as sklm 
import util

matrix_train = open('matrix_train', 'rb')
mat,key,cats = pickle.load(matrix_train)

matrix_test = open('matrix_test', 'rb')
test_mat,ids = pickle.load(matrix_test)

#best_features = [25,26,33,96,100]
best_features = [7,10,11,13,17,18,20,25,26,32,33,40,52,56,66,72,73,90,93,96,98,100]

#reduced_train = mat[:,best_features]
#reduced_test = test_mat[:,best_features]
reduced_train = mat
reduced_test = test_mat

logreg = sklm.LogisticRegression()
logreg.fit(reduced_train,cats)
preds = logreg.predict(reduced_test)
util.write_predictions(preds,ids,'logistic-4.csv')
