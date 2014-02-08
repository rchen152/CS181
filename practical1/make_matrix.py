from scipy.sparse import coo_matrix
import numpy as np
import util

pred_filename  = 'pred-weighted-book-user-mean6.csv'
train_filename = 'ratings-train.csv'
test_filename  = 'ratings-test.csv'
book_filename  = 'books.csv'
user_filename  = 'users.csv'

training_data  = util.load_train(train_filename)
test_queries   = util.load_test(test_filename)
book_list      = util.load_books(book_filename)
user_list      = util.load_users(user_filename)

num_train = len(training_data)
mean_rating = float(sum(map(lambda x: x['rating'], training_data)))/num_train

data = np.zeros((num_train))
book_num  = np.zeros((num_train))
user_id = np.zeros((num_train))

books = {}
index = 0
for book in book_list:
    books[book['isbn']] = index
    index = index + 1
    
for i in range (num_train):
    data[i] = training_data[i]['rating']
    book_num[i] = books[training_data[i]['isbn']]
    user_id[i] = training_data[i]['user']

max_user = 0
for user in user_list:
    if (max_user < user['user']):
        max_user = user['user']

mat = coo_matrix((data,(book_num,user_id)),shape=(len(book_list),max_user+1))
