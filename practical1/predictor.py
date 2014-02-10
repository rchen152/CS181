import numpy as np
import util
from sklearn.cluster import Ward
import kmeans_plus

# CRITICAL_BOOK_NUM = 4
# TODO change this value depending on what your system can handle
BOOK_MEMORY_ERROR = 5
NUM_CLUSTERS = 15

# This makes predictions based on the mean rating for each book in the
# training data.  When there are no training data for a book, it
# defaults to the global mean.

pred_filename  = 'predictor-kmeans9.csv'
train_filename = 'ratings-train.csv'
test_filename  = 'ratings-test.csv'
book_filename  = 'books.csv'
user_filename  = 'users.csv'

training_data  = util.load_train(train_filename)
test_queries   = util.load_test(test_filename)
book_list      = util.load_books(book_filename)
user_list      = util.load_users(user_filename)

# Compute the global mean rating for a fallback.
num_train = len(training_data)
mean_rating = float(sum(map(lambda x: x['rating'], training_data)))/num_train

# Turn the list of books into an ISBN-keyed dictionary.
# Store data for each book to keep track of the per-book average.
books = {}
for book in book_list:
    books[book['isbn']] = { 'total': 0, # For storing the total of ratings.
                            'count': 0, # For storing the number of ratings.
                            }

# Iterate over the training data to compute means.
for rating in training_data:
    books[rating['isbn']]['total'] += rating['rating']
    books[rating['isbn']]['count'] += 1

book_short = [book for book in book_list if books[book['isbn']]['count'] > BOOK_MEMORY_ERROR]

train_short = [rating for rating in training_data if books[rating['isbn']]['count'] > BOOK_MEMORY_ERROR]

# Turn the list of users into a dictionary.
# Store data for each user to keep track of the per-user average.
users = {}
for user in user_list:
    users[user['user']] = { 'total': 0, # For storing the total of ratings.
                            'count': 0, # For storing the number of ratings.
                            }
    
# Iterate over the training data to compute means.
for rating in train_short:
    user_id = rating['user']
    users[user_id]['total'] += rating['rating']
    users[user_id]['count'] += 1

user_short = [user for user in user_list if users[user['user']]['count'] > 0]

num_books = len(book_short)
num_users = len(user_short)

book_keys = {}
index = 0
for book in book_short:
    book_keys[book['isbn']] = index
    index += 1

user_keys = {}
inv_user_keys = {}
index = 0
for user in user_short:
    user_keys[user['user']] = index
    inv_user_keys[index] = user['user']
    index += 1

mat = np.zeros((num_users, num_books)).astype(float)
# book_pref = 0
for rating in train_short:
    user = users[rating['user']]
    
    '''book = books[rating['isbn']]
    if (book['count'] > CRITICAL_BOOK_NUM):
        book_pref = float(book['total']) / book['count']    
    else:
        book_pref = mean_rating
    mat[user_keys[rating['user']]][book_keys[rating['isbn']]] = rating['rating'] - float(user['total']) / user['count'] + mean_rating - book_pref
    mat[user_keys[rating['user']]][book_keys[rating['isbn']]] = rating['rating'] - float(user['total']) / user['count']'''

    mat[user_keys[rating['user']]][book_keys[rating['isbn']]] = rating['rating'] - float(user['total']) / user['count']

[mu,resp] = kmeans_plus.kmeans_plus(mat, NUM_CLUSTERS)

cluster_ids = []
for i in range(NUM_CLUSTERS):
    cluster_ids.append(set())
for i in range(num_users):
    cluster_ids[resp[i]].add(inv_user_keys[i])
# Make predictions for each test query.

long_book_keys = {}
index = 0
for book in book_list:
    long_book_keys[book['isbn']] = index
    index += 1

training_sorted = []
for i in range(len(book_list)):
    training_sorted.append(set())

for rating in training_data:
    training_sorted[long_book_keys[rating['isbn']]].add((rating['user'],rating['rating']))

for query in test_queries:
    user = users[query['user']]
    # TODO global mean rating or mean rating on book?
    if(query['user'] in user_keys):
        cluster = cluster_ids[resp[user_keys[query['user']]]]
        sum_zetas = 0
        num_zetas = 0
        for (use,rate) in training_sorted[long_book_keys[query['isbn']]]:
            if (use in cluster):
                sum_zetas += rate - float(users[use]['total'])/users[use]['count']
                num_zetas += 1
        if (num_zetas == 0):
            query['rating'] = mean_rating
        else:
            temp_rating = float(sum_zetas)/num_zetas + float(user['total'])/user['count']
            if(temp_rating > 5):
                temp_rating = 5
            if(temp_rating < 1):
                temp_rating = 1
            query['rating'] = temp_rating
    else:
        query['rating'] = mean_rating

# Write the prediction file.
util.write_predictions(test_queries, pred_filename)
