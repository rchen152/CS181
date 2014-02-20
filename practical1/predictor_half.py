import numpy as np
import util
from sklearn.cluster import Ward
import kmeans_plus
import random
import math

# CRITICAL_BOOK_NUM = 4
# TODO change this value depending on what your system can handle
BOOK_MEMORY_ERROR = 8
NUM_CLUSTERS = 8

# This makes predictions based on the mean rating for each book in the
# training data.  When there are no training data for a book, it
# defaults to the global mean.

pred_filename  = 'predictor-kmeans10.csv'
train_filename = 'ratings-train.csv'
test_filename  = 'ratings-test.csv'
book_filename  = 'books.csv'
user_filename  = 'users.csv'

training_data  = util.load_train(train_filename)
test_queries   = util.load_test(test_filename)
book_list      = util.load_books(book_filename)
user_list      = util.load_users(user_filename)

train_data = []
test_data = []

for datum in training_data:
    if (random.randrange(2) == 0):
        train_data.append(datum)
    else:
        test_data.append(datum)

# Compute the global mean rating for a fallback.
num_train = len(train_data)
mean_rating = float(sum(map(lambda x: x['rating'], train_data)))/num_train

# Turn the list of books into an ISBN-keyed dictionary.
# Store data for each book to keep track of the per-book average.
books = {}
for book in book_list:
    books[book['isbn']] = { 'total': 0, # For storing the total of ratings.
                            'count': 0, # For storing the number of ratings.
                            }

# Iterate over the training data to compute means.
for rating in train_data:
    books[rating['isbn']]['total'] += rating['rating']
    books[rating['isbn']]['count'] += 1

book_short = [book for book in book_list if books[book['isbn']]['count'] > BOOK_MEMORY_ERROR]

train_short = [rating for rating in train_data if books[rating['isbn']]['count'] > BOOK_MEMORY_ERROR]

# Turn the list of users into a dictionary.
# Store data for each user to keep track of the per-user average.
users_shortsum = {}
for user in user_list:
    users_shortsum[user['user']] = {'total': 0, 'count': 0}
    
# Iterate over the training data to compute means.
for rating in train_short:
    user_id = rating['user']
    users_shortsum[user_id]['total'] += rating['rating']
    users_shortsum[user_id]['count'] += 1

user_short = [user for user in user_list if users_shortsum[user['user']]['count'] > 0]
len_user_short = len(user_short)

book_keys = {}
index = 0
for book in book_short:
    book_keys[book['isbn']] = index
    index += 1

user_keys = {}
index = 0
for user in user_short:
    user_keys[user['user']] = index
    index += 1

mat = np.zeros((len_user_short, len(book_short))).astype(float)
for rating in train_short:
    user = users_shortsum[rating['user']]
    mat[user_keys[rating['user']]][book_keys[rating['isbn']]] = rating['rating'] - float(user['total']) / user['count']

[_,resp] = kmeans_plus.kmeans_plus(mat, NUM_CLUSTERS)

users_sum = {}
clusters = []
for i in range(NUM_CLUSTERS):
    clusters.append(set())
for i in range(len_user_short):
    user = user_list[i]['user']
    clusters[resp[i]].add(user)
    users_sum[user] = {'cluster' : resp[i], 'total' : 0., 'count' : 0}

for rate in train_data:
    if (rate['user'] in users_sum):
        users_sum[rate['user']]['total'] += rate['rating']
        users_sum[rate['user']]['count'] += 1

cluster_avgs = []
for i in range(NUM_CLUSTERS):
    sum_ratings = 0.
    num_ratings = 0.
    for user in clusters[i]:
        sum_ratings += users_sum[user]['total']
        num_ratings += users_sum[user]['count']
    cluster_avgs.append(sum_ratings/num_ratings)

sum_errors = 0.
for datum in test_data:
    if (datum['user'] in users_sum):
        sum_errors += math.pow(cluster_avgs[users_sum[datum['user']]['cluster']] - datum['rating'], 2)
    else:
        sum_errors += math.pow(mean_rating - datum['rating'], 2)
print math.sqrt(sum_errors / len(test_data))

'''[mu,resp] = kmeans_plus.kmeans_plus(mat, NUM_CLUSTERS)

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

for rating in train_data:
    training_sorted[long_book_keys[rating['isbn']]].add((rating['user'],rating['rating']))

sum_errors = 0.
for query in test_data:
    user = users[query['user']]
    if (query['user'] in user_keys):
        cluster = cluster_ids[resp[user_keys[query['user']]]]
        sum_zetas = 0
        num_zetas = 0
        for (use,rate) in training_sorted[long_book_keys[query['isbn']]]:
            if (use in cluster):
                sum_zetas += rate - float(users[use]['total'])/users[use]['count']
                num_zetas += 1
        if (num_zetas == 0):
            sum_errors += math.pow(mean_rating - query['rating'],2)
        else:
            temp_rating = float(sum_zetas)/num_zetas + float(user['total'])/user['count']
            if(temp_rating > 5):
                temp_rating = 5
            if(temp_rating < 1):
                temp_rating = 1
            sum_errors += math.pow(temp_rating - query['rating'],2)
    else:
        sum_errors += math.pow(mean_rating - query['rating'],2)

print math.sqrt(sum_errors / len(test_data))'''
