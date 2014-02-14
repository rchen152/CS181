import numpy as np
import util
from sklearn.cluster import Ward
import kmeans_plus
import random
import math

NUM_CLUSTERS = 8

# This makes predictions based on the mean rating for each book in the
# training data.  When there are no training data for a book, it
# defaults to the global mean.

pred_filename  = 'predictor_age-kmeans1.csv'
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

num_users = len(user_list)
mat = np.zeros((num_users,1))
for i in range(num_users):
    mat[i][0] = user_list[i]['age']

[_,resp] = kmeans_plus.kmeans_plus(mat, NUM_CLUSTERS);

users = {}
clusters = []
for i in range(NUM_CLUSTERS):
    clusters.append(set())
for i in range(num_users):
    clusters[resp[i]].add(user_list[i]['user'])
    users[user_list[i]['user']] = {'index':resp[i],'total':0.,'count':0}

'''users = {}
clusters = []
for i in range(5):
    clusters.append(set())
for i in range(num_users):
    cluster = 0
    if (user_list[i]['age'] > 0):
        if (user_list[i]['age'] <= 15):
            cluster = 1
        else:
            if (user_list[i]['age'] <= 30):
                cluster = 2
            else:
                if (user_list[i]['age'] <= 50):
                    cluster = 3
                else:
                    cluster = 4
    clusters[cluster].add(user_list[i]['user'])
    users[user_list[i]['user']] = {'index':cluster, 'total':0., 'count':0}'''

for rate in train_data:
    users[rate['user']]['total'] += rate['rating']
    users[rate['user']]['count'] += 1

'''index = 0
b_keys = {}
for book in book_list:
    b_keys[book['isbn']] = index
    index += 1

train_sorted = []
num_books = len(book_list)
for i in range(num_books):
    train_sorted.append(set())

for rating in train_data:
    train_sorted[b_keys[rating['isbn']]].add((rating['user'],rating['rating']))'''

cluster_avgs = []
for i in range(NUM_CLUSTERS):
    sum_ratings = 0.
    num_ratings = 0
    for user in clusters[i]:
        sum_ratings += users[user]['total']
        num_ratings += users[user]['count']
    cluster_avgs.append(sum_ratings/num_ratings)
sum_errors = 0.
for datum in test_data:
    '''cluster = clusters[users[datum['user']]['index']]
    sum_ratings = 0.
    num_ratings = 0
    for (u,r) in train_sorted[b_keys[datum['isbn']]]:
        if (u in cluster):
            sum_ratings += r
            num_ratings += 1
    if (num_ratings == 0):
        sum_errors += math.pow(mean_rating - datum['rating'],2)
    else:
        sum_errors += math.pow((sum_ratings/num_ratings)-datum['rating'],2)'''
    sum_errors += math.pow(cluster_avgs[users[datum['user']]['index']] - datum['rating'], 2)
print math.sqrt(sum_errors / len(test_data))

'''for query in test_queries:
    cluster = clusters[users[query['user']]]
    sum_ratings = 0.
    num_ratings = 0
    for (u,r) in train_sorted[b_keys[query['isbn']]]:
        if (u in cluster):
            sum_ratings += r
            num_ratings += 1
    if (num_ratings == 0):
        query['rating'] = mean_rating
    else:
        query['rating'] = sum_ratings / num_ratings

# Write the prediction file.
util.write_predictions(test_queries, pred_filename)'''
