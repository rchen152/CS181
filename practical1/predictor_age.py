import numpy as np
import util
from sklearn.cluster import Ward
import kmeans_plus

# CRITICAL_BOOK_NUM = 4
# TODO change this value depending on what your system can handle
BOOK_MEMORY_ERROR = 7
NUM_CLUSTERS = 10

def 

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

num_users = len(user_list)
mat = np.zeros((num_users, 1))

# Turn the list of users into a dictionary.
# Store data for each user to keep track of the per-user average.
users = {}
for i in range(num_users):
    users[user_list[i]['user']] = {'total': 0, 'count': 0}
    mat[i][0] = user_list[i]['age']
    
# Iterate over the training data to compute means.
for rating in training_data:
    user_id = rating['user']
    users[user_id]['total'] += rating['rating']
    users[user_id]['count'] += 1

[mu,resp] = kmeans_plus.kmeans_plus(mat, NUM_CLUSTERS)

for query in test_queries:
    query['rating'] = mean_rating

# Write the prediction file.
util.write_predictions(test_queries, pred_filename)
