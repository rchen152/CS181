import numpy as np
import util

USER_TO_MEAN = 5.
BOOK_TO_MEAN = .7
 
# This makes predictions based on the mean rating for each book in the
# training data.  When there are no training data for a book, it
# defaults to the global mean.

pred_filename  = 'pred-weighted-book-user-mean9.csv'
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
print "The global mean rating is %0.3f." % (mean_rating)

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

'''dist = np.zeros((500))
for book in book_list:
    dist[books[book['isbn']]['count']] += 1
for i in range(500):
    print(str(i) + " " + str(dist[i]))'''

# Turn the list of users into a dictionary.
# Store data for each user to keep track of the per-user average.
users = {}
for user in user_list:
    users[user['user']] = { 'total': 0, # For storing the total of ratings.
                            'count': 0, # For storing the number of ratings.
                            }
    
# Iterate over the training data to compute means.
for rating in training_data:
    user_id = rating['user']
    users[user_id]['total'] += rating['rating']
    users[user_id]['count'] += 1

'''dist = np.zeros((5000))
for user in user_list:
    dist[users[user['user']]['count']] += 1
for i in range(5000):
    print(str(i) + " " + str(dist[i]))'''

# Make predictions for each test query.
for query in test_queries:

    book = books[query['isbn']]

    user = users[query['user']]
    if (user['count'] != 0):
        user_avg = float(user['total']) / user['count']
    if (book['count'] != 0):
        book_avg = float(book['total']) / book['count']    
    
    if (user['count'] == 0) and (book['count']==0):
        query['rating'] = mean_rating
    elif (user['count'] != 0) and (book['count']==0):
        query['rating'] = (USER_TO_MEAN*user_avg + mean_rating)/(USER_TO_MEAN+1)
    elif (user['count'] == 0) and (book['count']!=0):
        query['rating'] = (BOOK_TO_MEAN*book_avg + mean_rating)/(BOOK_TO_MEAN+1)
    else:
        query['rating'] = (BOOK_TO_MEAN*book_avg + USER_TO_MEAN*user_avg + mean_rating)/(BOOK_TO_MEAN+USER_TO_MEAN+1)


# Write the prediction file.
util.write_predictions(test_queries, pred_filename)
