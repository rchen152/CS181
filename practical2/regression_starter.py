## This file provides starter code for extracting features from the xml files and
## for doing some learning.
##
## The basic set-up: 
## ----------------
## main() will run code to extract features, learn, and make predictions.
## 
## extract_feats() is called by main(), and it will iterate through the 
## train/test xml files and extract each instance into a util.MovieData object.
## It will then use a series of "feature-functions" that you will write/modify
## in order to extract dictionaries of features from each util.MovieData object.
## Finally, it will produce an N x D sparse design matrix containing the union
## of the features contained in the dictionaries produced by your "feature-functions."
## This matrix can then be plugged into your learning algorithm.
##
## The learning and prediction parts of main() are largely left to you, though
## it does contain code for naive linear regression and prediction so you
## have a sense of where/what to modify.
##
## Feature-functions:
## --------------------
## "feature-functions" are functions that take a util.MovieData object representing
## a single movie, and return a dictionary mapping feature names to their respective
## numeric values. 
## For instance, a simple feature-function might map a movie object to the
## dictionary {'company-Fox Searchlight Pictures': 1}. This is a boolean feature
## indicating whether the production company of this move is Fox Searchlight Pictures,
## but of course real-valued features can also be defined. Because this feature-function
## will be run over MovieData objects for each movie instance, we will have the (different)
## feature values of this feature for each movie, and these values will make up 
## one of the columns in our final design matrix.
## Of course, multiple features can be defined within a single dictionary, and in
## the end all the dictionaries returned by feature functions will be unioned
## so we can collect all the feature values associated with a particular instance.
##
## Two example feature-functions, metadata_feats() and unigram_feats() are defined
## below. These extract metadata and unigram text features, respectively.
##
## What you need to do:
## --------------------
## 1. Write new feature-functions (or modify the example feature-functions) to
## extract useful features for this prediction task.
## 2. Implement an algorithm to learn from the design matrix produced, and to
## make predictions on unseen data. Naive code for these two steps is provided
## below, and marked by TODOs.


from collections import Counter
try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET
import numpy as np
from scipy import sparse
from scipy.sparse import linalg as splinalg

import util


def extract_feats(ffs, datafile="train.xml", global_feat_dict=None):
    """
    arguments:
      ffs are a list of feature-functions.
      datafile is an xml file (expected to be train.xml or testcases.xml).
      global_feat_dict is a dictionary mapping feature_names to column-numbers; it
      should only be provided when extracting features from test data, so that 
      the columns of the test matrix align correctly.

    returns: 
      a sparse design matrix, a dict mapping features to column-numbers,
      a vector of target values, and a list of movie ids in order of their
      rows in the design matrix
    # TODO train here, and return regression parameters
    print "learning..."
    learned_w = splinalg.lsqr(X_train,y_train)[0]
    print "done learning"
    print
    
    # get rid of training data and load test data
    del X_train
    del y_train
    del train_ids
    print "extracting test features..."
    X_test,_,y_ignore,test_ids = extract_feats(ffs, testfile, global_feat_dict=global_feat_dict)
    print "done extracting test features"
    print
    
    # TODO make predictions on text data and write them out
    print "making predictions..."
    preds = X_test.dot(learned_w)
    print "done making predictions"
    print
    
    print "writing predictions..."
    util.write_predictions(preds, test_ids, outputfile)
    print "done!"

if __name__ == "__main__":
    main()
    
