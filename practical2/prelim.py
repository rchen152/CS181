import heapq as heap
import math
import matplotlib.pyplot as pp
import numpy as np
import regression_starter as rs
import scipy.stats as stats

CONT_FEATS = ['running_time','number_of_screens','production_budget',
              'num_highest_grossing_actors','num_oscar_winning_actors']

BOOL_FEATS = ['rating-R','highest_grossing_actors_present',
              'oscar_winning_directors_present', 'oscar_winning_actors_present',
              'summer_release', 'christmas_release', 'memorial_release',
              'independence_release', 'labor_release']

MOVIE_TARGET = 100
WORD_TARGET = 100

def proc_metadata_feats():
    mat,key,regy,_ = rs.extract_feats([rs.metadata_feats])
    num_movies = len(regy)

    for feat in CONT_FEATS:
        print feat
        print np.corrcoef(mat.getcol(key[feat]).todense().transpose(),regy)[0,1]
        print '-------------'

    for feat in BOOL_FEATS:
        feat_arr = mat.getcol(key[feat]).todense().transpose()
        tvec = [regy[i] for i in range(len(regy)) if feat_arr[0,i] == 1.]
        fvec = [regy[i] for i in range(len(regy)) if feat_arr[0,i] == 0.]
        tlen = len(tvec)
        flen = len(fvec)

        print feat + ' ' + str(tlen) + '/' + str(flen)
        print 'mean diff ' + str((sum(tvec)/tlen) - (sum(fvec)/flen))
        print 'log mean diff ' + str((sum([math.log(r) for r in tvec])/tlen) -
                                     (sum([math.log(r) for r in fvec])/flen))
        print stats.ks_2samp(np.array(tvec),np.array(fvec))[0]
        print '-------------'

def proc_unigram_feats():
    mat,key,regy,_ = rs.extract_feats([rs.unigram_feats])
    inv_key = {v:k for k,v in key.items()}
    num_movies,num_words = mat.get_shape()

    movies = [(regy[i],i) for i in range(num_movies)]
    min_movies = heap.nsmallest(MOVIE_TARGET,movies)
    max_movies = heap.nlargest(MOVIE_TARGET,movies)
    tot_min = 0.
    tot_max = 0.
    for mv in min_movies:
        tot_min += mat[mv[1]].sum()
    for mv in max_movies:
        tot_max += mat[mv[1]].sum()
    fix = tot_max/tot_min
    diffs = np.zeros((num_words))
    for mv in min_movies:
        diffs += -1.*fix*mat[mv[1]]
    for mv in max_movies:
        diffs += mat[mv[1]]

    with open("english.stop") as f:
        stop_words = set([line.strip() for line in f.readlines()])
        words = [(diffs[0,i],inv_key[i]) for i in range(num_words)
                 if inv_key[i] not in stop_words]
        worst_words = heap.nsmallest(WORD_TARGET, words)
        worst_words.sort()
        best_words = heap.nlargest(WORD_TARGET, words)
        best_words.sort()
        for wd in worst_words:
            print wd[1] + '\t' + str(wd[0])
        print '---------------------------------'
        for wd in best_words:
            print wd[1] + '\t' + str(wd[0])

def corr_words():
    

corr_words
