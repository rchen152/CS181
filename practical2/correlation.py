import numpy as np
import util
import math
import scipy.stats as stats
try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET
import matplotlib.pyplot as pp
import heapq as heap
import regression_starter as rs

CONT_FEATS = ['running_time','number_of_screens','production_budget',
              'num_highest_grossing_actors','num_oscar_winning_actors']

BOOL_FEATS = ['rating','highest_grossing_actors_present',
              'oscar_winning_directors_present', 'oscar_winning_actors_present',
              'summer_release', 'christmas_release', 'memorial_release',
              'independence_release', 'labor_release']

MOVIE_TARGET = 10
WORD_TARGET = 10

def correlate(lst1, lst2):
    return np.corrcoef(np.array(lst1), np.array(lst2))[0][1]

def comp_p(lst1, lst2):
    return stats.ks_2samp(np.array(lst1), np.array(lst2))

def part(k,v):
    if k == 'rating':
        return v == 'R'
    elif 'present' in k:
        return True
    else:
        return v

def extract_gen_feats(feats, data_file, all_feats, build, update, postproc):
    good_feats = []
    for feat in feats:
        if feat in all_feats:
            good_feats.append(feat)
        else:
            print (feat + " is not a supported feature")
    feats = good_feats

    feat_dict = build(feats)
    curr_inst = []

    begin_tag = "<instance" # for finding instances in the xml file
    end_tag = "</instance>"
    in_instance = False

    with open(data_file) as f:
        f.readline()
        f.readline()
        for line in f:
            if begin_tag in line:
                if in_instance:
                    assert False
                else:
                    curr_inst = [line]
                    in_instance = True
            elif end_tag in line:
                curr_inst.append(line)
                md = util.MovieData(ET.fromstring("".join(curr_inst)))
                feat_dict = update(md, feats, feat_dict)
                curr_inst = []
                in_instance = False
            elif in_instance:
                curr_inst.append(line)

    feat_dict = postproc(feat_dict)
    return feat_dict

def extract_cont_feats(feats, data_file):
    def build(feat_lst):
        feat_dict = {}
        for k in feat_lst:
            feat_dict[k] = {'total':0., 'count':0., 'vec':[]}
        feat_dict['regy'] = {'total':0., 'count':0., 'vec':[]}
        return feat_dict
    def update(md, feat_lst, feat_dict):
        for k,v in md.__dict__.iteritems():
            if k in feat_lst:
                feat_dict[k]['total'] += v
                feat_dict[k]['count'] += 1
                feat_dict[k]['vec'].append(v)
        feat_dict['regy']['total'] += md.target
        feat_dict['regy']['count'] += 1
        feat_dict['regy']['vec'].append(md.target)
        for k in feat_lst:
            if len(feat_dict[k]['vec']) < feat_dict['regy']['count']:
                feat_dict[k]['vec'].append(0.)
        return feat_dict
    def postproc(feat_dict):
        feat_out = {}
        for k in feat_dict:
            tot = feat_dict[k]['total']
            cnt = feat_dict[k]['count']
            zvec = [i - (tot/cnt) for i in feat_dict[k]['vec']]
            feat_out[k] = {'total':tot, 'count':cnt, 'vec': zvec}
        return feat_out
    return extract_gen_feats(feats, data_file, CONT_FEATS,
                             build, update, postproc)

def extract_bool_feats(feats, data_file):
    def build(feat_lst):
        feat_dict = {}
        for k in feat_lst:
            feat_dict[k] = {'sat'  :False,
                            'true' :
                            {'tot':0.,'lg_tot':0.,'count':0,'vec':[]},
                            'false':
                            {'tot':0.,'lg_tot':0.,'count':0,'vec':[]}}
        return feat_dict
    def update(md, feat_lst, feat_dict):
        for k in feat_lst:
            feat_dict[k]['sat'] = False
        for k,v in md.__dict__.iteritems():
            if k in feat_lst:
                feat_dict[k]['sat'] = part(k,v)
        for k in feat_lst:
            sat = str(feat_dict[k]['sat']).lower()
            feat_dict[k][sat]['vec'].append(md.target)
            feat_dict[k][sat]['tot'] += md.target
            feat_dict[k][sat]['lg_tot']+=math.log(md.target)
            feat_dict[k][sat]['count'] += 1
        return feat_dict
    def postproc(feat_dict):
        for k in feat_dict:
            feat_dict[k].pop('sat')
        return feat_dict
    return extract_gen_feats(feats, data_file, BOOL_FEATS,
                             build, update, postproc)

def extract_reviews(data_file):
    def build(feat_lst):
        return []
    def update(md, feat_lst, feat_dict):
        text = rs.unigram_feats(md)
        text['regy'] = md.target
        feat_dict.append(text)
        return feat_dict
    def postproc(feat_dict):
        return rs.make_design_mat(feat_dict)
    return extract_gen_feats([], data_file, [], build, update, postproc)

'''feat_dict = extract_cont_feats(['production_budget','ohaiyo'], 'train.xml')
for k in feat_dict:
    print k
    print feat_dict[k]['count']
    print correlate(feat_dict[k]['vec'], feat_dict['regy']['vec'])
    print '--------'

feat_dict = extract_bool_feats(['fizzbuzz', 'rating'], 'train.xml')
for k in feat_dict:
    true = feat_dict[k]['true']
    false = feat_dict[k]['false']
    diff = (true['tot']/true['count']) - (false['tot']/false['count'])
    lg_diff = (true['lg_tot']/true['count'])-(false['lg_tot']/false['count'])
    print k
    print 'diff ' + str(diff)
    print 'log mean diff ' + str(lg_diff)
    print 'true  ' + str(len(true['vec']))
    print 'false ' + str(len(false['vec']))
    print comp_p(true['vec'], false['vec'])'''
#    print '--------'

'''mat, key = extract_reviews('train.xml')

num_movies,_ = mat.get_shape()
xs = []
ys = []
for i in range(num_movies):
    regy = mat[i,key['regy']]
    mat[i,key['regy']] = 0
    xs.append(mat[i].sum())
    ys.append(math.log(regy))
pp.plot(xs,ys,'ro')
pp.show()'''

'''inv_key = {v:k for k,v in key.items()}
num_movies,num_words = mat.get_shape()
min_movies = []
max_movies = []

for i in range(num_movies):
    regy = mat[i,key['regy']]
    if len(min_movies) < MOVIE_TARGET:
        heap.heappush(min_movies, (-1.*regy,i))
    else:
        heap.heappushpop(min_movies, (-1.*regy,i))
    if len(max_movies) < MOVIE_TARGET:
        heap.heappush(max_movies, (regy,i))
    else:
        heap.heappushpop(max_movies, (regy,i))

words = np.zeros((num_words))
for mv in min_movies:
    words += -1*mat[mv[1]]
for mv in max_movies:
    words += mat[mv[1]]'''
