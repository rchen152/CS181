import numpy as np
import util
import math
import scipy.stats as stats
try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET
import matplotlib.pyplot as pp

CONT_FEATS = ['running_time','number_of_screens','production_budget',
              'num_highest_grossing_actors','num_oscar_winning_actors']

#BOOL_FEATS = ['summer_release', 'christmas_release', 'memorial_release',
              #'independence_release', 'labor_release']
#BOOL_FEATS = ['rating']
BOOL_FEATS = ['highest_grossing_actors_present', 'oscar_winning_directors_present', 'oscar_winning_actors_present']

def correlate(lst1, lst2):
    return np.corrcoef(np.array(lst1), np.array(lst2))[0][1]

def comp_p(lst1, lst2):
    return stats.ks_2samp(np.array(lst1), np.array(lst2))

def get_part(k,v):
    if k == 'rating':
        return v == 'R'
    elif 'present' in k:
        return True
    else:
        return v

def extract_cont_feats(feats, data_file):
    good_feats = []
    for feat in feats:
        if feat in CONT_FEATS:
            good_feats.append(feat)
        else:
            print (feat + " is not a supported feature")
    feats = good_feats

    feat_dict = {}
    num_feats = len(feats)
    for i in range(num_feats):
        feat_dict[feats[i]] = {'total':0., 'count':0., 'vec':[]}

    reg_dict = {'total':0., 'count':0., 'vec':[]}
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
                for k,v in md.__dict__.iteritems():
                    for i in range(num_feats):
                        if feats[i] == k:
                            feat_dict[feats[i]]['total'] += v
                            feat_dict[feats[i]]['count'] += 1
                            feat_dict[feats[i]]['vec'].append(v)
                reg_dict['total'] += md.target
                reg_dict['count'] += 1
                reg_dict['vec'].append(md.target)
                for i in range(num_feats):
                    if len(feat_dict[feats[i]]['vec']) < reg_dict['count']:
                        feat_dict[feats[i]]['vec'].append(0.)
                curr_inst = []
                in_instance = False
            elif in_instance:
                curr_inst.append(line)

    feat_dict['regy'] = reg_dict
    num_feats += 1
    feat_out = {}
    for k in feat_dict:
        tot = feat_dict[k]['total']
        cnt = feat_dict[k]['count']
        zvec = [i - (tot/cnt) for i in feat_dict[k]['vec']]
        feat_out[k] = {'total':tot, 'count':cnt, 'vec': zvec}
    return feat_out

def partition_bool_feats(feats, data_file):
    good_feats = []
    for feat in feats:
        if feat in BOOL_FEATS:
            good_feats.append(feat)
        else:
            print (feat + " is not a supported feature")
    feats = good_feats

    feat_dict = {}
    num_feats = len(feats)
    for i in range(num_feats):
        feat_dict[feats[i]] = {'true':{'tot':0.,'lg_tot':0.,'count':0,'vec':[]},
                               'false':{'tot':0.,'lg_tot':0.,'count':0,'vec':[]}}
    flags = ['false','false','false']
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
                flags = ['false','false','false']
                for k,v in md.__dict__.iteritems():
                    for i in range(num_feats):
                        if feats[i] == k:
                            flags[i] = 'true'
                            #part = 'false'
                            #if v == 'R':
                                #part = 'true'
                            #feat_dict[k][part]['vec'].append(md.target)
                            #feat_dict[k][part]['tot'] += md.target
                            #feat_dict[k][part]['lg_tot'] += math.log(md.target)
                            #feat_dict[k][part]['count'] += 1
                for i in range(3):
                    feat_dict[feats[i]][flags[i]]['vec'].append(md.target)
                    feat_dict[feats[i]][flags[i]]['tot'] += md.target
                    feat_dict[feats[i]][flags[i]]['lg_tot']+=math.log(md.target)
                    feat_dict[feats[i]][flags[i]]['count'] += 1
                curr_inst = []
                in_instance = False
            elif in_instance:
                curr_inst.append(line)

    return feat_dict

feat_dict = partition_bool_feats(BOOL_FEATS, 'train.xml')
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
    print comp_p(true['vec'], false['vec'])
    print '--------'

    pp.plot(true['vec'], 'ro')
    pp.plot(false['vec'], 'bo')
    pp.show()
    pp.clf()
