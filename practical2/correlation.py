import numpy as np
import util
try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET
import matplotlib.pyplot as pp

FEATS = ['running_time','number_of_screens','production_budget','num_oscar_winning_directors','num_highest_grossing_actors','num_oscar_winning_actors']

def correlate(lst1, lst2):
    return np.corrcoef(np.array(lst1), np.array(lst2))[0][1]

def to_float(v):
    if type(v) is bool:
        if v == True:
            return 1.
        else:
            return 0.
    else:
        return v

def extract_feats(feats, data_file):
    good_feats = []
    for feat in feats:
        if feat in FEATS:
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

feats = extract_feats(FEATS, 'train.xml')
computed = []
for k1 in feats:
    for k2 in feats:
        if not k1==k2 and not (k2,k1) in computed:
            print k1
            print k2
            print correlate(feats[k1]['vec'], feats[k2]['vec'])
            print '-------------'
            computed.append((k1,k2))
            pp.plot(feats[k1]['vec'], feats[k2]['vec'], 'ro')
            pp.show()
