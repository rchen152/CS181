import numpy as np
import util
try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET

def get_feats(feat, datafile):
    feat_lst = []
    reg_lst = []
    curr_inst = []

    begin_tag = "<instance" # for finding instances in the xml file
    end_tag = "</instance>"
    in_instance = False

    with open(datafile) as f:
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
                featPresent = False
                for k,v in md.__dict__.iteritems():
                    if feat == k:
                        feat_lst.append(1.)
                        featPresent = True
                        #if v == True:
                            #feat_lst.append(1.)
                        #if v == False:
                            #feat_lst.append(0.)
                        #feat_lst.append(v)
                        #reg_lst.append(md.target)
                if not featPresent:
                    feat_lst.append(0.)
                reg_lst.append(md.target)
                featPresent = False
                curr_inst = []
                in_instance = False
            elif in_instance:
                curr_inst.append(line)
    return [feat_lst, reg_lst]

def correlate(lst1, lst2):
    return np.corrcoef(np.array(lst1), np.array(lst2))[0][1]

[feat_lst, reg_lst] = get_feats("highest_grossing_actors_present", "train.xml")
#[feat_lst2, _] = get_feats("number_of_screens", "train.xml")
print len(feat_lst)
print len(reg_lst)
print correlate(feat_lst, reg_lst)
