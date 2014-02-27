import numpy as np
import util
import math
try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET
import matplotlib.pyplot as pp
import regression_starter

FEATS = ['running_time','number_of_screens','production_budget','num_oscar_winning_directors','num_highest_grossing_actors','num_oscar_winning_actors']

def get_movies(data_file):
    movies = []
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
                movies.append(util.MovieData(ET.fromstring("".join(curr_inst))))
                curr_inst = []
                in_instance = False
            elif in_instance:
                curr_inst.append(line)
    return movies

'''movies = get_movies('train.xml')
print regression_starter.metadata_feats(movies[0])
print '-------'
print regression_starter.unigram_feats(movies[0])'''

fds = [{'hi':1,'bye':0,'foo':3},{'hi':1,'hello':1,'foo':2,'bar':0}]
'''movies = get_movies('train.xml')
fd1 = regression_starter.metadata_feats(movies[0])
fd2 = regression_starter.metadata_feats(movies[1])
print fd1
print fd2
print '------'
fds = [fd1,fd2]'''
X, dict = regression_starter.make_design_mat(fds)
print X
print '------'
print dict
