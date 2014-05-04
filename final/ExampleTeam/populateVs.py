from distanceCalculator import Distancer
from game import Actions
from game import Directions
from observedState import ObservedState
import numpy as np
import csv

from util import random, manhattanDistance, Counter, chooseFromDistribution, raiseNotDefined
import cPickle as pickle
from util import random, manhattanDistance, Counter, chooseFromDistribution
import pickle
import classify as cfy


GAME_LEN = 1000
BAD_QUAD = 4
NUM_GHOSTS = 4

AVG_CLASS_JUICE =[28.867748179685883, 52.257299401447447, 153.57566648602614,
                  17.2900555038538, 0,0]

BG_RANGE = 4
GG_RANGE = 3
CAP_RANGE = 2
NUM_DIRS = 4
NUM_MOVES = NUM_DIRS + 1

badGhost = None
prevGhostStates = []



NUM_STATES = 9*13*33

sa_ds = pickle.load(open('ExampleTeam/pickled_sa.p','r'))

# Simply implement the Vs as a matrix.
# Note that the y axis is GAME_LEN + 1, as the first row is simply a dummy of zeros

the_V = np.zeros(NUM_STATES, GAME_LEN + 1)


for j in range(1, GAME_LEN + 1):
    for current_state in range(NUM_STATES):
        potential_vs = []
        for k in range(3):
            # for each action, take reward associated to (current_State, k) and add it 
            expected_reward = float(sa_ds[current_state * 4 + k]['total_reward'])/sa_ds[current_state * 4 + k]['count']
            
            # calculate probability
            expected_vs = 0
            for new_state in sa_ds[current_state * 4 + k]:
                if (new_state != 'count' & new_state != 'total_reward'):
                    my_p = float(sa_ds[current_state*4 + k][new_state])/sa_ds[current_state*4 + k]['count'] 
                    expected_vs += my_p * the_V[new_state][j-1]
            net_reward = expected_reward + expected_vs
            potential_vs.append(net_reward)
        # updates the v value to maximum
        the_V[current_state][j] = max (potential_vs)    
    
# Pickle the resulting matrix??






# Given the Vs, how to act:
global the_V
dir_dict = {Directions.NORTH:0,Directions.SOUTH:1,Directions.EAST:2,Directions.WEST:3,Directions.STOP:4}

# Note the remaining_time...
# Please check the correctness of indices?
def chooseAction(self,observedState, remaining_time):

    # this is the default algorithm
    
    legalActs = [a for a in observedState.getLegalPacmanActions()]
    fil_legal = filter(lambda x: x != Directions.STOP ,legalActs)
    total_reward = []          
    for i in range(len(fil_legal)):
        x = fil_legal[i]
        # calculate the award for (observedState, x)
        my_award = float(sa_ds[observedState * 4 + dir_dict[x]]['total_reward'])/sa_ds[observedState * 4 + dir_dict[x]]['count']
        # calculate expected vs
        expected_vs = 0
        for new_state in sa_ds[observedState * 4 + dir_dict[x]]:
            if (new_state != 'count' & new_state != 'total_reward'):
                my_p = float(sa_ds[observedState * 4 + dir_dict[x]][new_state])/sa_ds[observedState * 4 + dir_dict[x]]['count'] 
                # Not sure if this index is correct
                expected_vs += my_p * the_V[new_state][GAME_LEN + 1 -(remaining_time + 1)]
        net_reward = my_award + expected_vs
        total_reward.append(net_reward)
    # Then choose the x with maximum net_reward to return
    return fil_legal[total_reward.index(max(total_reward))]

    # if we are in some different situation, do other things?
    
#    if observedState == ??

# Rough heuristic:
# If bad ghost scared:

# If bad ghost not scared:

#no bad ghost and no good gh
