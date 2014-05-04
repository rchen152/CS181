import numpy as np
import cPickle as pickle
GAME_LEN = 1000

NUM_STATES = 9*13*33

sa_ds = pickle.load(open('pickled_sa.p','r'))

for i in range (len(sa_ds)):
    sa_ds[i]['total_reward'] = -sa_ds[i]['total_reward']

# Simply implement the Vs as a matrix.
# Note that the y axis is GAME_LEN + 1, as the first row is simply a dummy of zeros

the_V = np.zeros((GAME_LEN + 1,NUM_STATES))

for j in range(1, GAME_LEN + 1):
    print j
    for current_state in range(NUM_STATES):
        potential_vs = []
        for k in range(4):
            # for each action, take reward associated to (current_State, k) and add it
            if (sa_ds[current_state * 4 + k]['count']==0):
                expected_reward = 0
            else:
                expected_reward = float(sa_ds[current_state * 4 + k]['total_reward'])/sa_ds[current_state * 4 + k]['count']
            
            # calculate probability
            expected_vs = 0
            for new_state in sa_ds[current_state * 4 + k]:
                if (new_state != 'count' and new_state != 'total_reward'):
                    my_p = float(sa_ds[current_state*4 + k][new_state])/sa_ds[current_state*4 + k]['count'] 
                    expected_vs += my_p * the_V[j-1][new_state]
            net_reward = expected_reward + expected_vs
            potential_vs.append(net_reward)
        # updates the v value to maximum
        the_V[j][current_state] = max (potential_vs)    

print the_V
    
pmat = open("value_matrix.p","w")
pickle.dump(the_V,pmat)
pmat.close()



