import cPickle as pickle
import numpy as np

num_states = 450
num_actions = 4
sa_ds = np.array([])
for i in range(num_states * num_actions):
    sa_ds =np.append(sa_ds,{'count':0,'total_reward':0})

pickle.dump(sa_ds,open("small_arr.p","w"))

