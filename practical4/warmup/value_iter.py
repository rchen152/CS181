from itertools import izip
import numpy as np
import matplotlib.pyplot as plt

argmax = lambda array: max(izip(array, xrange(len(array))))[1]

num_states = 100
num_actions = 16

value_arr = range(num_states+16)
for i in range(100):
    value_arr[i] = 0
value_arr[100] = 1
for i in range(101,num_states+16):
    value_arr[i] = -1
policy_arr = np.zeros(num_states)

q_fn = np.zeros(num_states*num_actions).reshape((num_states,num_actions))

box_arr = [[0,0,0,0,0,0],[0,7,12,1,14,0],[0,2,13,8,11,0],[0,16,3,10,5,0],[0,9,6,15,4,0],[0,0,0,0,0,0]]

old_policy = np.ones(num_states)

iteration = 0

def list_equality (l1, l2):
    equal = True
    if (len(l1) != len(l2)):
        return false
    else:
        for i in range(len(l1)):
            equal = equal and (l1[i]==l2[i])
        return equal

while(not(list_equality (policy_arr, old_policy))):
    for i in range(len(old_policy)):
        old_policy[i] = policy_arr[i]
    for s in range(num_states):
        for a in range(num_actions):
            center = box_arr[a%4+1][a/4+1]
            north = box_arr[a%4+1][a/4+2]
            south = box_arr[a%4+1][a/4]
            east = box_arr[a%4+2][a/4+1]
            west = box_arr[a%4][a/4+1]
            q_fn[s][a] = .6*value_arr[s+center] + .1*(value_arr[s+north]+value_arr[s+south]+value_arr[s+east]+value_arr[s+west])
    for s in range(num_states):        
        policy_arr[s] = argmax (q_fn[s])
        value_arr[s] = max(q_fn[s])
    iteration +=1
    print iteration
            
print value_arr
print policy_arr
print old_policy

print box_arr[1][3]
x = np.arange(100)
plt.plot(x,value_arr[:100])
plt.show()

