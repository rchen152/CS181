import numpy as np
import matplotlib.pyplot as plt

# Constants
num_states = 100
num_actions = 16

# Array for recording reward at each score
value_arr = range(num_states+16)
for i in range(100):
    value_arr[i] = 0
value_arr[100] = 1
for i in range(101,num_states+16):
    value_arr[i] = -1

# The dart board
box_arr = [[0,0,0,0,0,0],[0,7,12,1,14,0],[0,2,13,8,11,0],[0,16,3,10,5,0],
           [0,9,6,15,4,0],[0,0,0,0,0,0]]

# For remembering the policy between iterations
policy_arr = np.zeros(num_states)
old_policy = np.ones(num_states)

# The q function
q_fn = np.zeros(num_states*num_actions).reshape((num_states,num_actions))

# Finding optimal policy and reward at each score
while not (policy_arr == old_policy).all():
    for i in range(len(old_policy)):
        old_policy[i] = policy_arr[i]
    for s in range(num_states):
        for a in range(num_actions):
            center = box_arr[a%4+1][a/4+1]
            north = box_arr[a%4+1][a/4+2]
            south = box_arr[a%4+1][a/4]
            east = box_arr[a%4+2][a/4+1]
            west = box_arr[a%4][a/4+1]
            q_fn[s][a] = (.6*value_arr[s+center] + .1*
                          (value_arr[s+north]+value_arr[s+south]+
                           value_arr[s+east]+value_arr[s+west]))
    for s in range(num_states):
        policy_arr[s] = np.argmax (q_fn[s])
        value_arr[s] = max(q_fn[s])

# Print results            
print value_arr
print argmax(value_arr[0:100])
print policy_arr

# Plot rewards
x = np.arange(101)
fig, ax = plt.subplots()
ax.bar(x, value_arr[:101], 0.5, color='white')
plt.show()
