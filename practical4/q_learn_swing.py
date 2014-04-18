import numpy as np
import numpy.random as npr
import sys
import matplotlib.pyplot as plt
from SwingyMonkey import SwingyMonkey

tree_dist_bins = 5
tree_top_bins = 5
m_vel_bins = 5
m_top_bins = 5

num_act = 2

screen_width  = 600
screen_height = 400
horz_speed    = 25
impulse       = 15
gravity       = 3
tree_mean     = 5
tree_gap      = 200
tree_offset   = -300
edge_penalty  = -10.0
tree_penalty  = -5.0
tree_reward   = 1.0
max_m_vel     = 15
min_m_vel     = 0
min_tree_dist = -100
min_tree_top  = 200
max_tree_top  = 350
max_m_top      = 250
min_m_top      = 50

GAMMA         = 1


EPSILON0      = float(1)

ALPHA0        = 1

score_vec = []
rolling_avg_param = 20



def get_coord(state):
    if (state['tree']['dist']<= min_tree_dist):
        tree_dist = 0
    else:
#assumes the distace to the tree is at most the distance of the screen. Computes the bin to put the distance in
        tree_dist = (state['tree']['dist']+min_tree_dist)*tree_dist_bins/(min_tree_dist+screen_width) 

    if (state['tree']['top'] <= min_tree_top):
        tree_top = 0
    elif (state['tree']['top'] >= max_tree_top):
        tree_top = tree_top_bins - 1
    else:
        tree_top = (state['tree']['top']-min_tree_top) * tree_top_bins/(max_tree_top - min_tree_top)


    if(state['monkey']['vel'] <= min_m_vel):
        m_vel = 0
    elif(state['monkey']['vel'] >= max_m_vel):
        m_vel = m_vel_bins - 1            
    else:
        m_vel = (state['monkey']['vel']-min_m_vel) * m_vel_bins / (max_m_vel - min_m_vel)

    if state['monkey']['top'] >= max_m_top:
        m_top = m_top_bins - 1
    elif state['monkey']['top'] <= min_m_top:
        m_top = 0
    else:
        m_top = (state['monkey']['top']-min_m_top) * m_top_bins / (max_m_top-min_m_top)

    return (tree_dist,tree_top,m_vel,m_top)

class Learner:

    def __init__(self):
        self.last_state  = None
        self.last_action = None
        self.last_reward = None
        self.q_fn = np.zeros((tree_dist_bins,tree_top_bins,m_vel_bins, m_top_bins,num_act))
        self.time_step   = 1
        self.counter = np.zeros((tree_dist_bins,tree_top_bins,m_vel_bins, m_top_bins,num_act))
        self.epoch = float (1)
        self.score = 0
        self.avg_score = float(0)

    def reset(self):
        self.last_state  = None
        self.last_action = None
        self.last_reward = None
        self.epoch +=1
        #print self.score
        self.avg_score = (self.avg_score*self.epoch + self.score)/(self.epoch+float(1))
        #print self.avg_score
        score_vec.append(self.score)
        self.score = 0
#        self.time_step   = 1

        if self.epoch > 100:
            td_inds = np.zeros(tree_dist_bins)
            tt_inds = np.zeros(tree_top_bins)
            mv_inds = np.zeros(m_vel_bins)
            mt_inds = np.zeros(m_top_bins)
            na_inds = np.zeros(num_act)
            for a in range(tree_dist_bins):
                for b in range(tree_top_bins):
                    for c in range(m_vel_bins):
                        for d in range(m_top_bins):
                            for e in range(num_act):
                                if not self.counter[a,b,c,d,e] == 0:
                                    td_inds[a] += 1
                                    tt_inds[b] += 1
                                    mv_inds[c] += 1
                                    mt_inds[d] += 1
                                    na_inds[e] += 1
            print td_inds
            print tt_inds
            print mv_inds
            print mt_inds
            print na_inds

    def action_callback(self, state):
        self.score = state['score']
        '''Implement this function to learn things and take actions.
        Return 0 if you don't want to jump and 1 if you do.'''

        # You might do some learning here based on the current state and the last state.

        # You'll need to take an action, too, and return it.
        # Return 0 to swing and 1 to jump.

        coords = get_coord(state)
        reward0 = self.q_fn[coords[0],coords[1],coords[2],coords[3],0]
        reward1 = self.q_fn[coords[0],coords[1],coords[2],coords[3],1]
        new_action = 0
        self.time_step += 1
        rand_num = npr.rand()
        if ((reward1 > reward0) and (rand_num > EPSILON0/float(self.time_step))) or ((reward1 < reward0) and (rand_num < EPSILON0/float(self.time_step))):
            new_action = 1
                
        if not self.last_state == None:
            old_coords = get_coord(self.last_state)
            old_action = self.last_action
            old_q_val = self.q_fn[old_coords[0],old_coords[1],old_coords[2],old_coords[3],old_action]
            self.counter[old_coords[0],old_coords[1],old_coords[2],old_coords[3],old_action] += 1
    
            curr_q_val = max(reward0, reward1)
<<<<<<< HEAD
            self.q_fn[old_coords[0],old_coords[1],old_coords[2],old_coords[3],old_action] = old_q_val + ALPHA0/self.counter[old_coords[0],old_coords[1],old_coords[2],old_coords[3],old_action] * ((self.last_reward + (GAMMA * curr_q_val)) - old_q_val)

=======
#            self.q_fn[old_coords[0],old_coords[1],old_coords[2],old_coords[3],old_action] = old_q_val + ALPHA0 /self.counter[old_coords[0],old_coords[1],old_coords[2],old_coords[3],old_action] * ((self.last_reward + (GAMMA * curr_q_val)) - old_q_val)
            self.q_fn[old_coords[0],old_coords[1],old_coords[2],old_coords[3],old_action] = old_q_val + ALPHA0/self.epoch * ((self.last_reward + (GAMMA * curr_q_val)) - old_q_val)
>>>>>>> 8f26177f967fd5b57e01d2702a8c63c8cb289bb9

#        print self.q_fn[coords[0],coords[1],coords[2],coords[3],new_action]
        self.last_action = new_action
        self.last_state  = state

        return self.last_action

    def reward_callback(self, reward):
        '''This gets called so you can see what reward you get.'''

        self.last_reward = reward
  
iters = 1000
learner = Learner()

for ii in xrange(iters):

    # Make a new monkey object.
    swing = SwingyMonkey(sound=False,            # Don't play sounds.
                         text="Epoch %d" % (ii), # Display the epoch on screen.
                         tick_length=1,          # Make game ticks super fast.
                         action_callback=learner.action_callback,
                         reward_callback=learner.reward_callback)

    # Loop until you hit something.
    while swing.game_loop():
        pass

    # Reset the state of the learner.
    learner.reset()

x = np.arange(len(score_vec))

#fig, ax = plt.subplots()
#ax.bar(x, score_vec, 0.5, color='white')

rolling_num = 20
avg_score_vec = [sum(score_vec[i-rolling_num:i])/float(rolling_num) for i in x]
fig, ax = plt.subplots()
ax.bar(x, avg_score_vec, 0.5, color='white')
plt.show()
