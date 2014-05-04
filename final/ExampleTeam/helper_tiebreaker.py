# function that returns the directions that reduce distance, if any. If none
def closer_dist (observedState, object_pos, allowed_moves):
    final_dir = []
    current_dis = self.distancer.getDistance(observedState.getPacmanPosition, object_pos)
    for d in allowed_moves:
        nextPos = observedState.pacmanFuturePosition([d])
        if (self.distancer.getDistance(nextPos, object_pos) < current_dis):
            final_dir.append(d)
    return final_dir

def farther_dist (observedState, object_pos, allowed_moves):
    final_dir = []
    current_dis = self.distancer.getDistance(observedState.getPacmanPosition, object_pos)
    for d in allowed_moves:
        nextPos = observedState.pacmanFuturePosition([d])
        if (self.distancer.getDistance(nextPos, object_pos) > current_dis):
            final_dir.append(d)
    return final_dir    
        
# returns direction farther away from ghost, if there are none, or if there are ties,
# breaks ties randomly
        
def filter_direction_bad_ghost_far (loc_bad_ghost, observedState, original_directions):
    good_dir = farther_dist (observedState, loc_bad_ghost, original_directions)
    if len(good_dir) == 0:
        return random.choice(original_directions)
    else:
        return random.choice(good_dir)

# returns direction closer to ghost, again breaks ties randomly 
def filter_direction_bad_ghost_close (loc_bad_ghost, observedState, original_directions):
    good_dir = closer_dist (observedState, loc_bad_ghost, original_directions)
    if len(good_dir) == 0:
        return random.choice(original_directions)
    else:
        return random.choice(good_dir)


