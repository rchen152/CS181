from distanceCalculator import Distancer
from game import Actions
from game import Directions
from observedState import ObservedState
import numpy as np
import csv
from util import random, manhattanDistance, Counter, chooseFromDistribution, raiseNotDefined
import pickle
import classify

GOOD_CAPS_CSV = 'data/good_caps_train.csv'
GAME_LEN = 1000
BAD_QUAD = 4
NUM_GHOSTS = 4
bad_ghost_vec = np.array([])
previous_ghost_state = np.array([])
avg_class_juice =[28.867748179685883, 52.257299401447447, 153.57566648602614, 17.2900555038538, 0,0]

class BaseStudentAgent(object):
    """Superclass of agents students will write"""

    def registerInitialState(self, gameState):
        """Initializes some helper modules"""
        import __main__
        self.display = __main__._display
        self.distancer = Distancer(gameState.data.layout, False)
        self.firstMove = True

    def observationFunction(self, gameState):
        """ maps true state to observed state """
        return ObservedState(gameState)

    def getAction(self, observedState):
        """ returns action chosen by agent"""
        return self.chooseAction(observedState)

    def chooseAction(self, observedState):
        "By default, a BustersAgent just stops.  This should be overridden."
        return Directions.STOP

class CoequalizerAgent(BaseStudentAgent):
    """
    An example TeamAgent. After renaming this agent so it is called <YourTeamName>Agent,
    (and also renaming it in registerInitialState() below), modify the behavior
    of this class so it does well in the pacman game!
    """



    def __init__(self, *args, **kwargs):
        """
        arguments given with the -a command line option will be passed here
        """

        pass # you probably won't need this, but just in case
    
    def registerInitialState(self, gameState):
        """
        Do any necessary initialization
        """
        # Here, you must replace "ExampleTeamAgent" with "<YourTeamName>Agent"
        super(CoequalizerAgent, self).registerInitialState(gameState)
        
        # Here, you may do any necessary initialization, e.g., import some
        # parameters you've learned, as in the following commented out lines
        # learned_params = cPickle.load("myparams.pkl")
        # learned_params = np.load("myparams.npy") 
        
        ghost_params = pickle.load(open('train/pickled_tree_101_200.p','r'))


    def badFeature(self, ghostState,observedState):
        badghosts = filter(lambda x: ObservedState.getGhostQuadrant(observedState,x) == BAD_QUAD,ghostState)
        if(len(badghosts)<1):
            print "fewer bad ghosts than expected error"
            return None
        elif(len(badghosts) > 1):
            print "more bad ghosts than expected error"
        else:    
            return badghosts[0].getFeatures()

    def updateBadGhost(self, observedState):
        global bad_ghost_vec
        global previous_ghost_state
        
        ghost_states = observedState.getGhostStates() # states have getPosition() and getFeatures() methods
        if(GAME_LEN == observedState.getNumMovesLeft()):
            return self.badFeature(ghost_states,observedState)

        ghost_features = map(lambda x : x.getFeatures(),ghost_states)

        if (filter(lambda x : (x == bad_ghost_vec).all(), ghost_features) == []):
            possible_ghosts = filter(lambda x: ObservedState.getGhostQuadrant(observedState,x) == BAD_QUAD,ghost_states)
            if(len(possible_ghosts)<1):
                print "error no quad 4 ghosts"
                return np.array([])
            if(len(possible_ghosts) == 1):
                return possible_ghosts[0].getFeatures()
            if(len(possible_ghosts)>1):
                prev_ghost_features = map(lambda x : x.getFeatures() , previous_ghost_state)
                b_g_candidates = filter(lambda x : (filter(lambda y : (y==x).all(),prev_ghost_features)==[]), possible_ghosts)
                if(len(b_g_candidates) != 1):
                    print "not exactly one ghost regenerated in quadrant 4 error"
                    return np.array([])
                else: return b_g_candidates[0].getFeatures()
        else:
            return bad_ghost_vec

    def chooseAction(self, observedState):
        """
        Here, choose pacman's next action based on the current state of the game.
        This is where all the action happens.
        
        This silly pacman agent will move away from the ghost that it is closest
        to. This is not a very good strategy, and completely ignores the features of
        the ghosts and the capsules; it is just designed to give you an example.
        """
        global bad_ghost_vec
        global previous_ghost_state


        ghost_states = observedState.getGhostStates() # states have getPosition() and getFeatures() methods
        ghost_features = map(lambda x : x.getFeatures(),ghost_states)
        if(len(ghost_features) != NUM_GHOSTS):
            print "unexpected number of ghosts" + str(len(ghost_features))

        bad_ghost_vec = self.updateBadGhost(observedState)
        bad_ghost = filter(lambda x: (bad_ghost_vec == x.getFeatures()).all(),ghost_states)[0]
        print ObservedState.getGhostQuadrant(observedState,bad_ghost)
        previous_ghost_state = ghost_states


        pacmanPosition = observedState.getPacmanPosition()


        legalActs = [a for a in observedState.getLegalPacmanActions()]
        ghost_dists = np.array([self.distancer.getDistance(pacmanPosition,gs.getPosition()) 
                              for gs in ghost_states])
        # find the closest ghost by sorting the distances
        closest_idx = sorted(zip(range(len(ghost_states)),ghost_dists), key=lambda t: t[1])[0][0]
        # take the action that minimizes distance to the current closest ghost
        best_action = Directions.STOP
        best_dist = -np.inf
        for la in legalActs:
            if la == Directions.STOP:
                continue
            successor_pos = Actions.getSuccessor(pacmanPosition,la)
            new_dist = self.distancer.getDistance(successor_pos,ghost_states[closest_idx].getPosition())
            if new_dist > best_dist:
                best_action = la
                best_dist = new_dist
        return best_action
    
class FuturePosAgent(BaseStudentAgent):
    def chooseAction(self, observedState):
        legalActs = [a for a in observedState.getLegalPacmanActions()]
        g_pos = map(lambda x : x.getPosition(),observedState.getGhostStates())
        print g_pos
#        print ObservedState.pacmanFuturePosition(observedState,[Directions.WEST])
#       print ObservedState.pacmanFuturePosition(observedState,[Directions.WEST,Directions.WEST,Directions.WEST])
        dir_lst = [Directions.NORTH,Directions.SOUTH,Directions.EAST,Directions.WEST,Directions.STOP]
        dir_dict = {Directions.NORTH:0,Directions.SOUTH:1,Directions.EAST:2,Directions.WEST:3,Directions.STOP:4}
        fst_g_pos = observedState.getGhostStates()[0].getPosition()
        response_lst = [ObservedState.ghostFuturePosition(observedState,0,[i]) for i in dir_lst]
        print response_lst
        act = random.choice(legalActs)
        print (act,response_lst[dir_dict[act]])
        return act                

class DataAgent(BaseStudentAgent):

    def chooseAction(self, observedState):
        pacmanPosition = observedState.getPacmanPosition()
        legalActs = [a for a in observedState.getLegalPacmanActions()]
        gcaps = observedState.getGoodCapsuleExamples()
        f = open(GOOD_CAPS_CSV, "w+")
        f.close()
        with open(GOOD_CAPS_CSV, 'wb') as csvfile:
            writer = csv.writer(csvfile, delimiter=' ',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
            for row in gcaps:
                writer.writerow(row)
        return random.choice(legalActs )



## Below is the class students need to rename and modify
'''

class ExampleTeamAgent(BaseStudentAgent):
    """
    An example TeamAgent. After renaming this agent so it is called <YourTeamName>Agent,
    (and also renaming it in registerInitialState() below), modify the behavior
    of this class so it does well in the pacman game!
    """
    
    def __init__(self, *args, **kwargs):
        """
        arguments given with the -a command line option will be passed here
        """
        pass # you probably won't need this, but just in case
    
    def registerInitialState(self, gameState):
        """
        Do any necessary initialization
        """
        # Here, you must replace "ExampleTeamAgent" with "<YourTeamName>Agent"
        super(ExampleTeamAgent, self).registerInitialState(gameState)
        
        # Here, you may do any necessary initialization, e.g., import some
        # parameters you've learned, as in the following commented out lines
        # learned_params = cPickle.load("myparams.pkl")
        # learned_params = np.load("myparams.npy")        
    
    def chooseAction(self, observedState):
        """
        Here, choose pacman's next action based on the current state of the game.
        This is where all the action happens.
        
        This silly pacman agent will move away from the ghost that it is closest
        to. This is not a very good strategy, and completely ignores the features of
        the ghosts and the capsules; it is just designed to give you an example.
        """
        pacmanPosition = observedState.getPacmanPosition()
        ghost_states = observedState.getGhostStates() # states have getPosition() and getFeatures() methods
        legalActs = [a for a in observedState.getLegalPacmanActions()]
        ghost_dists = np.array([self.distancer.getDistance(pacmanPosition,gs.getPosition()) 
                              for gs in ghost_states])
        # find the closest ghost by sorting the distances
        closest_idx = sorted(zip(range(len(ghost_states)),ghost_dists), key=lambda t: t[1])[0][0]
        # take the action that minimizes distance to the current closest ghost
        best_action = Directions.STOP
        best_dist = -np.inf
        for la in legalActs:
            if la == Directions.STOP:
                continue
            successor_pos = Actions.getSuccessor(pacmanPosition,la)
            new_dist = self.distancer.getDistance(successor_pos,ghost_states[closest_idx].getPosition())
            if new_dist > best_dist:
                best_action = la
                best_dist = new_dist
        return best_action
'''
