from distanceCalculator import Distancer
from game import Actions
from game import Directions
from observedState import ObservedState
import numpy as np
import csv
from util import random, manhattanDistance, Counter, chooseFromDistribution
import pickle
import classify

GAME_LEN = 1000
BAD_QUAD = 4
NUM_GHOSTS = 4

RANGE = 5
NUM_DIRS = 4
NUM_MOVES = NUM_DIRS + 1

badGhost = None
prevGhostStates = []

'''badGhostInfo = (direction,distance,isScared)
goodGhostInfo = (direction,distance)
(NOT USED) wallInfo = [isPresent,isPresent,isPresent,isPresent]
goodCapInfo = [(direction,distance),(direction,distance)]
size = numDirs*range*2 + 1, numDirs*range + 1, (numDirs*range+1)^2, numMoves'''

bg = 0
gg = 1
#w  = 2
gc = 2

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
        
        # TODO change the path to the pickled decision tree
        fTree = open('train/tree.pkl', 'r')
        ghost_params = pickle.load(fTree)
        fTree.close()

    def getStateNum(self, observedState):
        def dirInd(d):
            if d == Directions.STOP:
                return 4
            elif d == Directions.NORTH:
                return 3
            elif d == Directions.SOUTH:
                return 2
            elif d == Directions.EAST:
                return 1
            else:
                return 0

        pacPos = observedState.getPacmanPosition()

        bgInd = -1
        if badGhost:
            bgPos = badGhost.getPosition()
            bgDir = Directions.STOP
            bgDist = self.distancer.getDistance(pacPos, bgPos)
            bgScared = 0
            if 0 < bgDist and bgDist <= RANGE:
                posDirs = observedState.getLegalPacmanActions()
                for d in posDirs:
                    nextPos = observedState.pacmanFuturePosition([d])
                    if self.distancer.getDistance(nextPos, bgPos) < bgDist:
                        bgDir = d
                        break
            else:
                bgDist = 0
            scared = observedState.scaredGhostPresent()
            if scared:
                bgScared = 1
            if bgDist == 0:
                bgInd = -1
            else:
                bgInd = (RANGE * dirInd(bgDir)) + (2 * (bgDist - 1)) + bgScared


    def posBadGhosts(self, ghostState, observedState):
        return [g for g in ghostState if ObservedState.getGhostQuadrant(
                observedState,g) == BAD_QUAD]

    def updateBadGhost(self, observedState):
        global badGhost
        global prevGhostStates
        
        ghostStates = observedState.getGhostStates()
        posBadGhosts = self.posBadGhosts(ghostStates,observedState)
        numPosGhosts = len(posBadGhosts)

        if(GAME_LEN == observedState.getNumMovesLeft()):
            if numPosGhosts != 1:
                print 'Error: wrong number of bad ghosts'
                return None
            else:
                return posBadGhosts[0]

        bgList = [g for g in ghostStates
                  if (g.getFeatures() == badGhost.getFeatures()).all()]
        if not bgList:
            if numPosGhosts < 1:
                print 'Error: no quad 4 ghosts'
                return None
            elif numPosGhosts == 1:
                return posBadGhosts[0]
            else:
                bGCandidates = [g for g in posBadGhosts if not
                                [p for p in prevGhostStates if
                                 (g.getFeatures() == p.getFeatures()).all()]]
                if len(bGCandidates) != 1:
                    print 'Error: not exactly one ghost regenerated in quad 4'
                    return None
                else:
                    return bGCandidates[0]
        else:
            if len(bgList) > 1:
                print 'Error: multiple identical bad ghosts'
            else:
                return bgList[0]
    def chooseAction(self, observedState):
        global badGhost
        global prevGhostStates

        self.getStateNum(observedState)

        ghostStates = observedState.getGhostStates()
        if len(ghostStates) != NUM_GHOSTS:
            print 'Warning: unexpected no. of ghosts' + str(len(ghostStates))
        badGhost = self.updateBadGhost(observedState)
        print ObservedState.getGhostQuadrant(observedState,badGhost)
        prevGhostStates = ghostStates

        legalActs = observedState.getLegalPacmanActions()
        return random.choice(legalActs)

class DataAgent(BaseStudentAgent):

    def chooseAction(self, observedState):
        legalActs = observedState.getLegalPacmanActions()
        return random.choice(legalActs)

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
