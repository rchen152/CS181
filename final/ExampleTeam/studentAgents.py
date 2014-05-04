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

'''badGhostInfo = (direction,distance,isScared)
goodGhostInfo = (direction,distance)
(NOT USED) wallInfo = [isPresent,isPresent,isPresent,isPresent]
goodCapInfo = (direction,distance)
dim = (numDirs*bgRange*2 + 1, numDirs*ggRange + 1, numDirs*capRange + 1)'''

num_states = 3861
num_actions = 4
sa_ds = pickle.load(open('ExampleTeam/pickled_sa.p','r'))

dir_dict = {Directions.NORTH:0,Directions.SOUTH:1,Directions.EAST:2,Directions.WEST:3,Directions.STOP:4}

#previous_state = 0
#previous_action = Directions.NORTH
#previous_score = 0

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

        # Compute bad ghost state
        bgInd = -1
        if badGhost:
            # Get distance to bad ghost
            bgPos = badGhost.getPosition()
            bgDir = Directions.STOP
            bgDist = self.distancer.getDistance(pacPos, bgPos)
            bgScared = 0
            # If bad ghost in range, get closest direction to it
            if 0 < bgDist and bgDist <= BG_RANGE:
                posDirs = observedState.getLegalPacmanActions()
                for d in posDirs:
                    nextPos = observedState.pacmanFuturePosition([d])
                    if self.distancer.getDistance(nextPos, bgPos) < bgDist:
                        bgDir = d
                        # Check whether the bad ghost is scared
                        scared = observedState.scaredGhostPresent()
                        if scared:
                            bgScared = 1
                        # Get a number from the state
                        bgInd = ((2*BG_RANGE*dirInd(bgDir)) +
                                 (2*(bgDist - 1)) + bgScared)
                        break
        # Compute good ghost state
        ggInd = -1
        # Get good ghosts in range
        goodGhosts = [g for g in observedState.getGhostStates()
                      if not badGhost
                      or not (g.getFeatures() == badGhost.getFeatures()).all()]
        goodDists = [self.distancer.getDistance(pacPos, g.getPosition())
                     for g in goodGhosts]
        goodGhosts = [(goodGhosts[i],goodDists[i])
                       for i in range(len(goodGhosts))
                       if 0 < goodDists[i] and goodDists[i] <= GG_RANGE]
        if len(goodGhosts) > 0:
            # Get distance to juiciest ghost
            jness = [AVG_CLASS_JUICE[c] for c in
                     cfy.ghostClassify([g[0].getFeatures()
                                        for g in goodGhosts])]
            juicyInd = jness.index(max(jness))
            juicyPos = goodGhosts[juicyInd][0].getPosition()
            juicyDist = goodGhosts[juicyInd][1]
            # Get direction to juiciest ghost
            juicyDir = Directions.STOP
            posDirs = observedState.getLegalPacmanActions()
            for d in posDirs:
                nextPos = observedState.pacmanFuturePosition([d])
                if self.distancer.getDistance(nextPos, juicyPos) < juicyDist:
                    juicyDir = d
                    # Get a number from the state
                    ggInd = (GG_RANGE * dirInd(juicyDir)) + (juicyDist-1)
                    break       
 
        # Compute good capsule state
        gcInd = -1
        # Get all capsules
        caps = observedState.getCapsuleData()
        capPosLst = [c[0] for c in caps]
        capDists = [self.distancer.getDistance(pacPos, cp) for cp in capPosLst]
        # Classify
        capClasses = cfy.capClassify([c[1] for c in caps])
        # Filter by good capsules in range
        caps = [(capDists[i],capPosLst[i]) for i in range(len(caps))
                if 0 < capDists[i] and capDists[i] <= CAP_RANGE
                and capClasses[i]]
        if len(caps) > 0:
            # Get closest good capsule
            gcDist,gcPos = min(caps)
            # Compute direction to closest good capsule
            gcDir = Directions.STOP
            posDirs = observedState.getLegalPacmanActions()
            for d in posDirs:
                nextPos = observedState.pacmanFuturePosition([d])
                if self.distancer.getDistance(nextPos, gcPos) < gcDist:
                    gcDir = d
                    # Get a number from the state
                    gcInd = (CAP_RANGE * dirInd(gcDir)) + (gcDist-1)
                    break
        
        # Get overall state
        bgInd += 1
        ggInd += 1
        gcInd += 1
        numGGStates = NUM_DIRS*GG_RANGE + 1
        numGCStates = NUM_DIRS*CAP_RANGE + 1
        return ((bgInd * numGGStates * numGCStates) +
                (ggInd * numGCStates) + gcInd)

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
                return None
            else:
                return bgList[0]

    def chooseAction(self, observedState):
        global badGhost
        global prevGhostStates

        ghostStates = observedState.getGhostStates()
        if len(ghostStates) != NUM_GHOSTS:
            print 'Warning: unexpected no. of ghosts' + str(len(ghostStates))
        badGhost = self.updateBadGhost(observedState)
        print 'Bad quad: ' + str(ObservedState.getGhostQuadrant(
                observedState,badGhost))
        prevGhostStates = ghostStates
        return random.choice(observedState.getLegalPacmanActions())
    
class CollectAgent(BaseStudentAgent):

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

        # Compute bad ghost state
        bgInd = -1
        if badGhost:
            # Get distance to bad ghost
            bgPos = badGhost.getPosition()
            bgDir = Directions.STOP
            bgDist = self.distancer.getDistance(pacPos, bgPos)
            bgScared = 0
            # If bad ghost in range, get closest direction to it
            if 0 < bgDist and bgDist <= BG_RANGE:
                posDirs = observedState.getLegalPacmanActions()
                for d in posDirs:
                    nextPos = observedState.pacmanFuturePosition([d])
                    if self.distancer.getDistance(nextPos, bgPos) < bgDist:
                        bgDir = d
                        break
                # Check whether the bad ghost is scared
                scared = observedState.scaredGhostPresent()
                if scared:
                    bgScared = 1
                # Get a number from the state
                bgInd = ((2*BG_RANGE*dirInd(bgDir)) +
                         (2*(bgDist - 1)) + bgScared)

        # Compute good ghost state
        ggInd = -1
        # Get good ghosts in range
        goodGhosts = [g for g in observedState.getGhostStates()
                      if not badGhost
                      or not (g.getFeatures() == badGhost.getFeatures()).all()]
        goodDists = [self.distancer.getDistance(pacPos, g.getPosition())
                     for g in goodGhosts]
        goodGhosts = [(goodGhosts[i],goodDists[i])
                       for i in range(len(goodGhosts))
                       if 0 < goodDists[i] and goodDists[i] <= GG_RANGE]
        if len(goodGhosts) > 0:
            # Get distance to juiciest ghost
            jness = [AVG_CLASS_JUICE[c] for c in
                     cfy.ghostClassify([g[0].getFeatures()
                                        for g in goodGhosts])]
            juicyInd = jness.index(max(jness))
            juicyPos = goodGhosts[juicyInd][0].getPosition()
            juicyDist = goodGhosts[juicyInd][1]
            # Get direction to juiciest ghost
            juicyDir = Directions.STOP
            posDirs = observedState.getLegalPacmanActions()
            for d in posDirs:
                nextPos = observedState.pacmanFuturePosition([d])
                if self.distancer.getDistance(nextPos, juicyPos) < juicyDist:
                    juicyDir = d
                    break
            # Get a number from the state
            ggInd = (GG_RANGE * dirInd(juicyDir)) + (juicyDist-1)
        
        # Compute good capsule state
        gcInd = -1
        # Get all capsules
        caps = observedState.getCapsuleData()
        capPosLst = [c[0] for c in caps]
        capDists = [self.distancer.getDistance(pacPos, cp) for cp in capPosLst]
        # Classify
        capClasses = cfy.capClassify([c[1] for c in caps])
        # Filter by good capsules in range
        caps = [(capDists[i],capPosLst[i]) for i in range(len(caps))
                if 0 < capDists[i] and capDists[i] <= CAP_RANGE
                and capClasses[i]]
        if len(caps) > 0:
            # Get closest good capsule
            gcDist,gcPos = min(caps)
            # Compute direction to closest good capsule
            gcDir = Directions.STOP
            posDirs = observedState.getLegalPacmanActions()
            for d in posDirs:
                nextPos = observedState.pacmanFuturePosition([d])
                if self.distancer.getDistance(nextPos, gcPos) < gcDist:
                    gcDir = d
                    break
            # Get a number from the state
            gcInd = (CAP_RANGE * dirInd(gcDir)) + (gcDist-1)
        
        # Get overall state
        bgInd += 1
        ggInd += 1
        gcInd += 1
        numGGStates = NUM_DIRS*GG_RANGE + 1
        numGCStates = NUM_DIRS*CAP_RANGE + 1
        return ((bgInd * numGGStates * numGCStates) +
                (ggInd * numGCStates) + gcInd)

    def explore(self,observedState):
        global sa_ds
        global previous_state
        global previous_action
        global old_score

        old_s = self.getStateNum(previous_state)
        new_s = self.getStateNum(observedState)
        
        
        if new_s in sa_ds[old_s * 4 + dir_dict[previous_action]]:
            sa_ds[old_s * 4 + dir_dict[previous_action]][new_s] += 1
        else:
            sa_ds[old_s * 4 + dir_dict[previous_action]][new_s] = 1

        sa_ds[old_s * 4 + dir_dict[previous_action]]['count']+=1

        sa_ds[old_s * 4 + dir_dict[previous_action]]['total_reward']+= (old_score - observedState.getScore())
        
    def chooseAction(self,observedState):
        global sa_ds
        global previous_state
        global previous_action
        global old_score
        
        legalActs = [a for a in observedState.getLegalPacmanActions()]
        if (legalActs == [Directions.STOP]):
            print "error you are trapped by four walls"
            previous_state = observedState
            old_score = observedState.getScore()
            return Directions.STOP

        if (observedState.getNumMovesLeft == 1):
            pickle.dump(sa_ds,open("pickled_sa.p","w"))

        if (observedState.getNumMovesLeft() != GAME_LEN):
            self.explore(observedState)        
            
        fil_legal = filter(lambda x: x != Directions.STOP ,legalActs)
        s = self.getStateNum(observedState)

        print sa_ds
        count_lst = [None]*len(fil_legal)
        for i in range(len(fil_legal)):
            dir_num = dir_dict[fil_legal[i]]

            count_lst[i] = sa_ds[4*s + dir_num]['count'] 
                         
        previous_state = observedState
        old_score = observedState.getScore()

        state = self.getStateNum(observedState)
        print 'State: ' + str(state)
        if state != 0:
            act = fil_legal[count_lst.index(min(count_lst))]
            previous_action = act
            return act
        minDist = None
        minGhost = None
        for g in ghostStates:
            if not (g.getFeatures() == badGhost.getFeatures()).all():
                dist = self.distancer.getDistance(
                    observedState.getPacmanPosition(), g.getPosition())
                if not minGhost or dist < minDist:
                    minDist = dist
                    minGhost = g
        posDirs = observedState.getLegalPacmanActions()
        for d in posDirs:
            nextPos = observedState.pacmanFuturePosition([d])
            if self.distancer.getDistance(nextPos,
                                          minGhost.getPosition()) < minDist:
                return d
        return random.choice([d for d in observedState.getLegalPacmanActions()
                              if not d == Directions.STOP])
        
class FuturePosAgent(BaseStudentAgent):
    def chooseAction(self, observedState):
        legalActs = [a for a in observedState.getLegalPacmanActions()]
#        print legalActs
        act = random.choice(legalActs)
#        print (act,response_lst[dir_dict[act]])

#        g_pos = map(lambda x : x.getPosition(),observedState.getGhostStates())
#        p_pos = observedState.getPacmanPosition()
#        if(filter(lambda x: x == p_pos,g_pos) != []):
#            print g_pos
#            print p_pos
        print observedState.getNumMovesLeft()
        return act
'''        print ObservedState.pacmanFuturePosition(observedState,[Directions.WEST])
       print ObservedState.pacmanFuturePosition(observedState,[Directions.WEST,Directions.WEST,Directions.WEST])
        dir_lst = [Directions.NORTH,Directions.SOUTH,Directions.EAST,Directions.WEST,Directions.STOP]

        fst_g_pos = observedState.getGhostStates()[0].getPosition()
        response_lst = [ObservedState.ghostFuturePosition(observedState,0,[i]) for i in dir_lst]
        print response_lst'''
        



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
            if new_dist < 2: print new_dist
        return best_action
'''
