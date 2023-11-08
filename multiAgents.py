# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """



    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        
        #location as coordinates stored in a list
        newFoodList = newFood.asList()
        
        #distance of each food in the list
        foodDis = [manhattanDistance(newPos, food) for food in newFoodList]
        
        score = 0
        
        if(len(foodDis) > 0):
            minFood = min(foodDis)
        else:
            minFood = 1

        #score increases inversely to distance (reward for being closer to food)
        score += 1 / minFood
        
        return successorGameState.getScore() + score

        #util.raiseNotDefined()

def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"

        def isTerminalState(gameState, depth):
            return gameState.isWin() or gameState.isLose() or depth == 0

        def minValue(gameState, depth, ghostIndex):
            if isTerminalState(gameState, depth):
                return self.evaluationFunction(gameState)

            value = float("inf")
            for action in gameState.getLegalActions(ghostIndex):
                successor = gameState.generateSuccessor(ghostIndex, action)
                if ghostIndex == gameState.getNumAgents() - 1:
                    value = min(value, maxValue(successor, depth - 1))
                else:
                    value = min(value, minValue(successor, depth, ghostIndex + 1))
            return value

        def maxValue(gameState, depth):
            if isTerminalState(gameState, depth):
                return self.evaluationFunction(gameState)

            value = -float("inf")
            for action in gameState.getLegalActions(self.index):
                if action != Directions.STOP:
                    successor = gameState.generateSuccessor(self.index, action)
                    value = max(value, minValue(successor, depth, 1))
            return value

        # Initialize best action to the first legal action
        legalActions = gameState.getLegalActions(self.index)
        bestAction = Directions.STOP if Directions.STOP in legalActions else legalActions[0]
        maxUtility = -float("inf")

        # Evaluate the minimax utility for each legal action
        for action in legalActions:
            if action != Directions.STOP:
                successor = gameState.generateSuccessor(self.index, action)
                utility = minValue(successor, self.depth, 1)
                if utility > maxUtility:
                    maxUtility = utility
                    bestAction = action

        # Return the action with the highest utility
        return bestAction
            

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"

        
        scores = []
        
        a = -float("inf") #alpha
        b = float("inf") #beta

        for action in gameState.getLegalActions(0):
            newState = gameState.generateSuccessor(0, action)
            score = self.minValue(newState, self.depth, 1, a, b)
            scores.append((score, action))
            
            if max(scores)[0] > b:
                return max(scores)[1]

            a = max(max(scores)[0], a)

        return max(scores)[1]


    def maxValue(self, gameState, currentDepth, a, b):
      
        if gameState.isWin() or gameState.isLose() or currentDepth < 1:
            return self.evaluationFunction(gameState)
        
        actions = gameState.getLegalActions(0)
        
        scores = []
        
        for action in actions:
            scores.append(self.minValue(gameState.generateSuccessor(0, action), currentDepth, 1, a, b))
            maximum = max(scores)
            
            if maximum > b:
                return maximum
            
            a = max(maximum, a)

        if scores:
            return maximum
        else:
            return -float("inf")

    def minValue(self, gameState, currentDepth, ghostIdx, a, b):
      
        if gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)
        
        actions = gameState.getLegalActions(ghostIdx)
        
        scores = []
        
        for action in actions:
            ghostState = gameState.generateSuccessor(ghostIdx, action)

            if ghostIdx == gameState.getNumAgents() - 1:
                newVal = self.maxValue(ghostState, currentDepth - 1, a, b)
            else:
                newVal = self.minValue(ghostState, currentDepth, ghostIdx + 1, a, b)

            scores.append(newVal)
            minimum = min(scores)
            
            if minimum < a:
                return minimum

            b = min(b, minimum)

        if scores:
            return minimum
        else:
            return float("inf")
       
        #util.raiseNotDefined()

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        # util.raiseNotDefined()

        def isTerminalState(gameState, depth):
            return gameState.isWin() or gameState.isLose() or depth == 0

        # Expected value for the ghosts
        def getExpectedValue(gameState, depth, ghostIndex):
            if isTerminalState(gameState, depth):
                return self.evaluationFunction(gameState)

            totalValue = 0
            actionList = gameState.getLegalActions(ghostIndex)
            probability = 1.0 / len(actionList)
            nextGhostIndex = ghostIndex + 1 if ghostIndex < gameState.getNumAgents() - 1 else self.index

            for action in actionList:
                successor = gameState.generateSuccessor(ghostIndex, action)
                if ghostIndex < gameState.getNumAgents() - 1:
                    totalValue += probability * getExpectedValue(successor, depth, nextGhostIndex)
                else:
                    totalValue += probability * getMaxValue(successor, depth - 1)
            return totalValue

        # Max value for Pacman
        def getMaxValue(gameState, depth):
            if isTerminalState(gameState, depth):
                return self.evaluationFunction(gameState)

            value = -float("inf")
            for action in gameState.getLegalActions(self.index):
                if action != Directions.STOP:
                    successor = gameState.generateSuccessor(self.index, action)
                    value = max(value, getExpectedValue(successor, depth, 1))
            return value

        # Get the best action for Pacman
        legalActions = [action for action in gameState.getLegalActions(self.index) if action != Directions.STOP]
        bestAction = None
        maxUtility = -float("inf")

        for action in legalActions:
            successor = gameState.generateSuccessor(self.index, action)
            utility = getExpectedValue(successor, self.depth, 1)
            if utility > maxUtility:
                maxUtility = utility
                bestAction = action

        return bestAction if bestAction is not None else Directions.STOP

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: We decided to evaluate the function based on the food and score. The overall goal is to maximize the score and we can better evaluate performance with these two variables. The pacman should have the intention to maximize its score as well as obtain food which also inherently increases score. If pacman prioiritizes these factors then score should be maximized.
    """
    "*** YOUR CODE HERE ***"
    
    pacman = currentGameState.getPacmanPosition()    
    
    foodPos = currentGameState.getFood()
    foodList = foodPos.asList()
    
    #store distances from food
    foodDis = [manhattanDistance(pacman, food) for food in foodList]
    
    if(len(foodList) > 0) :
        minFood = min(foodDis)
    else:
        minFood = 1
        
    score = currentGameState.getScore()

    #the closer food is, it increases score by the inverse relationship to distance    
    return score + 1/minFood

    #util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction
