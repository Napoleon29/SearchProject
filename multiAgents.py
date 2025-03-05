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
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

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

        score = successorGameState.getScore()
        newFood = successorGameState.getFood().asList()

        if newFood:
            closest_food_dist = min(util.manhattanDistance(newPos, food) for food in newFood)
            score += 10.0 / (1.0 + closest_food_dist)  # incentivizes closest food consumption
            score += 10 * (1.0 / (1.0 + len(newFood)))  # incentivizes removing all food to avoid pacman getting stuck

        for ghost in range(len(newGhostStates)):
            ghostState = newGhostStates[ghost]
            scaredTime = newScaredTimes[ghost]
            ghostPos = ghostState.getPosition()
            ghostDist = util.manhattanDistance(newPos, ghostPos)

            if scaredTime > 0:
                score += 200.0 / (1.0 + ghostDist)  # heavily incentivizes eating scared ghosts
            else:
                if ghostDist < 3:
                    score -= 200 / (1.0 + ghostDist)  # heavily incentivizes avoiding ghosts

        return score  # returns score that considers closest food, ghost location, ghost status, and total food left


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

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        self.index = 0  # Pacman is always agent index 0
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

        def minimax(state, depth, agentIndex):

            if depth == self.depth or state.isWin() or state.isLose():  # returns eval score when depth is reached or game won or lost
                return self.evaluationFunction(state)

            numAgents = state.getNumAgents()

            pacmanTurn = (agentIndex == 0)  # sets True if Pacman's turn

            nextAgent = (agentIndex + 1) % numAgents  # Determine whether we need to increase depth bases on whose turn
            if nextAgent == 0:
                nextDepth = depth + 1
            else:
                nextDepth = depth

            actions = state.getLegalActions(agentIndex)
            if not actions:  # prevents pacman getting stuck
                return self.evaluationFunction(state)

            scores = []
            for action in actions:  # begins recursive loop of minimax to expand tree
                successor = state.generateSuccessor(agentIndex, action)
                score = minimax(successor, nextDepth, nextAgent)
                scores.append(score)

            if pacmanTurn:
                return max(scores)
            else:
                return min(scores)

        actions = gameState.getLegalActions(0)  # find moves from start
        bestScore = -1000000
        bestAction = None

        for action in actions:
            successorState = gameState.generateSuccessor(0, action)
            score = minimax(successorState, 0, 1)  # begins recursive call

            if score > bestScore:
                bestScore = score
                bestAction = action

        return bestAction

        # util.raiseNotDefined()


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        def alphaBeta(state, depth, agentIndex, alpha, beta):
            # Check for terminal state or if reached the maximum depth.
            if depth == self.depth or state.isWin() or state.isLose():
                return self.evaluationFunction(state)

            numAgents = state.getNumAgents()
            nextAgent = (agentIndex + 1) % numAgents
            nextDepth = depth + 1 if nextAgent == 0 else depth

            legalActions = state.getLegalActions(agentIndex)
            # If no legal actions, evaluate the state.
            if not legalActions:
                return self.evaluationFunction(state)

            # Pacman's turn: Maximizing player.
            if agentIndex == 0:
                value = float("-inf")
                for action in legalActions:
                    successor = state.generateSuccessor(agentIndex, action)
                    value = max(value, alphaBeta(successor, nextDepth, nextAgent, alpha, beta))
                    # Prune if value strictly exceeds beta.
                    if value > beta:
                        return value
                    alpha = max(alpha, value)
                return value
            # Ghosts' turn: Minimizing players.
            else:
                value = float("inf")
                for action in legalActions:
                    successor = state.generateSuccessor(agentIndex, action)
                    value = min(value, alphaBeta(successor, nextDepth, nextAgent, alpha, beta))
                    # Prune if value is strictly less than alpha.
                    if value < alpha:
                        return value
                    beta = min(beta, value)
                return value

        # At the root (Pacman's turn), choose action with highest minimax value.
        legalActions = gameState.getLegalActions(0)
        bestAction = None
        bestValue = float("-inf")
        alpha = float("-inf")
        beta = float("inf")

        for action in legalActions:
            successor = gameState.generateSuccessor(0, action)
            value = alphaBeta(successor, 0, 1, alpha, beta)
            if value > bestValue:
                bestValue = value
                bestAction = action
            alpha = max(alpha, bestValue)
        return bestAction


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
        return self.expectimax(gameState, 0, self.depth)[1]

    # Expectimax function
    # Arguments: game_state, agent_index, depth
    # Originally used self.depth, but it was causing depth to be decremented for every
    # ghost's turn, so I changed I added it as a variable to pass through
    def expectimax(self, game_state, a_index, depth2_electric_boogaloo):
        # Check for a terminal state or if the maximum depth has been reached
        if game_state.isWin() or game_state.isLose() or depth2_electric_boogaloo == 0:
            return self.evaluationFunction(game_state), "Stop"

        # Get the number of agents in the game
        numAgents = game_state.getNumAgents()
        a_index = a_index % numAgents

        # Decrease the depth if it is the last ghost's turn
        if a_index == numAgents - 1:
            depth2_electric_boogaloo -= 1

        # We want to maximize the score if it is Pacman's turn
        # and minimize the score if it is a ghost's turn
        if a_index == 0:
            # run through actions and get the max value, saving the action alongside the cost
            actions = []
            for action in game_state.getLegalActions(a_index):
                successor = game_state.generateSuccessor(a_index, action)
                value = self.expectimax(successor, a_index + 1, depth2_electric_boogaloo)[0]
                actions.append((value, action))

            if not actions:
                return self.evaluationFunction(game_state), None

            return max(actions)
        else:
            # Run through actions and get the average value
            total = 0
            legalActions = game_state.getLegalActions(a_index)
            for action in game_state.getLegalActions(a_index):
                successor = game_state.generateSuccessor(a_index, action)
                value = self.expectimax(successor, a_index + 1, depth2_electric_boogaloo)[0]
                total += value
            # Return the average value
            return (total / len(legalActions),)


def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    # So my thoughts here are to calculate a cumulative score based on the following:
    # - pacman
    # - food
    # - ghost(s) - starting off with closest ghost
    # - pills that allow pacman to eat ghosts
    # My plan is to calculate them all, and then sum them up to get the final score
    # I may also weigh them depending on some testing

    # First, fetch all the information we need about the game state
    pacmanPos = currentGameState.getPacmanPosition()
    food = currentGameState.getFood().asList()
    ghosts = currentGameState.getGhostStates()
    scaredTimes = [ghostState.scaredTimer for ghostState in ghosts]
    ghostDist = [manhattanDistance(pacmanPos, ghost.getPosition()) for ghost in ghosts]
    capsules = currentGameState.getCapsules()
    score = currentGameState.getScore()

    # Calculate the closest food
    if food:
        closest_food_dist = min([manhattanDistance(pacmanPos, foodPos) for foodPos in food])
        # Lets make 2 calculations for the food

        # First, lets incentive eating the closest food by giving points based on the distance
        # To the closest food
        score += 10.0 / (1.0 + closest_food_dist)

        # Second, lets incentive eating all the food by giving points based on the amount of food left
        # As the quantity of food decreases, the points will increase
        score += 10 * (1.0 / (1.0 + len(food)))

    # Calculate the closest ghost
    for ghost in range(len(ghosts)):
        ghostState = ghosts[ghost]
        scaredTime = scaredTimes[ghost]
        ghostPos = ghostState.getPosition()
        ghostDist = manhattanDistance(pacmanPos, ghostPos)

        # If the ghost is scared, we want to heavily incentive eating the ghost
        if scaredTime > 0:
            score += 200.0 / (1.0 + ghostDist)
        else:
            # If the ghost is not scared, we want to heavily incentive avoiding the ghost
            if ghostDist < 3:
                score -= 200 / (1.0 + ghostDist)

    # Calculate the closest capsule
    if capsules:
        closest_capsule_dist = min([manhattanDistance(pacmanPos, capsule) for capsule in capsules])
        score += 50.0 / (1.0 + closest_capsule_dist)

    return score


# Abbreviation
better = betterEvaluationFunction
