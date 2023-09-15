import random
import sys
import numpy as np
import numpy.typing as npt
import math

from hw2.utils import utility, successors, Node, Tree, GameStrategy


"""
Alpha Beta Search
"""


def max_value(state: npt.ArrayLike, alpha: float, beta: float, k: int):
    """Find the max value given state and return the utility and the game board
    state after the move. Please see Lecture 6 for pseudocode.

    Args:
        state (np.ndarray): the state of the game board, mxn numpy array.
        alpha (float): the alpha value
        beta (float): the beta value
        k (int): the number of consecutive marks
    Returns:
        tuple[float, np.ndarray]: utility value and the board after the move
    """

    # TODO:
    if( not (utility(state, k) is  None)):
        return utility(state, k), state #score or eval value of grid 
    v = (-1)*(sys.maxsize) # check 
    max_state = None 
    for i in successors(state, 'X') :
        v2, new_state  = min_value(i, alpha, beta, k)
        if(v2 > v):
            v = v2
            max_state = np.copy(i)
            alpha = max(alpha, v)
        if (v>=beta):
            return v, max_state
    return v, max_state


def min_value(state: npt.ArrayLike, alpha: float, beta: float, k: int):
    """Find the min value given state and return the utility and the game board
    state after the move. Please see Lecture 6 for pseudocode.

    Args:
        state (np.ndarray): the state of the game board, mxn numpy array.
        alpha (float): the alpha value
        beta (float): the beta value
        k (int): the number of consecutive marks
    Returns:
        tuple[float, np.ndarray]: utility value and the board after the move
    """

    # TODO:
    if( not (utility(state, k) is  None)):
        return utility(state, k), state #score or eval value of grid 
    v = sys.maxsize # check 
    min_state = None 
    for i in successors(state, 'O') :
        v2, new_state  = max_value(i, alpha, beta, k)
        if(v2 < v):
            v = v2
            min_state = np.copy(i)
            beta = min(beta, v)
        if (v <= alpha):
            return v, min_state
    return v, min_state


"""
Monte Carlo Tree Search
"""

def ucb_value(node:"Node", alpha: float):
    if node.N == 0:
        return 0
    return (node.w / node.N) + alpha * (math.sqrt(math.log(node.parent.N)/node.N))

def select(tree: "Tree", state: npt.ArrayLike, k: int, alpha: float):
    """Starting from state, find a terminal node or node with unexpanded
    children. If all children of a node are in tree, move to the one with the
    highest UCT value.

    Args:
        tree (utils.Tree): the search tree
        state (np.ndarray): the game board state
        k (int): the number of consecutive marks
        alpha (float): exploration parameter
    Returns:
        np.ndarray: the game board state
    """

    # TODO:
    while((utility(state, k) is None)):
        v = -1*sys.maxsize
        next_state = None
        flag = False
        for i in successors(state, tree.get(state).player):
            if (tree.get(i) is None): ##even if one child is not visited or completed unvisited
                return state
            # print("Tree :", tree.get(i))    
            # print("print", tree.get(i).parent)
            if (not (tree.get(i).parent.state == state).all()):
                continue  
            flag = True
            u = ucb_value(tree.get(i), alpha)
            if v < u :
                v = u
                next_state = np.copy(i) 
        if not flag:
            return state
        state = next_state
        # print("state : ", state)
    return state


def expand(tree: "Tree", state: npt.ArrayLike, k: int):
    """Add a child node of state into the tree if it's not terminal and return
    tree and new state, or return current tree and state if it is terminal.

    Args:
        tree (utils.Tree): the search tree
        state (np.ndarray): the game board state
        k (int): the number of consecutive marks
    Returns:
        tuple[utils.Tree, np.ndarray]: the tree and the game state
    """

    # TODO:
    if utility(state, k) is None:
        next_player = "O" if tree.get(state).player == "X" else "X"
        next_state = None 
        i = 0
        flag = False
        while (i<9):
            next_state = random.choice(successors(state, tree.get(state).player))
            if(tree.get(next_state) is None):
                flag = True
                break
            i+=1
        if not flag :
            return tree, state
        tree.add(Node(next_state, tree.get(state), next_player, 0, 1)) 
        return tree, next_state

    return tree, state


def simulate(state: npt.ArrayLike, player: str, k: int):
    """Run one game rollout from state to a terminal state using random
    playout policy and return the numerical utility of the result.

    Args:
        state (np.ndarray): the game board state
        player (string): the player, `O` or `X`
        k (int): the number of consecutive marks
    Returns:
        float: the utility
    """

    # TODO:
    current_player = player
    while (utility(state, k) is None):
        state = random.choice(successors(state, current_player))
        current_player = "O" if current_player == "X" else "X"
    return utility(state, k)


def backprop(tree: "Tree", state: npt.ArrayLike, result: float):
    """Backpropagate result from state up to the root.
    All nodes on path have N, number of plays, incremented by 1.
    If result is a win for a node's parent player, w is incremented by 1.
    If result is a draw, w is incremented by 0.5 for all nodes.

    Args:
        tree (utils.Tree): the search tree
        state (np.ndarray): the game board state
        result (float): the result / utility value

    Returns:
        utils.Tree: the game tree
    """

    # TODO:
    while(1):
        parent = tree.get(state).parent
        if parent is None:
            return tree
        if result == 1:
            tree.get(parent.state).w += 1
        elif result == 0:
            tree.get(parent.state).w += 0.5
        tree.get(parent.state).N += 1
        state = parent.state
    return tree


# ******************************************************************************
# ****************************** ASSIGNMENT ENDS *******************************
# ******************************************************************************


def MCTS(state: npt.ArrayLike, player: str, k: int, rollouts: int, alpha: float):
    # MCTS main loop: Execute MCTS steps rollouts number of times
    # Then return successor with highest number of rollouts
    tree = Tree(Node(state, None, player, 0, 1))

    print("*"*40, "player : ", player)

    for i in range(rollouts):
        print("#"*30)
        print(tree.get(state))
        leaf = select(tree, state, k, alpha)
        print("Leaf: ", leaf)
        tree, new = expand(tree, leaf, k)
        print("New state: ", new)
        result = simulate(new, tree.get(new).player, k)
        print("Result: ", result)
        tree = backprop(tree, new, result)

    nxt = None
    plays = 0

    for s in successors(state, tree.get(state).player):
        if tree.get(s).N > plays:
            plays = tree.get(s).N
            nxt = s

    return nxt


def ABS(state: npt.ArrayLike, player: str, k: int):
    # ABS main loop: Execute alpha-beta search
    # X is maximizing player, O is minimizing player
    # Then return best move for the given player
    if player == "X":
        value, move = max_value(state, -float("inf"), float("inf"), k)
    else:
        value, move = min_value(state, -float("inf"), float("inf"), k)

    return value, move


def game_loop(
    state: npt.ArrayLike,
    player: str,
    k: int,
    Xstrat: GameStrategy = GameStrategy.RANDOM,
    Ostrat: GameStrategy = GameStrategy.RANDOM,
    rollouts: int = 0,
    mcts_alpha: float = 0.01,
    print_result: bool = False,
):
    # Plays the game from state to terminal
    # If random_opponent, opponent of player plays randomly, else same strategy as player
    # rollouts and alpha for MCTS; if rollouts is 0, ABS is invoked instead


    current = player
    while utility(state, k) is None:
        # state = np.array([['X','O'],['O','.']])
        if current == "X":
            strategy = Xstrat
        else:
            strategy = Ostrat

        if strategy == GameStrategy.RANDOM:
            state = random.choice(successors(state, current))
        elif strategy == GameStrategy.ABS:
            _, state = ABS(state, current, k)
        else:
            state = MCTS(state, current, k, rollouts, mcts_alpha)

        current = "O" if current == "X" else "X"

        if print_result:
            print(state)

    return utility(state, k)
