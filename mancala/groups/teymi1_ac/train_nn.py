import numpy as np
import torch
from torch import Tensor
from torch.autograd import Variable
import array
import time

from typing import Tuple

from mancala.groups.group_random.action import action as random_action
from mancala.groups.minmax.action import action as minimax_action
from mancala.game import initial_board, legal_actions, is_finished, play_turn, winner, copy_board, board_repr, flip_board, game, Board

device = 'cpu'

# max_beans is the maximum number of beans in a given pit that we can represent in the encoded form.
# Higher bean values will be rounded down. This reduces the size of the size of the network significantly
# without sacrificing much performance, since high bean values occur very rarely
max_beans = 20
# nh is the number of hidden layers
nh = 10
# nx needs to match the length of the encoded form
nx = (max_beans*12+1)

# The weights of the neural network
w1 = Variable(0.1*torch.randn(nh,nx, device = device, dtype=torch.float), requires_grad = True)
b1 = Variable(torch.zeros((nh,1), device = device, dtype=torch.float), requires_grad = True)
w2 = Variable(0.1*torch.randn(1,nh, device = device, dtype=torch.float), requires_grad = True)
b2 = Variable(torch.zeros((1,1), device = device, dtype=torch.float), requires_grad = True)

# Game-related helper functions

def is_finished(board: Board) -> bool:
    """
    Returns true if the game is over, or if a player has enough beans to be guaranteed to win the game.
    Use this instead of the "is_finished" function from game.py if you want to terminate the training episodes
    faster, since you can stop if you know which player is going to win
    """
    AREA0 = slice(6)
    AREA1 = slice(7, 13)
    return (sum(board[AREA0]) == 0 or sum(board[AREA1]) == 0) or board[6] > 24 or board[13] > 24

def game_result(w):
    """
    Corrects the reward from the "winner" function in game.py
    Winnings gets 1 point, losing 0 points, drawing 0.5 points
    """
    if w == 0:
        return 1
    elif w == 1:
        return 0
    else:
        return 0.5

def flip_actions(legal_actions):
    """
    Returns the ids of these legal actions from the perspective of the other player
    """
    return tuple((i + 7) % 14 for i in legal_actions)


# Neural network and gradient-related functions

def evaluate(board):
    """
    Returns the value of the board from the perspective of the current player, based on the neural network.
    """
    xa = np.zeros((1, nx))
    xa[0,:] = encode(board)
    x = Variable(torch.tensor(xa.transpose(), dtype=torch.float, device=device))
    h = torch.mm(w1, x) + b1
    h_sigmoid = h.tanh()  # squash this with a sigmoid function
    y = torch.mm(w2, h_sigmoid) + b2  # multiply with the output weights w2 and add bias
    va = y.sigmoid().detach().cpu().numpy().flatten()
    return va[0]

def initialize_grads():
    """
    Helper function for initializing the gradients with values of zero
    """
    return (torch.zeros(w1.size(), device=device, dtype=torch.float), torch.zeros(b1.size(), device=device, dtype=torch.float),
            torch.zeros(w2.size(), device=device, dtype=torch.float), torch.zeros(b2.size(), device=device, dtype=torch.float))

def updateDuring(phi: array.array, player, alpha: float, gamma: float, lam: float, target: float, Z_w1, Z_b1, Z_w2, Z_b2: Tensor):
    """
    Borrowed from example implementation in C3 Colab file
    "target" is the estimated value of the board "phi"
    We always assume that the board faces the current player
    """
    # zero the gradients
    xold = Variable(torch.tensor(phi.reshape((len(phi), 1)), dtype=torch.float, device=device))

    h = torch.mm(w1,xold) + b1 # matrix-multiply x with input weight w1 and add bias
    h_sigmoid = h.tanh() # squash this with a sigmoid function
    y = torch.mm(w2,h_sigmoid) + b2 # multiply with the output weights w2 and add bias
    y_sigmoid = y.sigmoid() # squash the output
    # now compute all gradients
    y_sigmoid.backward()
    # update the eligibility traces using the gradients
    Z_w1 = gamma * lam * Z_w1 + w1.grad.data
    Z_b1 = gamma * lam * Z_b1 + b1.grad.data
    Z_w2 = gamma * lam * Z_w2 + w2.grad.data
    Z_b2 = gamma * lam * Z_b2 + b2.grad.data
    w1.grad.data.zero_()
    b1.grad.data.zero_()
    w2.grad.data.zero_()
    b2.grad.data.zero_()
    # zero the gradients
    # perform now the update for the weights
    delta = 0 + gamma * target - y_sigmoid.detach() # this is the usual TD error
    delta = torch.tensor(delta, dtype = torch.float, device = device)
    w1.data = w1.data + alpha * delta * Z_w1 # may want to use different alpha for different layers!
    b1.data = b1.data + alpha * delta * Z_b1
    w2.data = w2.data + alpha * delta * Z_w2
    b2.data = b2.data + alpha * delta * Z_b2

    return (Z_w1, Z_b1, Z_w2, Z_b2)

def encode(board):
    """
    Encodes the board using one-hot encoding.
    Instead of representing the score pits, it only includes the score delta
    (TODO maybe including the score pits is better?)
    Also, it rounds all numbers over "max_beans" down to max_beans to limit the size of the representation,
    though it sacrifices accuracy for the rare states where a large number of beans is in one pit.
    Note: if you change the size of the board representation, also change the nx variable above!
    """
    state = []
    slot_enc = [0 for i in range(max_beans)] # one-hot encoding for a single pit
    for slot in range(len(board)):
        beans = board[slot]
        if slot != 6 and slot != 13:
            if beans > max_beans:
                print("Warning, found state with " + str(beans) + " beans in one hole")
                beans = max_beans
            state.extend(slot_enc)
            if beans > 0:
                state[-beans] = 1
    state.append((board[6] - board[13]) / 24)
    return np.array(state)

def epsgreedy_custom(eps: float, is_current_player: bool, include_best: bool):
    """
    Wrapper for epsgreedy_action to determine if we want to return just the action
    or the action and value
    """
    def inner(board: array.array, legal_actions: Tuple[int, ...], player: int):
        if include_best:
            return epsgreedy_action(board, legal_actions, player, eps, is_current_player)
        else:
            return epsgreedy_action(board, legal_actions, player, eps, is_current_player)[0]
    return inner

def epsgreedy_action(board: array.array, legal_actions: Tuple[int, ...], player: int, eps: float, is_current_player: bool) -> int:
    """
    Returns the eps-greedy action from the perspective of the current player, and the value of the action
    """

    # We always evaluate the position from the perspective of player 0
    # Therefore, to act as player 1, we flip the board and legal actions, find the best action, then invert it again
    if not is_current_player:
        board = copy_board(flip_board(board))
        legal_actions = flip_actions(legal_actions)

    # xa holds the child states after performing the action
    # next_players keeps track of whose turn it is after the action - 0 for same player, 1 for the other
    xa = np.zeros((len(legal_actions), nx))
    next_players = []
    for i in range(len(legal_actions)):
        act = legal_actions[i]
        s = copy_board(board)
        next_player = play_turn(s, 0, act)
        if next_player == 0:
            xa[i,:] = encode(s)
        else:
            # if it's the other player's turn, we encode the board from their perspective,
            # but invert the value when it's calculated below
            xa[i,:] = encode(flip_board(s))
        next_players.append(next_player)

    # Convert board representations in xa to values in va using the neural network
    x = Variable(torch.tensor(xa.transpose(), dtype=torch.float, device=device))
    h = torch.mm(w1, x) + b1  # matrix-multiply x with input weight w1 and add bias
    h_sigmoid = h.tanh()  # squash this with a sigmoid function
    y = torch.mm(w2, h_sigmoid) + b2  # multiply with the output weights w2 and add bias
    va = y.sigmoid().detach().cpu().numpy().flatten()

    # Invert the value if it's the other player's turn
    for i in range(len(next_players)):
        if next_players[i] == 1:
            va[i] = 1 - va[i]

    As = np.array(legal_actions)
    # vmax is the value of the best eps-greedy action
    vmax = np.max(va)
    if np.random.rand() < eps:  # epsilon greedy
        a = np.random.choice(As, 1)  # pure random policy
        # If we chose a random action, we return its value instead of the value of the best action
        vmax = va[np.where(As == a)][0]
    else:
        a = np.random.choice(As[vmax == va], 1)  # greedy policy, break ties randomly
    if not is_current_player:
        # If we flipped the board/actions at the start of the function, flip them back
        return (a[0] + 7) % 14, vmax
    return a[0], vmax

# High-level training and testing functions


def selfplay(alpha, eps, doprint=False):
    """
    Perform one episode of training with self-play.
    The play is always performed from the perspective of the player that the board is facing.
    That is, when the turn changes, we flip the board and resume as if it's the same player, but use the inverted reward
    """
    (Z_w1, Z_b1, Z_w2, Z_b2) = initialize_grads()
    action_function = epsgreedy_custom(eps, True, True)
    board = initial_board()
    player = 0
    turn = 0
    while not is_finished(board):
        possible_actions = legal_actions(board, 0)
        # To save computation, we reuse the value we calculated while finding the eps-greedy move
        action, val = action_function(board, possible_actions, 0)
        encodedBoard = encode(board)
        try:
            nextPlayer = play_turn(board, 0, action)
            if doprint:
                print(board_repr(board, action))
            turn += 1
        except Exception as e:
            raise e
        if is_finished(board):
            target = game_result(winner(board))
        elif player == nextPlayer:
            target = val
        else:
            # Turns have changed so we flip the board
            board = flip_board(board)
            target = val
        (Z_w1, Z_b1, Z_w2, Z_b2) = updateDuring(encodedBoard, player, alpha, 1.0, 0.9, target, Z_w1, Z_b1, Z_w2, Z_b2)
    return winner(board)

def testCurrentPlayer(iterations, play0, play1):
    """
    Perform several test games between players play0 and play1.
    Log the results.
    """
    p0 = 0 # p0 wins
    p1 = 0 # p1 wins
    d = 0 # draw
    for i in range(iterations):
        w = game(play0, play1)
        if w == 0:
            p0 += 1
        elif w == 1:
            p1 += 1
        else:
            d += 1
    print("p0 win: " + str(p0/iterations) + " p1 win: " + str(p1/iterations) + " draw: " + str(d/iterations))

def train():
    """
    Performs training iterations and tests.
    Runs forever until program is terminated.
    Alpha and eps parameters may need tuning.
    """
    gameround = 0
    start = time.time()
    alpha = 0.001
    eps = 0.1
    while True:
        # Test and log the performance against
        # Change/remove depending on what opponents you want to test against and how often
        print("Testing after " + str(1000 * gameround) + " rounds (" + str(round(time.time() - start)) + " sec)")
        print("Test against random")
        testCurrentPlayer(20, epsgreedy_custom(0.0, True, False), random_action)
        print("Test against random as p1")
        testCurrentPlayer(20, random_action, epsgreedy_custom(0.0, False, False))
        # Occasional test against a stronger, but slower player:
        if gameround % 20 == 0:
            print("Test against minimax")
            testCurrentPlayer(10, epsgreedy_custom(0.0, True, False), minimax_action)
            print("Test against minimax as p1")
            testCurrentPlayer(10, minimax_action, epsgreedy_custom(0.0, False, False))

        # Perform the training:
        gameround +=1
        for i in range(1000):
            selfplay(alpha, eps)


if __name__ == "__main__":
    train()