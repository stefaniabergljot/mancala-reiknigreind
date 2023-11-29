"""
Player trained with selfplay using an actor-critic neural netowrk and TD wih eligibility traces (I hope)

"""
import array
from typing import Tuple
import torch
import numpy as np
from torch.autograd import Variable

from mancala.game import copy_board, flip_board, play_turn  # noqa: F401


NAME = "teymi1ac"

w1 = None
w2 = None
b1 = None
b2 = None
theta = None

filename = "nn_backups/selfplay/"
device = 'cpu'
nx= 20*12+1
nh = 10
max_beans = 20

def get_files(name):
    w1 = torch.load("mancala/groups/teymi1/nn/w1_ac_trained_" + name + ".pth", map_location=torch.device('cpu'))
    w2 = torch.load("mancala/groups/teymi1/nn/w2_ac_trained_" + name + ".pth", map_location=torch.device('cpu'))
    b1 = torch.load("mancala/groups/teymi1/nn/b1_ac_trained_" + name + ".pth", map_location=torch.device('cpu'))
    b2 = torch.load("mancala/groups/teymi1/nn/b2_ac_trained_" + name + ".pth", map_location=torch.device('cpu'))
    theta = torch.load("mancala/groups/teymi1/nn/theta_ac_trained_" + name + ".pth", map_location=torch.device('cpu'))
    return w1, w2, b1, b2, theta

def encode(board):
    state = []
    slot_enc = [0 for i in range(max_beans)]
    for slot in range(len(board)):
        beans = board[slot]
        if slot != 6 and slot != 13:
            if beans > max_beans:
                #print("Warning, found state with " + str(beans) + " beans in one hole")
                beans = max_beans
            state.extend(slot_enc)
            if beans > 0:
                state[-beans] = 1
    state.append((board[6] - board[13]) / 24)
    return np.array(state)
def flip_actions(legal_actions):
    return tuple((i + 7) % 14 for i in legal_actions)

def best_action(board: array.array, legal_actions: Tuple[int, ...], player: int, is_current_player: bool) -> int:
    """
        Returns the softmax-greedy action from the perspective of the current player, and the value of the action
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
            xa[i, :] = encode(s)
        else:
            # if it's the other player's turn, we encode the board from their perspective,
            # but invert the value when it's calculated below
            xa[i, :] = encode(flip_board(s))
        next_players.append(next_player)

    # Convert board representations in xa to values in va using the neural network
    x = Variable(torch.tensor(xa.transpose(), dtype=torch.float, device=device))
    h = torch.mm(w1, x) + b1  # matrix-multiply x with input weight w1 and add bias
    h_sigmoid = h.tanh()  # squash this with a sigmoid function
    y = torch.mm(w2, h_sigmoid) + b2  # multiply with the output weights w2 and add bias
    va = y.sigmoid().detach()

    # Invert the value if it's the other player's turn
    for i in range(len(next_players)):
        if next_players[i] == 1:
            va[0, i] = 1 - va[0, i]

    # TODO insert actor changes here
    h_tanh = h.tanh()  # same as h_sigmoid, could reuse
    pi = torch.mm(theta, h_tanh).softmax(1)
    m = torch.multinomial(pi, 1)  # soft

    # Convert back to the legal actions
    a = legal_actions[m]

    if not is_current_player:
        return (a + 7) % 14
    else:
        return a


def action(board: array.array, legal_actions: Tuple[int, ...], player: int) -> int:
    global w1, w2, b1, b2, theta
    if w1 is None:
        w1, w2, b1, b2, theta = get_files("123000")
    if legal_actions[0] < 6:
        # We are p0
        return best_action(board, legal_actions, 0, True)
    else:
        return best_action(board, legal_actions, 0, False)
