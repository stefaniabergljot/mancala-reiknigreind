"""
Player trained with selfplay using a neural netowrk and TD(lambda).

Has two versions of the neural network:
'action' (default) was trained using only selfplay, and
'action2' trained using a mix of selfplay and play against random
"""
import array
from typing import Tuple
import torch
import numpy as np
from torch.autograd import Variable

from mancala.game import copy_board, flip_board, play_turn  # noqa: F401


NAME = "teymi1"

w1 = None
w2 = None
b1 = None
b2 = None

filename = "nn_backups/selfplay/"
device = 'cpu'
nx= 20*12+1
nh = 10
max_beans = 20

def get_files(name):
    w1 = torch.load("mancala/groups/teymi1/nn/w1_trained_" + name + ".pth", map_location=torch.device('cpu'))
    w2 = torch.load("mancala/groups/teymi1/nn/w2_trained_" + name + ".pth", map_location=torch.device('cpu'))
    b1 = torch.load("mancala/groups/teymi1/nn/b1_trained_" + name + ".pth", map_location=torch.device('cpu'))
    b2 = torch.load("mancala/groups/teymi1/nn/b2_trained_" + name + ".pth", map_location=torch.device('cpu'))
    return w1, w2, b1, b2

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

def best_action(board: array.array, legal_actions: Tuple[int, ...], player: int, max: bool) -> int:
    if not max:
        tempboard = copy_board(flip_board(board))
        legal_actions = flip_actions(legal_actions)
    else:
        tempboard = copy_board(board)

    xa = np.zeros((len(legal_actions), nx))
    ps = []
    for i in range(len(legal_actions)):
        act = legal_actions[i]
        s = copy_board(tempboard)
        p = play_turn(s, 0, act)
        #xa[i,:] = encode(s, p)
        if p == 0:
            xa[i,:] = encode(s)
        else:
            xa[i,:] = encode(flip_board(s))
        ps.append(p)
    x = Variable(torch.tensor(xa.transpose(), dtype=torch.float, device=device))
    h = torch.mm(w1, x) + b1  # matrix-multiply x with input weight w1 and add bias
    h_sigmoid = h.tanh()  # squash this with a sigmoid function
    y = torch.mm(w2, h_sigmoid) + b2  # multiply with the output weights w2 and add bias
    va = y.sigmoid().detach().cpu().numpy().flatten()

    for i in range(len(ps)):
        if ps[i] == 1:
            va[i] = 1 - va[i]

    As = np.array(legal_actions)
    vmax = np.max(va)
    a = np.random.choice(As[vmax == va], 1)  # greedy policy, break ties randomly
    if not max:
        return (a[0] + 7) % 14
    return a[0]

def action(board: array.array, legal_actions: Tuple[int, ...], player: int) -> int:
    global w1, w2, b1, b2
    if w1 is None:
        w1, w2, b1, b2 = get_files("selfplay_finalcandidate_880000")
    if legal_actions[0] < 6:
        # We are p0
        return best_action(board, legal_actions, 0, True)
    else:
        return best_action(board, legal_actions, 0, False)

def action2(board: array.array, legal_actions: Tuple[int, ...], player: int) -> int:
    global w1, w2, b1, b2
    if w1 is None:
        w1, w2, b1, b2 = get_files("selfplay_finalcandidate_mixed_120000")
    if legal_actions[0] < 6:
        # We are p0
        return best_action(board, legal_actions, 0, True)
    else:
        return best_action(board, legal_actions, 0, False)
