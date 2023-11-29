from array import array
from typing import List, Tuple
import random
import numpy as np
import torch
from torch.autograd import Variable
# cuda will only create a significant speedup for large/deep networks and batched training
device = torch.device('cpu')

NAME = "Stefania-ac-baseline"

from mancala.game import (
    play_turn,
    copy_board,
    is_finished,
    Board,
    Player,
    legal_actions as get_legal_actions,
    get_score,
)
RANGE0 = range(6)  # Player's 0 pits and Mancala
RANGE1 = range(7, 13)  # Player's 1 pits and Mancala
RANGES = (RANGE0, RANGE1)
AREA0 = slice(6)
AREA1 = slice(7, 13)

nb = 20
model = 5 * [None]
loadtrainstep = 686
model[0] = torch.load('mancala/groups/actor_critic_baseline/ac/b1_trained_'+str(loadtrainstep)+'.pth')
model[1] = torch.load('mancala/groups/actor_critic_baseline/ac/w1_trained_'+str(loadtrainstep)+'.pth')
model[2] = torch.load('mancala/groups/actor_critic_baseline/ac/b2_trained_'+str(loadtrainstep)+'.pth')
model[3] = torch.load('mancala/groups/actor_critic_baseline/ac/w2_trained_'+str(loadtrainstep)+'.pth')
model[4] = torch.load('mancala/groups/actor_critic_baseline/ac/theta_'+str(loadtrainstep)+'.pth')

# assume that player 0 is allways the one to play!
def one_hot_encode(brd):
    nf = 12*nb + 1 # +1 just code my own Manacala pit
    x = np.zeros(nf)
    x[-1] = (brd[13]-brd[6]) / 24.0 
    for i in RANGE0:
        j = min(brd[i],nb-1)
        x[nb*i+j] += 1
    for i in RANGE1:
        j = min(brd[i],nb-1)
        x[nb*(i-1)+j] += 1  
    return x

def getfeatures(board, legal_actions):
    nf = 12*nb + 1 # +1 just code my own Manacala pit
    x = np.zeros((nf,len(legal_actions)))
    for i in range(len(legal_actions)):
        brd = list(board).copy()
        play_turn(brd, 0, legal_actions[i]) # always assumes player 0 is the one to play
        x[:,i] = one_hot_encode(brd)
    return x

def softmax_policy(xa,  model):
    (nx,na) = xa.shape 
    x = Variable(torch.tensor(xa, dtype = torch.float, device = device)) 
    # now do a forward pass to evaluate the board's after-state value
    h = torch.mm(model[1],x) + model[0] @ torch.ones((1,na), device = device)  # matrix-multiply x with input weight w1 and add bias
    h_tanh = h.tanh() # squash this with a sigmoid function
    y = torch.mm(model[3],h_tanh) + model[2] # multiply with the output weights w2 and add bias
    va = y.sigmoid().detach()
    # now for the actor:
    pi = torch.mm(model[4],h_tanh).softmax(1)
    m = torch.multinomial(pi, 1) # soft
    return va, pi, m

def action(board: array, legal_actions: Tuple[int, ...], player: int) -> int:

    if player == 1: # player 0 owns the neural network, player 1 borrows it!
        flip_brd = array("i", board[7:] + board[:7])
        flip_actions = tuple([a - 7 for a in legal_actions])
        x = getfeatures(flip_brd, flip_actions)
    else:
        x = getfeatures(board, legal_actions)
    va, pi, m  = softmax_policy(x,  model)
#    if random.random() <= 0.9:
#        m = np.argmax(va)
    chosen_action = legal_actions[m]
    return chosen_action
