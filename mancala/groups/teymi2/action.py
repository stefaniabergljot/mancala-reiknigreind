"""Example action.py.

Create a directory inside mancala/groups
and place all your code inside that directory.
Create a file called action.py within that directory.
We will only use the function action defined in mancala/group/your_group/action.py.
"""

from mancala.game import copy_board, flip_board, play_turn
import numpy as np
import torch
from torch.autograd import Variable


NAME = "HOPUR2"


def action(board, legal_actions, player: int) -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print(torch.cuda.current_device())
        print(torch.cuda.device(0))
        print(torch.cuda.device_count())
        print(torch.cuda.get_device_name(0))

    # if player = 1 flip board
    if player == 1:
        board = flip_board(board)
        legal_actions = [i-7 for i in legal_actions]

    epCount = 600000

    pth = 'mancala/groups/teymi2/'
    w1 = torch.load(pth + 'w1_trained_'+str(epCount)+'.pth', map_location=torch.device('cpu'))
    w2 = torch.load(pth + 'w2_trained_'+str(epCount)+'.pth', map_location=torch.device('cpu'))
    b1 = torch.load(pth + 'b1_trained_'+str(epCount)+'.pth', map_location=torch.device('cpu'))
    b2 = torch.load(pth + 'b2_trained_'+str(epCount)+'.pth', map_location=torch.device('cpu'))
    nx = 14
    na = len(legal_actions)
    xa = np.zeros((na,nx)) # all after-states for the different moves
    for i in range(0, na):
        xa[i,:] = play_turn(copy_board(board), 0, legal_actions[i])
    x = Variable(torch.tensor(xa.transpose(), dtype = torch.float, device = device))
    # now do a forward pass to evaluate the board's after-state value
    h = torch.mm(w1,x) + b1 # matrix-multiply x with input weight w1 and add bias
    h_sigmoid = h.tanh() # squash this with a sigmoid function
    y = torch.mm(w2,h_sigmoid) + b2 # multiply with the output weights w2 and add bias
    va = y.sigmoid().detach().cpu().numpy().flatten()
    vmax = np.max(va)
    if type(legal_actions) is not np.ndarray:
        legal_actions = np.array(legal_actions)

    a = np.random.choice(legal_actions[vmax == va],1)[0]

    if player == 1:
        a += 7
    return a
