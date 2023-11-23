import torch
import numpy as np
import math
from array import array

from mancala.groups.teymi3.Qnetwork import QNetwork


def copy_board(board):
    return array("i", board)



# Load the model from 'model.pth'
model = QNetwork(14,6)
model.load_state_dict(torch.load('mancala/groups/teymi3/agent_Qnetwork.pth'))

model.eval()


def action(state, legalactions, player):
    #transfroming the data to fit the trained model 
    #the model knows the state as [P1, P2, P1_states, P2_states]
    if len(legalactions) == 1: return legalactions[0]

   
    bord = copy_board(state)

    if player == 1:
        bord = bord[6:] + bord[:6]
        logskref = [x - 6 for x in legalactions]
    else:  logskref = [x + 1 for x in legalactions]



    q_inf = [-math.inf,-math.inf,-math.inf,-math.inf,-math.inf,-math.inf]
    q_values = []
    action = 0
    
    # Use the model to compute the state and action
    q_values = model(torch.from_numpy(np.array(bord)).float().unsqueeze(0)) # a tensor of shape (1, n_actions)
    q_values = q_values.flatten().tolist() #make q's as list for easy iteration
  
    for i in logskref:
        q_inf[i-1] = q_values[i-1]
    #q_inf = q_values

    action = q_values.index(max(q_inf)) # an integer representing the action with the highest q-value
    if player == 1: 
        action += 7

    return action 