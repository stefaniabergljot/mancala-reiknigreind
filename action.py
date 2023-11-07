import torch
import numpy as np
import math

from Qnetwork import QNetwork


# Load the model from 'model.pth'
model = QNetwork(14,6)
model.load_state_dict(torch.load('/Users/tarnarsson/Desktop/mancala/mancala/groups/group_3A/state_dict_model.pth'))
model.load_state_dict(torch.load('/mancala/groups/group_3A/state_dict_model.pth'))
model.eval()


def action(state, legalactions, player):
    #transfroming the data to fit the trained model 
    #the model knows the state as [P1, P2, P1_states, P2_states]
    p2 = state.pop(7)
    state.insert(1,p2)

    q_values = []
    action = 0
    
    # Use the model to compute the state and action
    q_values = model(torch.from_numpy(np.array(state)).float().unsqueeze(0)) # a tensor of shape (1, n_actions)
    q_values = q_values.flatten().tolist() #make q's as list for easy iteration
    #for i in legalactions:
    for i in range(6):
        if state[i+2]>0:
            q_values[i] = q_values[i]
        else: q_values[i] = -math.inf
            
    action = q_values.index(max(q_values)) # an integer representing the action with the highest q-value
    return action

