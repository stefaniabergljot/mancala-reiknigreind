import torch
import numpy as np
import math

def action(state, legalactions, player):
    
    q_values = []
    action = 0
    
    # Load the model from 'model.pth'
    model = torch.load('model.pth3')
    model.eval()

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

