import torch
import numpy as np

def action(state, legalactions, player):
    
    q_values = []
    action = 0
    
    # Load the model from 'model.pth'
    model = torch.load('model.pth')
    model.eval()

    # Use the model to compute the state and action
    q_values = model(torch.from_numpy(np.array(state)).float().unsqueeze(0)) # a tensor of shape (1, n_actions)
    #for i in legalactions:
    #    q_values = tmp_q_values[i] # to make sure we are taking a legal action, if the algorithm is right this should not have any effect.

    action = q_values.max(1)[1].item() # an integer representing the action with the highest q-value
    return action

