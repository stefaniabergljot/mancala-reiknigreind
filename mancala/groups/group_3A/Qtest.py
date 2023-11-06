import torch
import numpy as np
import math
from Qnetwork import Agent

legal_moves = []
state = [2, 10, 5, 0, 0, 5, 0, 7, 5, 0, 0, 0, 7, 7]


if sum(state) == 48:

    model = torch.load('model.pth')
    model.eval()
    print(model.eval())

        # Use the model to compute the state and action
    #q_values = model(torch.from_numpy(np.array(state)).float().unsqueeze(0)) # a tensor of shape (1, n_actions)
    #q_values = model(state)
    #print(type(q_values))
    #q_values = q_values.flatten().tolist()
    #print(type(q_values))
    
    for i in range(6):
        if state[i+2]>0:
            q_values[i] = q_values[i]
        else: q_values[i] = -math.inf

    action = q_values.index(max(q_values)) # an integer representing the action with the highest q-value
    print(f"We are going to take action {action} from the q values: {q_values} while the board looks like this: {state}   ")
else: print(f"invalid state: {state} the amount of stones is: {sum(state)}")