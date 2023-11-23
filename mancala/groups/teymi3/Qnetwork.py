import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque

GAMMA = 0.7
EPSILON = 1.0
EPSILON_DECAY = 0.995
EPSILON_MIN = 0.01
LEARNING_RATE = 0.0003 #0.0001
BATCH_SIZE = 32
MEMORY_SIZE = 10000

# Check if CUDA (GPU support) is available and use it; otherwise, use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

class QNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
def get_actions(self):
    actions = []
    for i in range(1,7):
        if self.get_state()[i] > 0:
            actions.append(i)
    return actions


class Agent:
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        self.model = QNetwork(input_size, output_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=LEARNING_RATE)
        self.memory = deque(maxlen=MEMORY_SIZE)

    def act(self, state):
        if random.random() < EPSILON:
            return random.randint(0, self.output_size - 1)
        else:
            state = torch.from_numpy(np.array(state)).float().unsqueeze(0)
            with torch.no_grad():
                q_values = self.model(state) 
                return q_values.max(1)[1].item()


    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def learn(self):
        if len(self.memory) < 4*BATCH_SIZE: #original was only batch size changed to 4*BS
            return
        batch = random.sample(self.memory, BATCH_SIZE)
        states, actions, rewards, next_states, dones = zip(*batch)
        states = torch.from_numpy(np.array(states)).float()
        actions = torch.from_numpy(np.array(actions)).long()
        rewards = torch.from_numpy(np.array(rewards)).float()
        next_states = torch.from_numpy(np.array(next_states)).float()
        dones = torch.from_numpy(np.array(dones)).float()
        q_values = self.model(states) 
        next_q_values = self.model(next_states) 
        q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        next_q_values = next_q_values.max(1)[0]
        targets = rewards + GAMMA * next_q_values * (1 - dones)
        loss = nn.MSELoss()(q_values, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        global EPSILON
        EPSILON = max(EPSILON * EPSILON_DECAY, EPSILON_MIN)


class Environment:
    def __init__(self):
        self.init_board = [0, 4, 4, 4, 4, 4, 4, 0, 4, 4, 4, 4, 4, 4] # change the order of the board elements
        self.board = self.init_board.copy()
        self.player = 1
        self.done = False

    def reset(self):
        self.board = self.init_board.copy()
        self.player = 1
        self.done = False
        return self.get_state()

    def step(self, action):
        reward = 0
        pit = action +1# add 1 to the action to get the corresponding index in the array
        #print(pit)
        store_pit = -1 # initialize store_pit to -1
        #change in return, checks if the move is valid, and terminates the game? original reward = -10, maybe not working well, tested -100 briefly now testing 0.
        #if self.board[pit] == 0:
        #    print("illegal move)")
        #    return self.get_state(), -10, True
        stones = self.board[pit]
        self.board[pit] = 0
        #distribution of stones across the entire board
        while stones > 0:
            pit = (pit + 1) % 14 # increment pit by 1 and wrap around if it exceeds the array length
            if (self.player == 1 and pit == 7) or (self.player == 2 and pit == 0): # skip the opponent's store
                continue
            if (self.player == 1 and pit == 0) or (self.player == 2 and pit == 7): # change so that we reward for passing over our pit.
                reward += 1
            self.board[pit] += 1
            stones -= 1

        if (self.player == 1 and pit in range(1,7)) or (self.player == 2 and pit in range(8,14)): # check if the last stone landed in the player's side
            if self.board[pit] == 1: # check if the last stone landed in an empty pit
                opposite_pit = (14 - pit) % 14 # get the index of the opposite pit
                store_pit = 7 * self.player - 7 # assign store_pit a value based on the player's number
                reward += self.board[pit] + self.board[opposite_pit] # change make the reward based on the amount in the pits wrather than 10
                self.board[store_pit] += self.board[pit] + self.board[opposite_pit] # capture the stones from both pits and add them to the store
                self.board[pit] = 0
                self.board[opposite_pit] = 0
                #reward += 10
            if pit == store_pit: # compare pit and store_pit
                reward += 1
                return self.get_state(), reward, False

        if all(self.board[i] == 0 for i in range(1,7)): # check if game is over by either player and move all the stones to the winner's store
            for i in range(8,14):
                self.board[7] += self.board[i]
                self.board[i] = 0
            self.done = True
        elif all(self.board[i] == 0 for i in range(8,14)):
            for i in range(1,7):
                self.board[0] += self.board[i]
                self.board[i] = 0
            self.done = True

        if self.done:
            reward += (self.board[7 * self.player - 7] -
                    self.board[7 - 7 * self.player]) * 100 # compare the scores of both players and multiply by a large factor
        else:
            self.player = 3 - self.player # switch the player's turn

        return self.get_state(), reward, self.done


    def get_state(self):
        state = self.board
        if self.player == 2:
            tmp_state = state.copy()
            tmp_state[0],tmp_state[7] = tmp_state[7], tmp_state[0]
            for i in range(1,7):
                j = i+7
                tmp_state[i], tmp_state[j] = tmp_state[j], tmp_state[i]  
                state = tmp_state
        return state
