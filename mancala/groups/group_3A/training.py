import Qnetwork
from Qnetwork import Agent, Environment
import time
#from mancala.game import initial_board, is_finished
#from groups.group_3A.Qnetwork import Agent

# Create an instance of the environment
env = Environment()

# Create an instance of the agent
agent = Agent(14, 6)

# Set the number of episodes to train
episodes = 5000

start = time.time()
# Loop over the episodes
for i in range(episodes):
    # Reset the environment and get the initial state
    state = env.reset()

    # Set the total reward to zero
    total_reward = 0

    # Loop until the episode is done
    while not env.done:
        # Choose an action using the agent's policy
        action = agent.act(state)

        # Take the action and observe the next state, reward, and done flag
        next_state, reward, done = env.step(action)

        # Remember the transition for replay
        agent.remember(state, action, reward, next_state, done)

        # Update the state
        state = next_state

        # Update the total reward
        total_reward += reward

        # Learn from the replay memory
        agent.learn()

    # Print the episode number and the total reward
    print(f"Episode {i+1}: Total reward = {total_reward}")

endtime = time.time()
total = endtime - start
print(f"Startime at : {start} and time ends at : {endtime} for a total of : {total}      ")

import torch

# Assume agent.model is defined as a subclass of torch.nn.Module
model = agent.model

# Save the model to a file named 'model.pth'
torch.save(model, 'model.pth')