import Qnetwork
from Qnetwork import Agent, Environment
import torch
# Check if CUDA (GPU support) is available and use it; otherwise, use CPU
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device ="cuda"
#print(device)
# Create an instance of the environment
env = Environment()

# Create an instance of the agent
agent = Agent(14, 6)

# Set the number of episodes to train
episodes = 10000 #start small then find out what the gpu is capable of.
start = 0
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

# Assume agent.model is defined as a subclass of torch.nn.Module
model = agent.model

# Save the model to a file named 'model.pth'
torch.save(model.state_dict(), 'mancala/groups/teymi3_th/agent_Qnetwork.pth') #/Users/tarnarsson/Desktop/mancala/mancala/groups/group_3A/sdmodel_50k.pth