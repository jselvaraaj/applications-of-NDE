from environment import GridWorldEnv
import policy
import generator
import networks
import torch
import solver
import experiment
from sklearn.model_selection import train_test_split


device = "cuda" if torch.cuda.is_available() else "cpu"

size = 5
world = GridWorldEnv(render_mode="rgb_array",size = size)

data_collecting_policy = policy.Policy(4,4)

state_dim = 4 + data_collecting_policy.weights.shape[0]

print("State dim: ",state_dim)

baseline_model = networks.NNBaseline(state_dim, 8)

baseline_model.to(device)

test_model = networks.DynamicsFunction(input_channels=state_dim, hidden_channels=8, output_channels=state_dim)

test_model.to(device)

experiment.Experiment(baseline_model,test_model,world,episode_len=10,num_episodes=10,num_policy=5).run()