from environment import GridWorldEnv
import policy
import generator
import networks
import torch
import solver
import experiment
from sklearn.model_selection import train_test_split
import wandb
import utils

wandb.init(
            project="NDE_as_Mental_Models"
        )


device = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_default_tensor_type(torch.FloatTensor)

size = 5
world = GridWorldEnv(render_mode="rgb_array",size = size)

observation_space_dim = 4
action_space_dim = 4

data_collecting_policy = policy.Policy(observation_space_dim,action_space_dim)

print("data collecting policy total parameters - ", data_collecting_policy.weights.shape[0])
degenerated_state_space_dim = observation_space_dim + data_collecting_policy.weights.shape[0]
print("Degenerated State space dim - ",degenerated_state_space_dim)

hidden_layer_size = 8

baseline_model = networks.NNBaseline(degenerated_state_space_dim, hidden_layer_size)
baseline_model.to(device)
test_model = networks.DynamicsFunction(degenerated_state_space_dim, hidden_layer_size)
test_model.to(device)

episode_len=80
num_episodes=10
num_policy=10

wandb.config.grid_size = size
wandb.config.observation_space_dim = observation_space_dim
wandb.config.action_space_dim = action_space_dim
wandb.config.degenerated_state_space_dim = degenerated_state_space_dim
wandb.config.hidden_layer_size = hidden_layer_size
wandb.config.episode_len = episode_len
wandb.config.num_episodes = num_episodes
wandb.config.num_policy = num_policy

print("Number of parameters in baseline: ", utils.count_parameters(baseline_model))
print("Number of parameters in test model: ", utils.count_parameters(test_model))


experiment.Experiment(baseline_model,test_model,world,episode_len=episode_len,num_episodes=num_episodes,num_policy=num_policy).run()