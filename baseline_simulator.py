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
            project="NDE_as_Mental_Models",
            tags=["Neural_Network_Baseline"]
        )

device = utils.get_device()
torch.set_default_tensor_type(torch.FloatTensor)

print("Device: ",device)

size = 100
world = GridWorldEnv(render_mode="rgb_array",size = size)

observation_space_dim = 4 # not a hyperparameter
action_space_dim = 4 # not a hyperparameter

#This is only used for getting the total number paramaeters in policy
data_collecting_policy = policy.Policy(observation_space_dim,action_space_dim)

print("data collecting policy total parameters - ", data_collecting_policy.weights.shape[0])
degenerated_state_space_dim = observation_space_dim + data_collecting_policy.weights.shape[0]
print("Degenerated State space dim - ",degenerated_state_space_dim)

hidden_layer_size = 32

baseline_model = networks.NNBaseline(degenerated_state_space_dim, hidden_layer_size,observation_space_dim)
baseline_model.to(device)

episode_len=80
num_episodes=100
num_policy=100
num_epochs = 100

wandb.config.grid_size = size
wandb.config.observation_space_dim = observation_space_dim
wandb.config.action_space_dim = action_space_dim
wandb.config.degenerated_state_space_dim = degenerated_state_space_dim
wandb.config.hidden_layer_size = hidden_layer_size
wandb.config.episode_len = episode_len
wandb.config.num_episodes = num_episodes
wandb.config.num_policy = num_policy
wandb.config.num_epochs = num_epochs

print("Number of parameters in baseline: ", utils.count_parameters(baseline_model))

exp = experiment.Experiment(world,episode_len=episode_len,num_episodes=num_episodes,num_policy=num_policy,num_epochs=num_epochs)
exp.run(baseline_model)
