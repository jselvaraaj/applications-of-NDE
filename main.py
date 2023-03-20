from environment import GridWorldEnv
import policy
import generator
import networks
import torch
import solver
import experiment
from sklearn.model_selection import train_test_split
import logger


logger_wrapper = logger.Logger()

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_default_tensor_type(torch.FloatTensor)

size = 5
step_size = 0.1
world = GridWorldEnv(render_mode="rgb_array",size = size,step_size=step_size)

observation_space_dim = 4
action_space_dim = 4

data_collecting_policy = policy.Policy(observation_space_dim,action_space_dim)

degenerated_state_space_dim = observation_space_dim + data_collecting_policy.weights.shape[0]
print("Degenerated State space dim: ",degenerated_state_space_dim)

evolve_len = 10
hidden_layer_size = 8

baseline_model = networks.NNBaseline(degenerated_state_space_dim, hidden_layer_size,evolve=evolve_len)
baseline_model.to(device)

test_model = networks.DynamicsFunction(input_channels=degenerated_state_space_dim, hidden_channels=hidden_layer_size, output_channels=degenerated_state_space_dim,evolve=evolve_len)
test_model.to(device)

episode_len=80
num_episodes=10
num_policy=5

logger_wrapper.config["grid_size"] = size
logger_wrapper.config["grid_step_size"] = step_size
logger_wrapper.config["observation_space_dim"] = observation_space_dim
logger_wrapper.config["action_space_dim"] = action_space_dim
logger_wrapper.config["degenerated_state_space_dim"] = degenerated_state_space_dim
logger_wrapper.config["evolve_len"] = 10
logger_wrapper.config["hidden_layer_size"] = hidden_layer_size
logger_wrapper.config["episode_len"] = episode_len
logger_wrapper.config["num_episodes"] = num_episodes
logger_wrapper.config["num_policy"] = num_policy


experiment.Experiment(baseline_model,test_model,world,evolve_len=evolve_len,episode_len=episode_len,num_episodes=num_episodes,num_policy=num_policy,logger_wrapper = logger_wrapper).run()