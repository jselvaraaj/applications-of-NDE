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
import hydra
from omegaconf import DictConfig, OmegaConf


@hydra.main(version_base=None, config_path="conf",config_name="config")
def main(cfg: DictConfig):
    
    baseline = cfg.baseline
    
    
    if baseline:
        tags = ["Neural_Network_Baseline"]
    else:
        tags = ["NODE"]
    
    wandb.init(
                project="NDE_as_Mental_Models",
                tags= tags
            )

    device = utils.get_device()
    torch.set_default_tensor_type(torch.FloatTensor)

    print("Device: ",device)

    size = cfg.grid_size
    world = GridWorldEnv(render_mode="rgb_array",size = size)

    observation_space_dim = 4 # not a hyperparameter
    action_space_dim = 4 # not a hyperparameter

    #This is only used for getting the total number paramaeters in policy
    data_collecting_policy = policy.Policy(observation_space_dim,action_space_dim)

    print("data collecting policy total parameters - ", data_collecting_policy.weights.shape[0])
    degenerated_state_space_dim = observation_space_dim + data_collecting_policy.weights.shape[0]
    print("Degenerated State space dim - ",degenerated_state_space_dim)

    hidden_layer_size = cfg.hidden_layer_size

    episode_len= cfg.episode_len
    num_episodes= cfg.num_episodes
    num_policy= cfg.num_policy
    num_epochs = cfg.num_epochs

    wandb.config.grid_size = size
    wandb.config.observation_space_dim = observation_space_dim
    wandb.config.action_space_dim = action_space_dim
    wandb.config.degenerated_state_space_dim = degenerated_state_space_dim
    wandb.config.hidden_layer_size = hidden_layer_size
    wandb.config.episode_len = episode_len
    wandb.config.num_episodes = num_episodes
    wandb.config.num_policy = num_policy
    wandb.config.num_epochs = num_epochs

    if baseline:
        model = networks.NNBaseline(degenerated_state_space_dim, hidden_layer_size,observation_space_dim)
        print("Number of parameters in baseline: ", utils.count_parameters(model))
    else:
        model = networks.DynamicsFunction(degenerated_state_space_dim, hidden_layer_size,observation_space_dim)
        print("Number of parameters in NODE: ", utils.count_parameters(model))
        
    exp = experiment.Experiment(world,episode_len=episode_len,num_episodes=num_episodes,num_policy=num_policy,num_epochs=num_epochs)
    model.to(device)
    exp.run(model)
        
if __name__ == "__main__":
    main()
