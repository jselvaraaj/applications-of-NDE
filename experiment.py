import generator
from sklearn.model_selection import train_test_split
import solver
import torch
import policy
import wandb

class Experiment:

    def __init__(self,baseline_policy,test_policy,env,episode_len,num_episodes,num_policy):
        self.baseline_policy = baseline_policy
        # self.data_collecting_policy = data_collecting_policy
        self.test_policy = test_policy
        self.env = env
        self.episode_len = episode_len
        self.num_episodes = num_episodes
        self.num_policy = num_policy
        
    
    def run(self):
        print("Generating data...")
        data_gen = generator.DataGenerator(self.env)
        train_split, val_split, test_split = 0.8,0.1,0.1
        X, y = [],[]

        wandb.config.train_split = train_split
        wandb.config.test_split = test_split
        wandb.config.val_split = val_split

        obs_space_dim = wandb.config.observation_space_dim
        act_space_dim = wandb.config.action_space_dim

        data_collecting_policies = []
        for i in range(self.num_policy):
            data_collecting_policy = policy.Policy(obs_space_dim,act_space_dim)
            data_collecting_policies.append(data_collecting_policy)

        train_dataset, val_dataset, test_dataset = data_gen.get_X_y(data_collecting_policies,self.num_episodes,self.episode_len,train_split, val_split, test_split)

        print("\n\n")

        print(f"Train dataset size {len(train_dataset)}")
        print(f"Val dataset size {len(val_dataset)}")
        print(f"Test dataset size {len(test_dataset)}")


        print("\n\n")

        print("Baseline model")
        print("Training model...")
        solver.train(train_dataset, val_dataset, self.baseline_policy,verbose=True)

        print("Testing model...")
        solver.test(test_dataset, self.baseline_policy)

        print("\n\n")

        print("Test model")
        print("Training model...")

        solver.train(train_dataset, val_dataset, self.test_policy,verbose=True)

        print("Testing model...")
        solver.test(test_dataset, self.test_policy)
