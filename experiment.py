import generator
from sklearn.model_selection import train_test_split
import solver
import torch
import policy
import wandb
import pickle
import os.path
import utils

class Experiment:

    def __init__(self,env,episode_len,num_episodes,num_policy,num_epochs,use_cache=False):
        self.env = env
        self.episode_len = episode_len
        self.num_episodes = num_episodes
        self.num_policy = num_policy
        self.num_epochs = num_epochs
        self.use_cache = use_cache

        self.get_data()

    def get_data(self):
      if os.path.isfile('data_for_exp.pkl') and self.use_cache:
        print("Reading from cache")
        
        with open("data_for_exp.pkl", "rb") as f:
            (self.train_dataset,self.val_dataset,self.test_dataset) = pickle.load(f)
        
      else:
        self.gen_data()
        
        # with open("data_for_exp.pkl", "wb") as f:
        #     pickle.dump((self.train_dataset,self.val_dataset,self.test_dataset), f, protocol=pickle.HIGHEST_PROTOCOL)
            

    def gen_data(self):
        data_gen = generator.DataGenerator(self.env)
        train_split, val_split, test_split = 0.8,0.1,0.1
        X, y = [],[]

        wandb.config.train_split = train_split
        wandb.config.test_split = test_split
        wandb.config.val_split = val_split

        obs_space_dim = wandb.config.observation_space_dim
        act_space_dim = wandb.config.action_space_dim
        grid_size = wandb.config.grid_size
        
        print("Aligning policy")
        data_collecting_policies = []
        for i in range(self.num_policy):
            data_collecting_policy = policy.Policy(obs_space_dim,act_space_dim)
            
            data_collecting_policy = utils.align_policy(data_collecting_policy,grid_size)
            
            data_collecting_policies.append(data_collecting_policy)
        
        print("Generating episode data...")
        self.train_dataset, self.val_dataset, self.test_dataset = data_gen.get_X_y(data_collecting_policies,self.num_episodes,self.episode_len,train_split, val_split, test_split,obs_space_dim=obs_space_dim)
        
        print("\n\n")

        print(f"Train dataset size {len(self.train_dataset)}")
        print(f"Val dataset size {len(self.val_dataset)}")
        print(f"Test dataset size {len(self.test_dataset)}")

    def run(self,simulator):
        print("\n\n")
        print("Training model...")
        solver.train(self.train_dataset, self.val_dataset, simulator,verbose=True,num_epochs = self.num_epochs)

        print("Testing model...")
        solver.test(self.test_dataset, simulator)