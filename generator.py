import torch
from MP import DegeneratedMarkovProcess 
from sklearn.model_selection import train_test_split
import utils

class DataGenerator:

    def __init__(self,env) -> None:
        self.env = env

        self.device = utils.get_device()

    def get_time_series(self, policy, n=10, t = 20):
        data = []

        dmp = DegeneratedMarkovProcess(self.env,policy)

        for i in range(n):
            
            dmp.reset()
            observation = dmp.get_obs()            
            episode = [observation]
            terminated = False

            for j in range(t):
                terminated = dmp.step() 

                if terminated: # Happens iff agent reachs target
                    print(f"Finished after {j+1} timesteps")
                    break
                
                observation = dmp.get_obs()
                episode.append(observation)

            if terminated:
              for k in range(j,t):
                  episode.append(observation)

            episode = torch.stack(episode,dim=0)
            data.append(episode)

        data = torch.stack(data,dim=0)
        
        data = data.to(self.device)

        return data

    def get_X_y(self, policies, n, t, train_split, val_split, test_split,obs_space_dim = 4):

        train_policies, test_policies = utils.split(policies,train_split)
        test_policies,val_policies = utils.split(test_policies,test_split/(test_split+val_split))

        state_time_series_train = []
        for i,policy in enumerate(train_policies):
          print(f"Train policy {i+1}...",end='\t')
          state_time_series_train.append(self.get_time_series(policy, n, t))
          print()
        state_time_series_train = torch.cat(state_time_series_train,dim=0)
        print()

        state_time_series_val = []
        for i,policy in enumerate(val_policies):
          print(f"Validation policy {i+1}...",end='\t')
          state_time_series_val.append(self.get_time_series(policy, n, t))
          print()
        state_time_series_val = torch.cat(state_time_series_val,dim=0)
        print()

        state_time_series_test = []
        for i,policy in enumerate(test_policies):
          print(f"Test policy {i+1}...",end='\t')
          state_time_series_test.append(self.get_time_series(policy, n, t))
          print()
        state_time_series_test = torch.cat(state_time_series_test,dim=0)
        print()


        print("Visualize ...")

        print('\nTrain Policy')
        utils.visualize_policy(self.env,train_policies[0],t,'train')
        print()        

        print('\nValidation Policy')
        utils.visualize_policy(self.env,val_policies[0],t,'val')
        print()

        print('\nTest Policy')
        utils.visualize_policy(self.env,val_policies[0],t,'test')
        print()

        possible_evolve_lengths = list(range(1,t))

        train_evolve_lens, test_evolve_lens = utils.split(possible_evolve_lengths,train_split)
        test_evolve_lens,val_evolve_lens = utils.split(test_evolve_lens,test_split/(test_split+val_split))

        print("Evolution lengths")

        print('\nTrain Policy')
        print(train_evolve_lens)        

        print('\nValidation Policy')
        print(val_evolve_lens)        

        print('\nTest Policy')
        print(test_evolve_lens)        
        print()

        train_evolve_lens = set(train_evolve_lens)
        test_evolve_lens = set(test_evolve_lens)
        val_evolve_lens = set(val_evolve_lens)
        
        print("Shape of X used for training for a given evolution length", state_time_series_train[:,0].shape)
        print("Shape of y used for training for a given evolution length", state_time_series_train[:,0,:obs_space_dim].shape)
        print()

        train_dataset = {}
        val_dataset = {}
        test_dataset = {}
        for t1 in range(t):
          for t2 in range(t1+1,t):
            if (t2-t1) in train_evolve_lens:
              X_train = state_time_series_train[:,t1]
              y_train = state_time_series_train[:,t2,:obs_space_dim]
              train_dataset[t2-t1] = X_train, y_train
            elif (t2-t1) in val_evolve_lens:
              X_valid = state_time_series_val[:,t1]
              y_valid = state_time_series_val[:,t2,:obs_space_dim]
              val_dataset[t2-t1] = X_valid, y_valid
            else:
              X_test = state_time_series_test[:,t1]
              y_test = state_time_series_test[:,t2,:obs_space_dim]
              test_dataset[t2-t1] = X_test, y_test

        return train_dataset, val_dataset, test_dataset
