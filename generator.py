import torch
from MP import DegeneratedMarkovProcess 
from sklearn.model_selection import train_test_split
import utils

class DataGenerator:

    def __init__(self,env,device = None) -> None:
        self.env = env

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

    def get_time_series(self, policy, n=10, t = 20):
        data = []

        dmp = DegeneratedMarkovProcess(self.env,policy)

        for i in range(n):
            
            dmp.reset()
            observation = dmp.get_obs()            
            episode = [observation]

            for j in range(t):

                    terminated = dmp.step()

                    if terminated:
                        print(f"Finished after {t+1} timesteps")

                        for k in range(j,t):
                            episode.append(observation)

                        break
                    
                    observation = dmp.get_obs()
                    episode.append(observation)

            episode = torch.stack(episode,dim=0)
            data.append(episode)

        data = torch.stack(data,dim=0)
        
        data = data.to(self.device)

        return data

    def get_X_y(self, policies, n, t, train_split, val_split, test_split):

        train_policies, test_policies = utils.split(policies,train_split)
        test_policies,val_policies = utils.split(test_policies,test_split/(test_split+val_split))

        state_time_series_train = []
        for i,policy in enumerate(train_policies):
          print(f"Generating data for policy {i+1}...")
          state_time_series_train.append(self.get_time_series(policy, n, t))
        state_time_series_train = torch.cat(state_time_series_train,dim=0)


        state_time_series_val = []
        for i,policy in enumerate(val_policies):
          print(f"Generating data for policy {i+1}...")
          state_time_series_val.append(self.get_time_series(policy, n, t))
        state_time_series_val = torch.cat(state_time_series_val,dim=0)

        state_time_series_test = []
        for i,policy in enumerate(test_policies):
          print(f"Generating data for policy {i+1}...")
          state_time_series_test.append(self.get_time_series(policy, n, t))
        state_time_series_test = torch.cat(state_time_series_test,dim=0)


        possible_evolve_lengths = list(range(1,t))

        train_evolve_lens, test_evolve_lens = utils.split(possible_evolve_lengths,train_split)
        test_evolve_lens,val_evolve_lens = utils.split(test_evolve_lens,test_split/(test_split+val_split))

        train_evolve_lens = set(train_evolve_lens)
        test_evolve_lens = set(test_evolve_lens)
        val_evolve_lens = set(val_evolve_lens)


        train_dataset = {}
        val_dataset = {}
        test_dataset = {}
        for t1 in range(t):
          for t2 in range(t1+1,t):
            if (t2-t1) in train_evolve_lens:
              X_train = state_time_series_train[:,t1]
              y_train = state_time_series_train[:,t2]
              train_dataset[t2-t1] = X_train, y_train
            elif (t2-t1) in val_evolve_lens:
              X_valid = state_time_series_val[:,t1]
              y_valid = state_time_series_val[:,t2]
              val_dataset[t2-t1] = X_valid, y_valid
            else:
              X_test = state_time_series_test[:,t1]
              y_test = state_time_series_test[:,t2]
              test_dataset[t2-t1] = X_test, y_test

        return train_dataset, val_dataset, test_dataset
