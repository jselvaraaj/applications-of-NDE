import torch
from MP import DegeneratedMarkovProcess 

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

    def get_X_y(self, policies, n=10, t = 20,t_train_end_from_last=10):

        state_time_series = []
        for i,policy in enumerate(policies):
          print(f"Generating data for policy {i+1}...")
          state_time_series.append(self.get_time_series(policy, n, t))

        state_time_series = torch.cat(state_time_series,dim=0)

        # X = state_time_series[:,:-t_train_end_from_last]
        # y = state_time_series[:,-1]

        X = []
        y = []
        # for t1 in range(0,t):
        t1 = 0 # not looping for t1 as grid world is markovian
        for t2 in range(t1+1,t):
          for t3 in range(t2+1,t):
            new_s_t = state_time_series[:,t1:t2]
            # pred_time = torch.full((n,t2-t1+1,1),t3)

            # new_s_t = torch.cat((state_time_series,pred_time),2)

            pred = state_time_series[:,t3]

            X.append((new_s_t.float(),t3))
            y.append(pred.float())

        return X,y
