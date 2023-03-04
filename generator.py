import torch
from MP import DegeneratedMarkovProcess 

class DataGenerator:

    def __init__(self,env,device = None) -> None:
        self.env = env

        if device is None:
            self. device = "cuda" if torch.cuda.is_available() else "cpu"
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

        return data

    def get_X_y(self, policy, n=10, t = 20):

        state_time_series = self.get_time_series(policy, n, t)

        X = state_time_series[:,:-1]
        y = state_time_series[:,-1]

        return X,y
