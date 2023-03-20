import generator
from sklearn.model_selection import train_test_split
import solver
import torch
import policy
import logger

class Experiment:

    def __init__(self,baseline_policy,test_policy,env,evolve_len,episode_len,num_episodes,num_policy,logger_wrapper):
        self.baseline_policy = baseline_policy
        # self.data_collecting_policy = data_collecting_policy
        self.test_policy = test_policy
        self.env = env
        self.evolove_len = evolve_len
        self.episode_len = episode_len
        self.num_episodes = num_episodes
        self.num_policy = num_policy
        self.logger_wrapper = logger_wrapper
        
    
    def run(self):
        print("Generating data...")
        data_gen = generator.DataGenerator(self.env)
        train_size, test_size = 0.8,0.2
        X, y = [],[]

        self.logger_wrapper.config["train_size"] = train_size
        self.logger_wrapper.config["test_size"] = test_size

        obs_space_dim = self.logger_wrapper.config["observation_space_dim"]
        act_space_dim = self.logger_wrapper.config["action_space_dim"]

        for i in range(self.num_policy):
            print(f"Generating data for policy {i+1}...")
            data_collecting_policy = policy.Policy(obs_space_dim,act_space_dim)
            X_, y_ = data_gen.get_X_y(data_collecting_policy,self.num_episodes,self.episode_len,self.evolove_len)
            X.append(X_)
            y.append(y_)
        
        X = torch.cat(X,dim=0)
        y = torch.cat(y,dim=0)

        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size, test_size=test_size)

        print("\n\n")

        with self.logger_wrapper.make_run(tags=["Baseline","RNN+NN"]) as run:
            print("Baseline model")
            print("Training model...")
            solver.train(X_train,y_train, self.baseline_policy,verbose=True)

            print("Testing model...")
            solver.test(X_test,y_test, self.baseline_policy)

        print("\n\n")

        with self.logger_wrapper.make_run(tages=["NDE","NCDE+NODE"]) as run:
            print("Test model")
            print("Training model...")

            solver.train(X_train,y_train, self.test_policy,DE=True,verbose=True)

            print("Testing model...")
            solver.test(X_test,y_test, self.test_policy,DE=True)
