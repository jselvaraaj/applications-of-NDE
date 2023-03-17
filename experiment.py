import generator
from sklearn.model_selection import train_test_split
import solver

class Experiment:

    def __init__(self,data_collecting_policy,baseline_policy,test_policy,env,episode_len,num_episodes):
        self.baseline_policy = baseline_policy
        self.data_collecting_policy = data_collecting_policy
        self.test_policy = test_policy
        self.env = env
        self.episode_len = episode_len
        self.num_episodes = num_episodes
    
    def run(self):
        print("Generating data...")
        data_gen = generator.DataGenerator(self.env)
        train_size, test_size = 0.8,0.2
        X, y = data_gen.get_X_y(self.data_collecting_policy,self.num_episodes,self.episode_len)

        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size, test_size=test_size)

        print("Baseline policy")
        print("Training model...")

        solver.train(X_train,y_train, self.baseline_policy)

        print("Testing model...")
        solver.test(X_test,y_test, self.baseline_policy)


        print("Test policy")
        print("Training model...")

        solver.train(X_train,y_train, self.test_policy)

        print("Testing model...")
        solver.test(X_test,y_test, self.test_policy)
