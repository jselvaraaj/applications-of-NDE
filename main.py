from environment import GridWorldEnv
import policy
import generator
import networks
import torch
import solver
from sklearn.model_selection import train_test_split


device = "cuda" if torch.cuda.is_available() else "cpu"

size = 5
world = GridWorldEnv(render_mode="rgb_array",size = size)

policy1 = policy.Policy(4,4,device)

data_gen = generator.DataGenerator(world)

train_size, test_size = 0.8,0.2
X, y = data_gen.get_X_y(policy1,100,20)

state_dim = X.shape[-1]

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size, test_size=test_size)

model = networks.DynamicsFunction(input_channels=state_dim, hidden_channels=8, output_channels=state_dim)

model.to(device)

print("Training model...")

solver.train(X_train,y_train, model)

print("Testing model...")
solver.test(X_test,y_test, model)