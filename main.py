from environment import GridWorldEnv
import policy
import generator
import networks
import torch
import solver


device = "cuda" if torch.cuda.is_available() else "cpu"


size = 5
world = GridWorldEnv(render_mode="rgb_array",size = size)

policy1 = policy.Policy(4,4,device)

data_gen = generator.DataGenerator(world)

train_X, train_y = data_gen.get_X_y(policy1,100,20)

test_X, test_y = data_gen.get_X_y(policy1,100,20)

state_dim = train_X.shape[-1]

model = networks.DynamicsFunction(input_channels=state_dim, hidden_channels=8, output_channels=state_dim)

model.to(device)

solver.train(train_X, train_y, model)

solver.test(test_X, test_y, model)