import random
import math
import torch
import mediapy as media
import time
import numpy as np
import wandb

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def split(arr,ratio):
    n = len(arr)

    res = random.sample(arr,n)
    d = math.floor(n*ratio)

    return res[:d], res[d:] 

def shuffle(X,y):
    n = X.shape[0]

    r=torch.randperm(n)

    return X[r], y[r]

def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"

def align_policy(policy,grid_size,epochs=1e3):
    device = get_device()
    
    # policy.policy.train()
    
    optimizer = torch.optim.Adam(policy.policy.parameters())
    
    y = torch.nn.functional.one_hot(torch.flatten(torch.randint(0,4,(grid_size,grid_size),device=device))).to(torch.float)
    
    state = torch.from_numpy(np.asarray(list(np.ndindex(grid_size,grid_size)))).to(torch.float).to(device)
    
    target = torch.full_like(state,grid_size//2,device=device).to(torch.float)
    
    X = torch.cat((state,target),dim=1)
    
    loss = 0
    for i in range(int(epochs)):
        optimizer.zero_grad()
        
        pred = policy.policy(X)
        
        loss = torch.nn.functional.cross_entropy(pred, y)
        
        loss.backward()
        optimizer.step()
        
        if (i+1) % (epochs/10) == 0:
            print(f"Epoch {i+1}, Loss ", loss.item())
            
        
    return policy

def visualize_policy(world,policy,epi_len,video_name):
    world.reset()
    
    device = get_device()

    framerate = 5 
    frames = []

    for t in range(epi_len):
        frame = world.render()

        frames.append(frame)
        action = policy.get_action(torch.from_numpy(world._get_obs()[0]).to(device))
        observation, reward, terminated, truncated, info = world.step(action)

        if terminated:
            print("Finished after {} timesteps".format(t+1))
            break

    wandb.log({f"{video_name}": wandb.Video(np.transpose(np.asarray(frames),(0,3,1,2)), fps=framerate, format="mp4")})
    
    
    
    