import random
import math
import torch
import mediapy as media
import time
import numpy as np

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

def align_policy(policy,grid_size,epochs=1e4):
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
        
    print(loss.item())
        
    return policy

    
    
    
    