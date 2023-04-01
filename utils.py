import random
import math
import torch

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