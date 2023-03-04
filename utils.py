import torch

def get_weights_as_vec(model):
  policy_weights = []

  for name, param in model.named_parameters():
      policy_weights.append(torch.flatten(param.detach().clone()))

  policy_weights = torch.hstack(policy_weights)

  return policy_weights