import torch
import utils

def get_state_time_series_from_env(env,policy,n=10,t=100):
  res = []

  maxT = float("-inf")
  policy_weights = utils.get_weights_as_vec(policy)

  for i in range(n):
    
    state_time_series = []
    env.reset()

    for j in range(t):
            observation = torch.from_numpy(env._get_obs()[0])

            action = get_action(policy,observation)

            _, reward, terminated, truncated, info = env.step(action)
            state_time_series.append(observation)
            if terminated:
                print("Finished after {} timesteps".format(t+1))
                break
    
    maxT = max(maxT,len(state_time_series))

    temp = couple_state_policy(state_time_series,policy_weights)

    res.append(temp)

    res2 = []

  print("Finished sampling states")
  
  #assuming ele is of the form t*features
  for ele in res:
    t = ele.shape[0]
    temp = ele.repeat_interleave(torch.cat([torch.ones(t-1,dtype=int),torch.tensor([maxT-t+1])]).squeeze(),dim=0)    
    res2.append(temp)

  return torch.stack(res2)


def get_action(policy,obs):
  return torch.argmax(policy(obs[None,:].to(torch.float))).cpu().detach().item()


def couple_state_policy(state_time_series,policy_weights):
  res = []

  for state in state_time_series:
    res.append(torch.hstack((state,policy_weights)))

  return torch.stack(res,dim = 0)


def split_basec_on_markov(time_series,t_len = 20,num_tim_pairs = 10):
  n,t,f = time_series.shape

  if t_len >= t:
    raise Exception("t_end cannot be greater than t")

  l = torch.randint(high = t-t_len,size=(1,))[0]

  # l = torch.floor(torch.rand(num_tim_pairs) * (t-t_len))
  # u = torch.floor(l+t_len)

  # res = []
  # for i in range(l.shape[0]):
  #   arange = torch.arange(l[i], u[i])
    
  #   X = time_series.index_select(1, arange)

  #   res.append(X)

  arange = torch.arange(l,l+t_len+1)
  X = time_series.index_select(1, arange)

  print(X.shape)

  X_train = X[:,:-1]
  y_train = X[:,-1]

  return X_train,y_train



  

