import torch
import torchcde
import wandb

def train(X,y,model,num_epochs= 100,batch_size = 32,verbose = False, DE = False):

  optimizer = torch.optim.Adam(model.parameters())


  for i in range(X):

    if DE:
      print("Started making the X data continuous")
      X[i] = torchcde.hermite_cubic_coefficients_with_backward_differences(X[i][0]),X[i][1]
      print("Finished making the X data continuous")

    evolve_len = X[i][1]
    train_dataset = torch.utils.data.TensorDataset(X[i][0], y[i])
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size)
    for epoch in range(num_epochs):
      if verbose:
        print(f'Epoch: {epoch}',end=' ')
      for batch in train_dataloader:
        batch_X, batch_y = batch

        pred_y = model(batch_X,evolve_len).squeeze(-1)

        loss = torch.nn.functional.mse_loss(pred_y, batch_y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

      if verbose:
        print(f'Training loss: {loss.item()}')
      
      wandb.log({"loss":loss.item()})
    
  # wandb.log_artifact(model)

def test(X,y,model,DE=False):
  
  if DE:
    X = torchcde.hermite_cubic_coefficients_with_backward_differences(X)
  pred = model(X)
  loss = torch.nn.functional.mse_loss(pred, y)

  print(f'Test loss: {loss}')