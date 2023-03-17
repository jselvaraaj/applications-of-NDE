import torch
import torchcde

def train(X,y,model,num_epochs= 100,batch_size = 32,verbose = False):

  optimizer = torch.optim.Adam(model.parameters())

  print("Making the X data continuous")

  train_coeffs = torchcde.hermite_cubic_coefficients_with_backward_differences(X)

  train_dataset = torch.utils.data.TensorDataset(train_coeffs, y)
  train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size)
  for epoch in range(num_epochs):
    if verbose:
      print(f'Epoch: {epoch}',end=' ')
    for batch in train_dataloader:
      batch_coeffs, batch_y = batch

      pred_y = model(batch_coeffs).squeeze(-1)
      loss = torch.nn.functional.mse_loss(pred_y, batch_y)
      loss.backward()
      optimizer.step()
      optimizer.zero_grad()

    if verbose:
      print(f'Training loss: {loss.item()}')

def test(X,y,model):
  
  test_coeffs = torchcde.hermite_cubic_coefficients_with_backward_differences(X)
  pred = model(test_coeffs)
  loss = torch.nn.functional.mse_loss(pred, y)

  print(f'Test loss: {loss}')