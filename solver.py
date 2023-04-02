import torch
import torchcde
import wandb
from custom_dataset import EpisodesDataset
from utils import shuffle

def train(train_dataset, val_dataset,model,num_epochs= 100,batch_size = 32,verbose = False):
  optimizer = torch.optim.Adam(model.parameters())

  train_dataset = EpisodesDataset(train_dataset)
  train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size)
  for epoch in range(num_epochs):
    if verbose:
      print(f'Epoch: {epoch}',end=' ')
    model.train()
    for batch in train_dataloader:
      batch_X, batch_y, evolve_lens = batch

      train_loss = 0
      for i,[evolve_len] in enumerate(evolve_lens):
        X,y = shuffle(batch_X[i],batch_y[i])

        pred_y = model(X,evolve_len).squeeze(-1)

        loss = torch.nn.functional.mse_loss(pred_y, y)
        train_loss += loss.item()
        loss.backward()
    
      optimizer.step()
      optimizer.zero_grad()

    with torch.no_grad():
      val_loss = 0
      model.eval()
      for evolve_len in val_dataset:
        X,y = val_dataset[evolve_len]
        pred_y = model(X,evolve_len).squeeze(-1)
        val_loss += torch.nn.functional.mse_loss(pred_y, y).item()

    if verbose:
      print(f'Training loss: {train_loss}')
      print(f'Validation loss: {val_loss}')

    wandb.log({"Validation loss":val_loss},step=epoch)
    wandb.log({"Training loss":train_loss},step=epoch)

  # torch.save(model.state_dict(), 'model.pth')
  # wandb.save('model.pth')

def test(test_dataset, model):
  test_loss = 0
  for evolve_len in test_dataset:
    X,y = test_dataset[evolve_len]
    pred_y = model(X,evolve_len).squeeze(-1)
    test_loss += torch.nn.functional.mse_loss(pred_y, y).item()

  print(f'Test loss: {test_loss}')
  
  wandb.log({"Test loss":test_loss})



