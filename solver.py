import torch
import torchcde
import wandb
from custom_dataset import EpisodesDataset
from utils import shuffle
import os

def train(train_dataset, val_dataset,model,num_epochs= 100,batch_size = 16,verbose = False):
    optimizer = torch.optim.Adam(model.parameters())

    train_dataset = EpisodesDataset(train_dataset)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size)
    for epoch in range(int(num_epochs)):
        if verbose:
          print(f'Epoch: {epoch}',end=' ')
        model.train()
        for batch in train_dataloader:
            optimizer.zero_grad()
            batch_X, batch_y, evolve_lens = batch

            train_loss = 0
            preds = []
            ys = []
            once = True
            for i,[evolve_len] in enumerate(evolve_lens):
                X,y = shuffle(batch_X[i],batch_y[i])
                pred_y = model(X,evolve_len).squeeze(-1)
                preds.append(pred_y)
                ys.append(y)
                
                if (epoch %100 == 0 and once) or epoch == int(num_epochs)-1:
                    print('t',evolve_len)
                    print('X',X[:10])
                    print('pred_y', pred_y[:10])
                    print('y',y[:10])
                    once = False
                    # exit(0)
            
            preds = torch.stack(preds,dim = 0)
            ys = torch.stack(ys,dim=0)
            loss = torch.nn.functional.mse_loss(preds, ys)
            train_loss = loss.item()
            loss.backward()
            optimizer.step()


        with torch.no_grad():
            val_loss = 0
            model.eval()
            preds = []
            ys = []
            for evolve_len in val_dataset:
                X,y = val_dataset[evolve_len]
                pred_y = model(X,evolve_len).squeeze(-1)           
                preds.append(pred_y)
                ys.append(y)
            
            preds = torch.stack(preds,dim = 0)
            ys = torch.stack(ys,dim=0)
            val_loss = torch.nn.functional.mse_loss(preds, ys).item()

        if verbose:
            print(f'Training loss: {train_loss}',end='\t')
            print(f'Validation loss: {val_loss}')

        wandb.log({"Validation loss":val_loss},step=epoch)
        wandb.log({"Training loss":train_loss},step=epoch)
    
    # model.save(os.path.join(wandb.run.dir, "node.model"))
    torch.save(model.state_dict(), os.path.join(wandb.run.dir, "exp.model"))
    wandb.save(os.path.join(wandb.run.dir, "exp.model"))
  # wandb.save('model.pth')

def test(test_dataset, model):
    test_loss = 0
    preds = []
    ys = []
    for evolve_len in test_dataset:
        X,y = test_dataset[evolve_len]
        pred_y = model(X,evolve_len).squeeze(-1)
        # print('t',evolve_len)
        preds.append(pred_y)
        ys.append(y)
        # print('X',X)
        # print('pred_y', pred_y)
        # print('y',y)
        # exit(0)
        
    preds = torch.stack(preds,dim = 0)
    ys = torch.stack(ys,dim=0)
    test_loss = torch.nn.functional.mse_loss(preds, ys).item()

    print(f'Test loss: {test_loss}')

    wandb.log({"Test loss":test_loss})



