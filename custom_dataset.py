import torch
import utils

class EpisodesDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_dict):
        self.dataset_dict = list(dataset_dict.items())
        self.device = utils.get_device()

    def __len__(self):
        return len(self.dataset_dict)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        e_len, (X,y) = self.dataset_dict[idx]
        return X, y , torch.tensor([e_len]).to(self.device)
