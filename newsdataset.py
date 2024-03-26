import torch
from torch.utils.data import Dataset
import numpy as np

class NewsDataset(Dataset):
    def __init__(self, embeddings, labels):
        # super(NewsDataset, self).__init__()
        self.embeddings = embeddings
        self.labels = labels

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]
        embeddings = torch.tensor(self.embeddings[idx], dtype=torch.float32)
        labels = torch.tensor(self.labels[idx], dtype=torch.float32)
        return embeddings, labels