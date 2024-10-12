"""
Dataloader
"""

import torch
import pickle
from torch.utils.data import Dataset
import numpy as np

class Graph_DataLoader(Dataset):
    def __init__(self, data, data_dir):
        self.data_dir = data_dir
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return idx, self.data[idx,5].__float__()/100 #return reaction indices and yields
        
        