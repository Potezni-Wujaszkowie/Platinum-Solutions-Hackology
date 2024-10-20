import torch
import torch.nn as nn
import random
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import math


class SequenceDataset(Dataset):

    def __init__(self, src_sequences, tar_sequences):
        self.src_sequences = src_sequences
        self.tar_sequences = tar_sequences

    def __len__(self):
        return len(self.src_sequences)

    def __getitem__(self, idx):
        src_seq = self.src_sequences[idx]
        tar_seq = self.tar_sequences[idx]
        return src_seq, tar_seq

class GRURegressionModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers, dropout=0.1):
        super(GRURegressionModel, self).__init__()

        self.inp = nn.Linear(input_dim, hidden_dim)
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim 
        self.output_dim = output_dim
        self.n_layers = n_layers
        
        self.gru = nn.GRU(input_size=input_dim, 
                          hidden_size=hidden_dim, 
                          num_layers=n_layers, 
                          dropout=dropout, 
                          batch_first=True)


        self.linear = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = x.unsqueeze(-1)
        # x = self.inp(x)
        h0 = torch.zeros(2, x.size(0), self.hidden_dim).to("cuda")
        out, _ = self.gru(x, h0)
        
        last_output = out[:, -1, :]
        
        x = self.linear(last_output)
        
        return x