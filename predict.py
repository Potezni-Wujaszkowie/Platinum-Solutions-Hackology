import torch
import torch.nn as nn
import random
from models import GRURegressionModel
import pandas as pd
import numpy as np

if __name__ == "__main__":
    seed = 42
    valid_size = 0.1
    num_workers = 8
    batch = 32
    csv_path = '/home/lukass/hackology/data2.csv'

    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

    data_df = pd.read_csv(csv_path)

    eps_to_load = 45
    device = torch.device("cuda")

    input_dim = 1  # Wymiar wejściowy (jedna cecha)
    hidden_dim = 64  # Wymiar ukryty
    output_dim = 1  # Wymiar wyjściowy (przewidywana wartość)
    n_layers = 2  # Liczba warstw GRU
    dropout = 0.1  # Współczynnik dropout

    # Tworzenie instancji modelu
    model = GRURegressionModel(input_dim, hidden_dim, output_dim, n_layers, dropout)

    model.load_state_dict(torch.load(f"models/data2/model_weights{eps_to_load}.pth", weights_only=True))
    model.eval()
    model = model.to(device)

    idcs = [
        0, 1, 2, 3, 4, 5, 6, 7, 8,
        374, 375, 376, 377, 378, 379, 380, 381, 382]
    
    last_column = data_df["products_sold"].tolist()
    trends = data_df["trends"].tolist()
    bias = len(last_column) - idcs[-1] - 1
    idcs = [idx + bias for idx in idcs]
    trend = trends[-1]

    fut_pred = 30
    # Predict future values
    preds = []
    for _ in range(fut_pred):
        sold = [last_column[idx] for idx in idcs]
        sold.insert(0, trend)
        print(sold)
        sequence = torch.tensor(sold).unsqueeze(0).to(device).float()
        out = model(sequence)
        out = torch.squeeze(out).item()
        preds.append(out)
        print(out)
        last_column.append(out)
        idcs = [idx + 1 for idx in idcs]

import matplotlib.pyplot as plt

eps = [i for i in range(len(preds))]
plt.plot(eps, preds)
plt.savefig("chuj.png")