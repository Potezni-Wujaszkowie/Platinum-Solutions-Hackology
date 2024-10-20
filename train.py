import sys
import torch
import torch.nn as nn
import json
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F
import random
from models import GRURegressionModel, SequenceDataset
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

if __name__ == "__main__":
    seed = 42
    data_num = 3
    valid_size = 0.1
    num_workers = 8
    batch = 32
    csv_path = f'/home/lukass/hackology/data{data_num}.csv'

    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    data_lim = 390
    data_range = 9

    data_df = pd.read_csv(csv_path)
    data_df["trends"] = (data_df["trends"] - data_df["trends"].min()) / (data_df["trends"].max() - data_df["trends"].min())
    data_len = len(data_df)
    sequences = [
        np.concatenate((data_df.loc[i + np.arange(data_range), data_df.columns == "products_sold"].to_numpy(),
                         data_df.loc[i + 365 + np.arange(data_range), data_df.columns == "products_sold"].to_numpy())) for i in range(data_len - data_lim + data_range)]
    trends_idcs = [i + 365 + data_range for i in range(data_len - data_lim + data_range)]

    sequences = torch.tensor(sequences).squeeze()
    src = sequences[:, :-1]
    max_values = src.max(dim=1, keepdim=True)[0]
    src = src / max_values
    trg = sequences[:, -1]
    
    trends = data_df.loc[trends_idcs, 'trends'].to_numpy()
    trends_tensor = torch.tensor(trends).float()

    trends_tensor = trends_tensor.unsqueeze(-1)
    combined_sequences = torch.cat((trends_tensor, src), dim=1)

    trg = trg.float()
    src = src.float()


    seq_train, seq_val, tar_train, tar_val = train_test_split(
        src, trg, test_size=valid_size, random_state=42
    )

    print(f"Train size: {len(seq_train)}, Validation size: {len(seq_val)}")

    dataset_train = SequenceDataset(seq_train, tar_train)
    dataset_val = SequenceDataset(seq_val, tar_val)

    dataloader_train = DataLoader(dataset_train, batch_size=batch, shuffle=True, drop_last=True, num_workers=num_workers)
    dataloader_val = DataLoader(dataset_val, batch_size=batch, shuffle=True, drop_last=True, num_workers=num_workers)
    print(f"train len: {len(dataloader_train)}")
    print(f"val len: {len(dataloader_val)}\n")


    input_dim = 1
    hidden_dim = 64
    output_dim = 1
    n_layers = 2
    dropout = 0.1

    # Tworzenie instancji modelu
    model = GRURegressionModel(input_dim, hidden_dim, output_dim, n_layers, dropout)

    # train params
    eps= 60
    lr = 0.003
    device = torch.device("cuda")
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Training loop
    epochs = 60
    for epoch in range(epochs):
        loss_train = 0
        model.train()
        for seq, labels in dataloader_train:
            seq, labels = seq.to(device), labels.to(device)
            #print(labels)
            optimizer.zero_grad()
            y_pred = model(seq)  # Add batch dimension
            y_pred = y_pred.squeeze()
            # print(y_pred)
            loss = criterion(y_pred, labels)
            loss.backward()
            loss_train += loss.item()
            optimizer.step()

        loss_eval = 0
        model.eval()
        for seq, labels in dataloader_val:
            seq, labels = seq.to(device), labels.to(device)
            y_pred = model(seq)
            y_pred = y_pred.squeeze()
            loss = criterion(y_pred, labels)
            loss_eval += loss.item()
        torch.save(model.state_dict(), f"models/data{data_num}/model_weights{epoch}.pth")
            
        # if epoch % 20 == 0:
        print(f'Epoch {epoch} Loss train: {loss_train/len(dataloader_train)}, Loss valid: {loss_eval/len(dataloader_val)}')
