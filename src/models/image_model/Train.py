import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data

from featDataset import featDataset
from featModel import featModel

import torch.optim as optim
import torch.utils.data as data
import argparse
import numpy as np

from addict import Dict
import yaml
import os
import time

SEED = 14
torch.manual_seed(SEED)


def get_arg():
    parser = argparse.ArgumentParser(description="Grid anchor based image cropping")
    parser.add_argument("config", type=str, help="path of a config file")
    args = parser.parse_args()

    config = Dict(yaml.safe_load(open(args.config)))
    return config


def main():
    config = get_arg()
    config.save_folder = os.path.join(config.save_folder, config.model)
    if not os.path.exists(config.save_folder):
        os.makedirs(config.save_folder)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda":
        torch.set_default_tensor_type("torch.cuda.FloatTensor")
    else:
        torch.set_default_tensor_type("torch.FloatTensor")

    Traindataset = featDataset(mode="train")
    Testdataset = featDataset(mode="test")

    model = featModel()
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, 1 + config.epochs):
        print("epoch:", epoch)
        dataset_perm = np.random.permutation(range(len(Traindataset)))
        t0 = time.time()
        train(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            dataset=Traindataset,
            config=config,
            device=device,
            dataset_perm=dataset_perm,
        )
        t1 = time.time()
        print("trainin time :", round(r1 - r0))
        test(model=model, dataset=Testdataset, config=config, device=device)


def train(model, optimizer, criterion, dataset, config, device, dataset_perm):
    model.train()
    total_loss = 0
    for i in dataset_perm:
        batch_dataset, batch_label = batch_maker(
            dataset[i], batch_size=config.batch_size
        )
        for data, label in zip(batch_dataset, batch_label):

            data = data.to(device)
            label = label.to(device)

            output = model(data)
            loss = criterion(output, label)

            # backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print("\r", loss.item(), end="")


def test(model, dataset, config, device):
    print("")
    model.eval()
    labels = []
    preds = []
    eval = evaluater()
    for i in range(len(dataset)):
        batch_dataset, batch_label = batch_maker(
            dataset[i], shuffle=False, drop_last=False, batch_size=config.batch_size
        )
        for data, label in zip(batch_dataset, batch_label):
            data = data.to(device)
            output = model(data)
            output = torch.argmax(output, axis=1)

            labels.extend(label.detach().numpy())
            preds.extend(output.cpu().detach().numpy())
    labels, preds = np.array(labels), np.array(preds)
    eval.acc(labels, preds)


class evaluater:
    def __init__(self):
        pass

    def acc(self, x, y):
        accuracy = len(x[x == y]) / len(x)
        print("accuracy:", accuracy)


def batch_maker(dataset, shuffle=True, drop_last=True, batch_size=8):
    data, label = dataset[0], dataset[1]
    if shuffle:
        p = torch.randperm(data.size()[0])
        data = data[p]
        label = label[p]
    n = len(data) // batch_size

    batched_data = []
    batched_label = []
    if not drop_last:
        n += 1
    for i in range(n):
        b = data[i * batch_size : i * batch_size + batch_size, :, :, :]
        l = label[i * batch_size : i * batch_size + batch_size]
        batched_data.append(b)
        batched_label.append(l)
    return batched_data, batched_label


if __name__ == "__main__":
    main()
