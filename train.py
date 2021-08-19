import argparse
import os

import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from addict import Dict
from dataset import return_data
from models import build_model

from evaluater import evaluater
import tqdm

SEED = 14
torch.manual_seed(SEED)


def get_arg():
    parser = argparse.ArgumentParser(description="image model for action segmentation")
    parser.add_argument("config", type=str, help="path of a config file")
    args = parser.parse_args()
    config = Dict(yaml.safe_load(open(args.config)))
    return config


def main():
    config = get_arg()
    config.save_folder = os.path.join(config.save_folder, config.model)
    if not os.path.exists(config.save_folder):
        os.makedirs(config.save_folder)

    device = config.device
    torch.set_default_tensor_type("torch.cuda.FloatTensor")
    print("device:", device)
    train_set, test_set = return_data.return_dataset(config)
    # train_set, test_set = return_data.return_dataloader(config)
    model = build_model.build_model(config)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
    weight = torch.ones(11).cuda()
    if True:
        weight[0] = 0
        weight[-1] = 0
    criterion = nn.CrossEntropyLoss(weight=weight)

    best_eval = 0
    for epoch in range(1, 1 + config.epochs):
        print("\nepoch:", epoch)
        dataset_perm = np.random.permutation(range(len(train_set)))
        if config.repeat:
            for _ in range(config.repeat):
                dp = np.random.permutation(range(len(train_set)))
                dataset_perm = np.concatenate([dataset_perm, dp])

        t0 = time.time()
        train(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            dataset=train_set,
            config=config,
            device=device,
            dataset_perm=dataset_perm,
        )
        scheduler.step()
        print(f"\nlr: {scheduler.get_last_lr()}")
        t1 = time.time()
        print(f"\ntraining time :{round(t1 - t0)} sec")

        best_eval = test(
            model=model,
            dataset=test_set,
            config=config,
            device=device,
            best_eval=best_eval,
        )
    torch.save(model.state_dict(), "model.pth")


def train(model, optimizer, criterion, dataset, config, device, dataset_perm):
    model.train()
    total_loss = 0
    counter = 0
    for i in tqdm.tqdm(dataset_perm):
        batch_dataset, batch_label = dataset[dataset_perm[i]]
        for data, label in zip(batch_dataset, batch_label):
            # for data, label in zip(batch_dataset, batch_label):
            data = data.to(device)
            label = label.to(device)

            output = model(data)
            loss = criterion(output, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            counter += 1
    print(f"\rtotal_loss: [{total_loss / counter}]", end="")


def test(model, dataset, config, device, best_eval=0, th=0.6):
    model.eval()
    labels = []
    preds = []
    eval = evaluater()
    for i in range(len(dataset)):
        batch_dataset, batch_label = dataset[i]
        for data, label in zip(batch_dataset, batch_label):
            data = data.to(device)
            output = model(data)
            output = torch.argmax(output, axis=1)

            labels.extend(label.detach().cpu().numpy())
            preds.extend(output.detach().cpu().numpy())

    labels, preds = np.array(labels), np.array(preds)
    # with open("test.pkl", "wb") as f:
    # pickle.dump(preds, f)
    eval.set_data(labels, preds)
    eval.print_eval(["accuracy"])
    score = eval.return_eval_score()
    if score > best_eval and score > th:
        path = os.path.join(
            config.save_folder,
            config.model
            + "_"
            + config.type
            + "_"
            + str(score).replace(".", "")
            + ".pth",
        )
        # torch.save(model.state_dict(), path)
    return max(score, best_eval)

    # eval.acc(labels, preds)


# def batch_maker(dataset, shuffle=True, drop_last=True, batch_size=8, n_sample = 100):
#     data, label = dataset[0], dataset[1]
#     if shuffle:
#         p = torch.randperm(data.size()[0])
#         data = data[p]
#         label = label[p]
#     n = len(data) // batch_size
#
#     batched_data = []
#     batched_label = []
#     if not drop_last:
#         n += 1
#     for i in range(n):
#         if n_sample < n:
#             break
#         b = data[i * batch_size : i * batch_size + batch_size, :, :, :]
#         lb = label[i * batch_size : i * batch_size + batch_size]
#         batched_data.append(b)
#         batched_label.append(lb)
#     return batched_data, batched_label

if __name__ == "__main__":
    main()
