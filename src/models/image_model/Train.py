import argparse
import os
import sys

sys.path.append("../")
import pickle
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import yaml
from addict import Dict
from featDataset import featDataset, imgDataset
from featModel import featModel, imgModel
from stacking import stacking

from evaluater import evaluater

SEED = 14
torch.manual_seed(SEED)


def return_transform():
    transform = transforms.Compose(
        [
            # transforms.ToPILImage(),
            transforms.Resize(255),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return transform


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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    if device == "cuda":
        torch.set_default_tensor_type("torch.cuda.FloatTensor")
    else:
        torch.set_default_tensor_type("torch.FloatTensor")

    if config.type == "feat":
        Traindataset = featDataset(mode="train", feat_model=config.model)
        Testdataset = featDataset(mode="test", feat_model=config.model)
        if config.model == "mobilenet":
            model = featModel(input_channel=1280)
        else:
            model = featModel()
    if config.type == "vgg_stacking":
        model = stacking()

    elif config.type == "img":
        Traindataset = imgDataset(mode="train", transform=return_transform())
        Testdataset = imgDataset(mode="test", transform=return_transform())
        model = imgModel()

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
        scheduler.step()
        print(f"\nlr: {scheduler.get_lr()}")
        t1 = time.time()
        print(f"\ntraining time :{round(t1 - t0)} sec")

        best_eval = test(
            model=model,
            dataset=Testdataset,
            config=config,
            device=device,
            best_eval=best_eval,
        )


def train(model, optimizer, criterion, dataset, config, device, dataset_perm):
    model.train()
    total_loss = 0
    counter = 0
    for i in dataset_perm:
        batch_dataset, batch_label = batch_maker(
            dataset[i], batch_size=config.batch_size, shuffle=config.shuffle
        )
        for data, label in zip(batch_dataset, batch_label):

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
