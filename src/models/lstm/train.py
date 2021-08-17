import argparse
import os
import sys

sys.path.append("../")
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from addict import Dict
from batch_gen import BatchGenerator

import evaluater
import utils
from evaluater import evaluater
from lstm import LSTMclassifier
import tqdm

SEED = 14
torch.manual_seed(SEED)
random.seed(SEED)
torch.cuda.manual_seed_all(SEED)


def get_arg():
    parser = argparse.ArgumentParser(description="video model for action segmentation")
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

    # Traindataset = featDataset(mode="train", feat_model=config.model)
    num_classes = 11
    actions_dict = {
        "opening": 0,
        "moving": 1,
        "hidden": 2,
        "painting": 3,
        "battle": 4,
        "respawn": 5,
        "superjump": 6,
        "object": 7,
        "special": 8,
        "map": 9,
        "ending": 10,
    }
    actions_dict = utils.label_to_id
    gt_path = "../../../data/training/feature_ext/vgg"
    features_path = "../../../data/training/feature_ext/vgg"
    Traindataset = BatchGenerator(num_classes, actions_dict, gt_path, features_path)
    Traindataset.read_data()
    Testdataset = BatchGenerator(num_classes, actions_dict, gt_path, features_path)
    Testdataset.read_data(mode="test")

    num_stages = 2
    num_layers = 2
    num_f_maps = 8
    features_dim = 4
    num_f_maps = 64
    features_dim = 512 * 8 * 8
    # num_f_maps = 512 * 8 * 8
    # features_dim = 2048

    model = LSTMclassifier(1, 1, 256)

    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
    criterion = nn.CrossEntropyLoss()

    best_eval = 0
    for epoch in range(1, 1 + config.epochs):
        print("epoch:", epoch)
        t0 = time.time()
        train(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            dataset=Traindataset,
            config=config,
            device=device,
            # dataset_perm=dataset_perm,
        )
        t1 = time.time()
        scheduler.step()
        print(f"\nlr: {scheduler.get_last_lr()}")
        t1 = time.time()
        print(f"\ntraining time :{round(t1 - t0)} sec")

        best_eval = test(
            model=model,
            dataset=Testdataset,
            config=config,
            device=device,
            best_eval=best_eval,
        )


def train(model, optimizer, criterion, dataset, config, device, dataset_perm=None):
    model.train()
    total_loss = 0
    counter = 0
    print("train")

    while dataset.has_next():
        srcdata, srclabel = dataset.next_batch(config.batch_size)
        bd, bl = batch_maker([srcdata[0], srclabel[0]])
        for data, label in zip(bd, bl):
            data = data.unsqueeze(0)
            label = label.unsqueeze(0)
            data = data.to(device)
            label = label.to(device)
            clf_output = model(data)
            loss_clf = criterion(clf_output, label[0])
            # loss = 0
            # loss += loss_clf
            # loss = loss_clf
            # # for:wqai in range(len(output)):
            #     loss += criterion(output[i], label)

            # backprop
            optimizer.zero_grad()
            # loss.backward()
            loss_clf.backward()
            optimizer.step()

            total_loss += loss_clf.item()
            counter += 1
            print("\r", total_loss / counter, end="")
    dataset.reset()


def test(model, dataset, config, device, best_eval=0, th=0.6):
    model.eval()
    labels = []
    preds = []
    eval = evaluater()
    # eval = evaluater.evaluater()
    # for i in range(len(dataset)):
    while dataset.has_next():
        data, label = dataset.next_batch(config.batch_size)
        # batch_dataset, batch_label = batch_maker(
        #     dataset[i], shuffle=False, drop_last=False, batch_size=config.batch_size
        # )
        # for data, label in zip(batch_dataset, batch_label):
        data = data.to(device)
        output = model(data)
        output = torch.argmax(output, axis=1)
        # output = torch.argmax(output[-1], axis=1)
        # print(label.detach().numpy().shape)
        # print(preds.detach().numpy()[0])

        labels.extend(label.detach().numpy().reshape(-1))
        preds.extend(output.cpu().detach().numpy().reshape(-1))
        # labels.extend(list(label.detach().numpy()[0]))
        # preds.extend(list(output.cpu().detach().numpy()[0]))
    dataset.reset()
    labels, preds = np.array(labels), np.array(preds)
    # print(labels[:5])
    # print(labels.shape)
    # print(preds[:5])
    # with open("test.pkl", "wb") as f:
    # pickle.dump(preds, f)
    eval.set_data(labels, preds)
    eval.print_eval(["accuracy"])
    score = eval.return_eval_score()
    print(score)
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


def batch_maker(dataset, drop_last=True, batch_size=64):
    data, label = dataset[0], dataset[1]
    n = len(data) // batch_size

    batched_data = []
    batched_label = []
    if not drop_last:
        n += 1
    # print(data.shape)
    # exit()
    for i in range(n):
        b = data[i * batch_size : i * batch_size + batch_size, :, :, :]
        l = label[i * batch_size : i * batch_size + batch_size]
        batched_data.append(b)
        batched_label.append(l)
    return batched_data, batched_label


if __name__ == "__main__":
    main()
