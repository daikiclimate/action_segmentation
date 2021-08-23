import sys

import numpy as np
import torch
import tqdm
import argparse
from addict import Dict
import yaml

from dataset import return_data
from evaluator import evaluator
from models import build_model

import utils


def get_arg():
    parser = argparse.ArgumentParser(description="image model for action segmentation")
    parser.add_argument("config", type=str, help="path of a config file")
    args = parser.parse_args()
    config = Dict(yaml.safe_load(open(args.config)))
    return config


def main():
    config = get_arg()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, test_set = return_data.return_dataset(config)
    model = build_model.build_model(config)
    model = model.to(device)
    model.load_state_dict(torch.load("model.pth"))
    model.eval()

    for batch_dataset, batch_label in test_set:
        eval = evaluator()
        labels = []
        preds = []
        for data, label in zip(batch_dataset, batch_label):
            data = data.to(device)
            output = model(data)
            output = torch.argmax(output, axis=1)
            preds.extend(output.cpu().detach().numpy())
            labels.extend(np.array(label))
        labels, preds = np.array(labels), np.array(preds)
        from sklearn.metrics import confusion_matrix
        name = utils.return_labels()
        val_mat = confusion_matrix(labels, preds)
        count_labels(labels)
        count_labels(preds)
        # eval.set_data(labels, preds)
        # eval.print_eval(["accuracy"])
        # print(name)
        # print(val_mat)

    


def count_labels(labels):
    num_labels = np.zeros(11, dtype = np.int)
    for l in labels:
        num_labels[l] += 1
    print(num_labels)


if __name__ == "__main__":
    main()
