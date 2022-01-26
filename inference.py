import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
import tqdm
import yaml
from addict import Dict
from sklearn.metrics import confusion_matrix

import utils
from dataset import return_data
from evaluator import evaluator
from models import build_model


def get_arg():
    parser = argparse.ArgumentParser(description="image model for action segmentation")
    parser.add_argument("config", type=str, help="path of a config file")
    args = parser.parse_args()
    config = Dict(yaml.safe_load(open(args.config)))
    return config


def main():
    # use_mse = True
    use_mse = False
    config = get_arg()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    _, test_set = return_data.return_dataset(config)
    model = build_model.build_model(config)
    config.save_folder = os.path.join(config.save_folder, config.model)
    model = model.to(device)
    use_time = False
    use_time = True
    if use_mse:
        model_path = os.path.join(config.save_folder, f"{config.head}_mse.pth")
    elif use_time:
        model_path = os.path.join(config.save_folder, f"{config.head}_time.pth")
    else:
        model_path = os.path.join(config.save_folder, f"{config.head}.pth")
    model.load_state_dict(torch.load(model_path))
    model.eval()
    eval_total = evaluator()
    total_labels = []
    total_preds = []
    for i, (batch_dataset, batch_label, file_name) in enumerate(test_set):
        print(file_name)
        if file_name == "r21":
            pass
        else:
            continue
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
        name = utils.return_labels()

        use_confusion_mat = False
        if use_confusion_mat == True:
            val_mat = confusion_matrix(labels, preds)
        # count_labels(labels)
        # count_labels(preds)
        eval.set_data(labels, preds)
        eval.print_eval(["accuracy"])
        total_labels.extend(labels)
        total_preds.extend(preds)
        # print(name)
        # print(val_mat)
        continue
        df = pd.DataFrame()
        df[f"pred"] = preds
        df["labels"] = labels

        if use_mse:
            vis_dir = f"visualize_output/{config.model}_{config.head}_mse/"
        elif use_time:
            vis_dir = f"visualize_output/{config.model}_{config.head}_time/"
        else:
            vis_dir = f"visualize_output/{config.model}_{config.head}/"
        if not os.path.exists(vis_dir):
            os.makedirs(vis_dir)

        df.to_csv(vis_dir + f"pred_labels_{file_name}.csv")
        # df.to_csv(vis_dir + f"pred_labels_{file_name}.csv")
    print("\ntotal score")
    eval_total.set_data(total_labels, total_preds)
    eval_total.print_eval(["accuracy"])


def count_labels(labels):
    num_labels = np.zeros(11, dtype=np.int)
    for l in labels:
        num_labels[l] += 1
    print(num_labels)


if __name__ == "__main__":
    main()
